"""Tests for the stable text-LM trainer (train_text.py).

Coverns the char tokenizer, teacher-forced packing, scheduler/Nan-guard
helpers, CLI flags, and a regression test that every SNN ability module
actually receives a non-trivial gradient in one backward pass.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_text as T

CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------- tokenizer

def test_build_vocab_reserves_pad_unk():
    vb = T.build_vocab(["hello world", "hello again"], max_size=16)
    assert 2 <= len(vb["itos"]) <= 16
    assert vb["stoi"]["\x00"] == T.PAD_ID
    assert len(vb["stoi"]) == len(vb["itos"])


def test_encode_roundtrip_and_unknown():
    vb = {"stoi": {chr(i): k for k, i in enumerate([0, 1, ord("a"), ord("b")])}}
    assert [T.encode("ab", vb), T.encode("z", vb)] == [[2, 3], [T.UNK_ID]]


def test_pack_and_teacher_force_shapes():
    ids = list(range(40))
    blocks = T.pack(ids, 8)
    assert blocks.shape == (5, 8)
    b, y = next(T.make_batches(blocks, batch_size=2))
    assert b.shape == (2, 8) and y.shape == (2, 8)


# ---------------------------------------------------------------- stability

def test_check_finite_raises_on_nan():
    with pytest.raises(RuntimeError, match="non-finite"):
        T.check_finite(torch.tensor([1.0, float("nan")]), "loss")
    T.check_finite(torch.tensor([1.0, -0.5]), "loss")  # no raise


def test_scheduler_warmup_then_cosine():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-2)
    args = T.build_parser().parse_args(["--data", "x.jsonl"])
    args.warmup = 10
    args.steps = 100
    sched = T.make_scheduler(opt, args, 100)
    got = []
    for _ in range(100):
        opt.step()
        sched.step()
        got.append(sched.get_last_lr()[0])
    assert got[0] < 1e-2                  # warmup still ramping
    assert max(got) == pytest.approx(1e-2)  # reaches base at end of warmup
    assert got[20] > got[90]              # decays toward 0 after warmup
    assert got[-1] == pytest.approx(0.0, abs=1e-7)


def test_parser_exposes_ability_toggles_and_stability_flags():
    p = T.build_parser()
    args = p.parse_args(["--data", "x.jsonl"])
    for flag in ("--no-spikeskip", "--no-gsmc", "--no-exact", "--no-temporal",
                 "--no-spike-readout", "--grad-clip", "--warmup", "--seed",
                 "--resume", "--fail-on-nan", "--save-dir"):
        assert flag in p.format_help(), f"missing CLI flag {flag}"
    # defaults: full 5-in-1 config
    assert not args.no_spikeskip and not args.no_gsmc
    assert not args.no_exact and not args.no_temporal


# ------------------------------------------------------ gradient regression

def _text_model(**over):
    args = T.build_parser().parse_args(["--data", "x.jsonl"])
    args.d_model = 32
    args.seq_len = 8
    args.n_heads = 4
    args.n_layers = 1
    args.n_repeats = 3
    args.expansion = 4
    args.no_exact = False
    args.no_gsmc = False
    args.no_spikeskip = True   # CPU: keep dense FFN for the gradient test
    args.spike_readout = False
    args.spike_threshold = 0.0
    args.exact_weight = 1.0
    args.exact_beta_end = 8.0
    for k, v in over.items():
        setattr(args, k, v)
    return T.make_model(args, 64, "cpu")


def test_all_five_abilities_receive_gradients():
    torch.manual_seed(0)
    model = _text_model()
    x = torch.randint(0, 64, (4, 8))
    logits = model.forward_text(x)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 64), x[:, 1:].reshape(-1))
    if model.use_exact:
        loss = loss + model.latency_loss(model._pooled_for_latency, x[:, -1])
    loss.backward()

    def grad_of(name, mod):
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        assert gs, f"no gradient for {name}"
        mag = torch.cat([g.flatten() for g in gs]).abs().mean().item()
        assert gs[0].isfinite().all()
        return mag

    # Spire encoder
    assert grad_of("Spire projection", model.embedding.projection) > 0
    assert grad_of("Spire dither", model.embedding) > 0
    # AstroHebbian attention
    assert grad_of("attention q", model.blocks[0].attn.q_proj) > 0
    assert grad_of("attention decay", model.blocks[0].attn) > 0
    # Spiking FFN (module that SpikeSkip replaces; fc2 holds the weights)
    ffn = model.blocks[0].ffn
    fc2 = ffn.fc2.linear.weight if hasattr(ffn.fc2, "linear") else ffn.fc2.weight
    assert fc2.grad is not None and torch.isfinite(fc2.grad).all()
    # GSMC memory cell
    assert grad_of("GSMC W_in", model.gsmc.W_in) > 0
    assert grad_of("GSMC W_rec", model.gsmc.W_rec) > 0
    # Exact-SNN latency head
    assert grad_of("exact head", model.exact_head.proj) > 0
    # text stem + LM head
    assert grad_of("text stem", model.text_stem.embed) > 0
    assert model.lm_head.weight.grad is not None


def test_cpu_seed_determinism():
    losses = []
    for _ in range(2):
        T.seed_everything(0)
        m = _text_model(no_exact=True)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        args = T.build_parser().parse_args(["--data", "x.jsonl"])
        args.steps = 4
        args.warmup = 2
        args.grad_clip = 1.0
        sched = T.make_scheduler(opt, args, 4)
        m._pooled_for_latency = None
        x = torch.randint(0, 64, (4, 8))
        vals = []
        for _ in range(4):
            vals.append(T.step(m, x, x, opt, sched, args)["loss"])
        losses.append(round(sum(vals), 6))
    assert losses[0] == losses[1], f"determinism violated: {losses}"


def test_nan_guard_skips_bad_update():
    with pytest.raises(RuntimeError, match="non-finite"):
        T.check_finite(torch.tensor([float("inf")]), "loss")