"""
Stable next-token text-language-modeling trainer for SpikeStack Lite.

Exposes every SNN ability behind a flag (defaults to the full 5-in-1 config):

    base        : --no-spikeskip --no-gsmc --no-exact --no-temporal --no-spike-readout
    spiking-core: --no-spikeskip --no-gsmc --no-exact
    all-5       : (defaults)         Spire + AstroHebbian + SpikeSkip + GSMC + Exact-SNN

Stability defaults (research-grade A/B runs diverge without them):
    - gradient clipping (--grad-clip)
    - LR warmup + cosine decay (--warmup)
    - NaN/inf guards that skip bad steps instead of corrupting the weights
    - full seed determinism (--seed) and manager-compatible checkpoints (--save-dir)

Usage:
    python train_text.py --data /path/to/corpus.jsonl --steps 2000 \
        --d-model 128 --n-layers 3 --seq-len 128
    python train_text.py --data ... --no-spikeskip --no-gsmc --no-exact   # spiking core
    python train_text.py --data ... --resume runs/run-1/checkpoints/last.pt
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from spikestack_lite.models.transformer import SpikingTransformer

PAD_ID = 0
UNK_ID = 1
_NAN_RETRIES = 5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab(texts, max_size: int = 256) -> dict:
    """Char-level vocab from the most frequent characters in ``texts``.

    PAD=0, UNK=1 always exist; ``max_size`` is the total vocab size.
    """
    counts: dict = {}
    for t in texts:
        for ch in t:
            counts[ch] = counts.get(ch, 0) + 1
    ordered = [PAD_ID, UNK_ID] + [ord(c) for c, _ in
                                  sorted(counts.items(), key=lambda kv: -kv[1])]
    ordered = list(dict.fromkeys(ordered))  # dedupe PAD/UNK if actual chars
    vocab = ordered[: max(2, max_size)]
    return {"itos": [chr(i) for i in vocab], "stoi": {chr(i): k for k, i in enumerate(vocab)}}


def encode(text: str, vocab: dict) -> list:
    return [vocab["stoi"].get(ch, UNK_ID) for ch in text]


def read_jsonl(path: str, limit: int = 50000, max_chars: int = 2_000_000) -> list:
    """Read JSON-lines corpus entries into a list of plain strings.

    Tries to pull a text-ish field out of each JSON object, falling back to the
    raw line. Stops once ``max_chars`` characters have been accumulated.
    """
    texts, total = [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i >= limit or total >= max_chars:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text") or obj.get("content") or obj.get("code") or ""
            except json.JSONDecodeError:
                txt = line
            if not txt:
                continue
            texts.append(str(txt))
            total += len(txt)
    return texts


def pack(ids, seq_len: int) -> torch.Tensor:
    """Chunk a flat token list into (num_blocks, seq_len) long tensor."""
    total = (len(ids) // seq_len) * seq_len
    return torch.tensor(ids[:total], dtype=torch.long).view(-1, seq_len)


def make_batches(blocks: torch.Tensor, batch_size: int, device=None):
    """Yield (x, y) teacher-forced batches from (num_blocks, seq_len) tensor.

    x = full block; next-token targets = the same block shifted left by 1,
    applied by the caller over logits[:, :-1] vs ids[:, 1:].
    """
    order = torch.randperm(blocks.size(0), device=blocks.device)
    blocks = blocks[order]
    for i in range(0, blocks.size(0), batch_size):
        b = blocks[i: i + batch_size]
        if b.size(0) < 2:
            continue
        if device is not None:
            b = b.to(device)
        yield b, b


def make_model(args, vocab_size: int, device):
    temporal = not args.no_temporal
    use_gsmc = (not args.no_gsmc) and temporal
    if (not args.no_gsmc) and not temporal:
        print("NOTE: --no-temporal and GSMC are incompatible; GSMC disabled.")

    model = SpikingTransformer(
        d_model=args.d_model,
        seq_len=args.seq_len,
        num_classes=vocab_size,
        num_heads=args.n_heads,
        v_levels=1,
        num_layers=args.n_layers,
        n_repeats=args.n_repeats,
        expansion=args.expansion,
        use_spikeskip=not args.no_spikeskip,
        spike_readout=not args.no_spike_readout,
        temporal=temporal,
        spike_threshold=args.spike_threshold,
        use_gsmc=use_gsmc,
        gsmc_pool="mean",
        gsmc_readout=False,
        use_exact=not args.no_exact,
        exact_weight=args.exact_weight,
        exact_beta_init=1.0,
        exact_beta_end=args.exact_beta_end,
        text_vocab=vocab_size,
        padding_idx=PAD_ID,
    ).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model: d={args.d_model} layers={args.n_layers} heads={args.n_heads} "
          f"repeats={args.n_repeats} expansion={args.expansion} params={nparam/1e6:.2f}M")
    print(f"abilities: temporal={temporal} spikeskip={not args.no_spikeskip} "
          f"gsmc={use_gsmc} exact={not args.no_exact} "
          f"spike_readout={not args.no_spike_readout}")
    return model


def check_finite(x: torch.Tensor, what: str):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"non-finite {what}: {x.detach().abs().max().item():.3e}")


def make_scheduler(optimizer, args, steps: int):
    def scale(step: int):
        if step >= steps:
            return 0.0
        if step < args.warmup:
            return float(step + 1) / max(1, args.warmup)
        frac = (step - args.warmup) / max(1e-6, (steps - args.warmup))
        return 0.5 * (1.0 + float(torch.cos(torch.tensor(frac * 3.14159265))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def step(model, ids, batch, opt, sched, args, step_i: int = 0):
    """One optimizer step. Returns a metrics dict; ``None`` result means the
    step was skipped (non-finite loss) and the caller should continue."""
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model.forward_text(batch)                 # (B, N, vocab)
    # Teacher-forced next-token prediction: logits[..., :-1] -> ids[..., 1:].
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, model.lm_head.out_features),
                           ids[:, 1:].reshape(-1))
    with torch.no_grad():
        acc = (logits[:, :-1].argmax(-1) == ids[:, 1:]).float().mean().item()

    exact = torch.zeros((), device=logits.device)
    if model.use_exact:
        exact = model.latency_loss(model._pooled_for_latency, ids[:, -1])
        model.anneal_exact(step_i / max(1, args.steps), warmup=0.1)
        loss = loss + exact

    check_finite(loss, "loss")
    loss.backward()
    if len(ids) >= args.grad_clip > 0:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_norm = torch.cat([g.flatten() for g in grads]).norm().item()
    else:
        grad_norm = float("nan")
    if args.grad_clip > 0:
        nn_utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    sched.step()
    return {
        "loss": loss.item(),
        "ce": (loss - exact).item(),
        "exact": exact.item(),
        "acc": acc,
        "lr": opt.param_groups[0]["lr"],
        "grad_norm": grad_norm,
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="path to .jsonl corpus")
    ap.add_argument("--limit", type=int, default=50000, help="max lines to read")
    ap.add_argument("--max-chars", type=int, default=2_000_000)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--spike-threshold", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    ap.add_argument("--exact-weight", type=float, default=0.1,
                    help="weight on the Exact-SNN latency regularizer. The exact "
                         "head is trained as a *small* regularizer beside the "
                         "surrogate CE path; weights >= ~0.5 let its gradient "
                         "dominate and destabilize training (kept low by default).")
    ap.add_argument("--exact-beta-end", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exp-name", default="text-lm")
    ap.add_argument("--save-dir", default="runs")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", default=None, help="path to a .pt checkpoint")
    ap.add_argument("--fail-on-nan", action="store_true",
                    help="abort instead of skipping non-finite steps")
    ap.add_argument("--no-spikeskip", action="store_true")
    ap.add_argument("--no-gsmc", action="store_true")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--no-spike-readout", action="store_true")
    return ap


def save_ckpt(model, opt, sched, args, step_i, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step_i,
        "args": vars(args),
    }, path)


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    texts = read_jsonl(args.data, args.limit, args.max_chars)
    if not texts:
        sys.exit(f"no data read from {args.data}")
    vocab = build_vocab(texts, args.vocab_size)
    ids = [t for text in texts for t in encode(text, vocab)][: args.steps * args.batch_size * args.seq_len * 2]
    blocks = pack(ids, args.seq_len)
    print(f"data: {len(texts)} docs, {len(ids)} chars, {blocks.size(0)} blocks, "
          f"vocab={len(vocab['itos'])}")

    model = make_model(args, len(vocab["itos"]), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = make_scheduler(opt, args, args.steps)

    step_i = 0
    ckpt_dir = os.path.join(args.save_dir, args.exp_name, "checkpoints")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step_i = ck["step"]
        print(f"resumed from {args.resume} at step {step_i}")

    print(f"ready on {device}\nstep\tloss\tce\texact\tacc\tlr\tgrad_norm\tchar/s")
    losses = []
    start = time.perf_counter()
    nan_run = 0
    while step_i < args.steps:
        for batch, batch_ids in make_batches(blocks, args.batch_size, device):
            if step_i >= args.steps:
                break
            if nan_run >= _NAN_RETRIES:
                sys.exit(f"aborting: {_NAN_RETRIES} non-finite steps in a row")
            try:
                m = step(model, batch_ids, batch, opt, sched, args, step_i)
            except RuntimeError as e:
                if "non-finite" in str(e) and not args.fail_on_nan:
                    print(f"[step {step_i}] non-finite loss; skipping step "
                          f"(nan_run={nan_run + 1})")
                    opt.zero_grad(set_to_none=True)
                    nan_run += 1
                    continue
                raise
            nan_run = 0
            losses.append(m["loss"])
            if step_i % 10 == 0:
                elapsed = max(1e-6, time.perf_counter() - start)
                print(f"{step_i}\t{m['loss']:.4f}\t{m['ce']:.4f}\t{m['exact']:.4f}\t"
                      f"{m['acc']:.4f}\t{m['lr']:.2e}\t{m['grad_norm']:.3f}\t"
                      f"{len(ids) / elapsed:.0f}")
            if args.ckpt_every and (step_i + 1) % args.ckpt_every == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                save_ckpt(model, opt, sched, args, step_i + 1,
                          os.path.join(ckpt_dir, "last.pt"))
            step_i += 1

    if args.ckpt_every:
        os.makedirs(ckpt_dir, exist_ok=True)
        save_ckpt(model, opt, sched, args, step_i, os.path.join(ckpt_dir, "last.pt"))
    mean = sum(losses) / max(1, len(losses))
    print(f"\ndone in {time.perf_counter() - start:.1f}s | mean loss {mean:.4f} | "
          f"final acc ... (see last step column)")


if __name__ == "__main__":
    main()