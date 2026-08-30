import torch

from spikestack_lite.encode.spire import SpireEncoder


def test_output_shape():
    enc = SpireEncoder(input_dim=48, d_model=128, seq_len=64, n_repeats=3)
    x = torch.randn(2, 64, 48)
    out = enc(x)
    assert out.shape == (2, 64, 128)


def test_binary_when_single_repeat():
    enc = SpireEncoder(input_dim=48, d_model=128, seq_len=64, n_repeats=1)
    out = enc(torch.randn(4, 64, 48))
    uniq = set(out.unique().tolist())
    assert uniq <= {0.0, 1.0}
    # make sure it actually fires (not all zeros)
    assert out.max().item() > 0.0


def test_graded_code_when_multi_repeat():
    enc = SpireEncoder(input_dim=48, d_model=128, seq_len=64, n_repeats=3)
    out = enc(torch.randn(8, 64, 48))
    allowed = {round(v, 5) for v in (0.0, 1 / 3, 2 / 3, 1.0)}
    uniq = set(round(v, 5) for v in out.unique().tolist())
    assert uniq <= allowed
    # graded (not strictly binary) for random inputs
    assert any(v not in {0.0, 1.0} for v in uniq)


def test_repeats_floor_at_one():
    enc = SpireEncoder(input_dim=16, d_model=32, seq_len=8, n_repeats=0)
    assert enc.n_repeats == 1


def test_learnable_parameters():
    enc = SpireEncoder(input_dim=48, d_model=128, seq_len=64, n_repeats=3)
    for name in ("frontier", "temporal_frontier", "dither"):
        assert isinstance(getattr(enc, name), torch.nn.Parameter)
    assert isinstance(enc.projection.weight, torch.nn.Parameter)
    assert enc.dither.shape == (1, 1, 1, 3)
    assert enc.dither.requires_grad


def test_gradients_reach_encoder_params():
    enc = SpireEncoder(input_dim=48, d_model=128, seq_len=64, n_repeats=3)
    loss = enc(torch.randn(2, 64, 48)).sum()
    loss.backward()
    assert enc.dither.grad is not None
    assert enc.frontier.grad is not None
    assert enc.temporal_frontier.grad is not None
    assert enc.projection.weight.grad is not None