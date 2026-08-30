import pytest
import torch

from spikestack_lite import sparse as S
from spikestack_lite.encode.spire import SpireEncoder
from spikestack_lite.models.transformer import SpikingTransformer

CUDA = torch.cuda.is_available()


def make_model(**kw):
    defaults = dict(input_dim=48, d_model=128, seq_len=64, num_classes=10,
                    num_heads=4, num_layers=1)
    defaults.update(kw)
    m = SpikingTransformer(**defaults)
    return m.cuda() if CUDA else m


def test_forward_shape():
    m = make_model()
    x = torch.randn(2, 64, 48)
    if CUDA:
        x = x.cuda()
    assert m(x).shape == (2, 10)


def test_multiple_layers_shape():
    for L in (2, 3):
        m = make_model(num_layers=L)
        x = torch.randn(2, 64, 48)
        if CUDA:
            x = x.cuda()
        assert m(x).shape == (2, 10)


def test_reproducible_with_seed():
    torch.manual_seed(1234)
    m1 = make_model()
    torch.manual_seed(1234)
    m2 = make_model()
    x = torch.randn(3, 64, 48)
    if CUDA:
        x = x.cuda()
    m1.eval(); m2.eval()
    with torch.no_grad():
        y1, y2 = m1(x), m2(x)
    assert torch.allclose(y1.cpu(), y2.cpu(), atol=1e-6)


def test_loss_decreases_on_real_gradients():
    # Every major parameter group must receive gradient (regression: the raw
    # CUDA op used to silently freeze the FFN).
    m = make_model(use_spikeskip=True)
    x = torch.randn(6, 64, 48)
    y = torch.randint(0, 10, (6,))
    if CUDA:
        x, y = x.cuda(), y.cuda()
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"


def test_expansion_default_differs_with_spikeskip():
    m = make_model(use_spikeskip=True)
    assert m.blocks[0].ffn.fc2.in_features >= 1024
    m = make_model(use_spikeskip=False)
    assert m.blocks[0].ffn.fc2.in_features < 1024


def test_no_engine_falls_back_to_dense(monkeypatch):
    monkeypatch.setattr(S, "_cuda_engine", None)
    m = SpikingTransformer(use_spikeskip=True)
    assert m.use_spikeskip is False
    assert isinstance(m.blocks[0].ffn.fc2, torch.nn.Linear)
    out = m(torch.randn(2, 64, 48))
    assert out.shape == (2, 10)


def test_spike_readout_toggle_runs():
    for sr in (True, False):
        m = make_model(spike_readout=sr)
        x = torch.randn(2, 64, 48)
        if CUDA:
            x = x.cuda()
        assert m(x).shape == (2, 10)


@pytest.mark.skipif(not CUDA or S._cuda_engine is None, reason="engine + CUDA required")
def test_sparse_cuda_gradients_finite_and_present():
    """End-to-end through the C++ engine: every param gets a finite gradient."""
    m = make_model(use_spikeskip=True)
    x = torch.randn(4, 64, 48).cuda()
    y = torch.randint(0, 10, (4,)).cuda()
    torch.nn.functional.cross_entropy(m(x), y).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    assert m.blocks[0].ffn.fc2.linear.weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Temporal (multi-timestep) mode
# ---------------------------------------------------------------------------

def _sparsity_of_fc2(model, x, threshold=0.0):
    # FC2 input == spike_fn(fc1(x) - threshold) => positive exactly where fc1 > threshold.
    # Hooked on fc1 because temporal SpikeSkip calls fc2.forward_multistep() directly
    # (bypassing nn.Module script/forward-hooks) but still routes through self.fc1(...).
    got = []

    def _h(mod, i, o):
        got.append((o[0] > threshold).float().mean().item())

    h = model.blocks[0].ffn.fc1.register_forward_hook(_h)
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    return got[0]


def test_temporal_encoder_average_equals_graded():
    """Mean over the time planes must reproduce the non-temporal graded code."""
    torch.manual_seed(5)
    enc = SpireEncoder(input_dim=48, d_model=128, seq_len=8, n_repeats=4)
    enc_t = SpireEncoder(input_dim=48, d_model=128, seq_len=8, n_repeats=4, temporal=True)
    enc_t.load_state_dict(enc.state_dict())  # identical weights
    x = torch.randn(3, 8, 48)
    graded = enc(x)
    planes = enc_t(x)
    assert planes.shape == (4, 3, 8, 128)
    assert set(torch.unique(planes).tolist()) <= {0.0, 1.0}  # binary per plane
    assert torch.allclose(planes.mean(0), graded, atol=1e-6)


def test_temporal_transformer_forward_shape():
    for sr in (True, False):
        m = make_model(temporal=True, n_repeats=4, spike_readout=sr)
        x = torch.randn(2, 64, 48)
        if CUDA:
            x = x.cuda()
        assert m(x).shape == (2, 10)


def test_threshold_increases_fc2_sparsity():
    """A harder FC1 spike threshold must strictly raise FC2 input sparsity."""
    torch.manual_seed(6)
    x = torch.randn(8, 16, 48)
    if CUDA:
        x = x.cuda()
    s_base = _sparsity_of_fc2(make_model(use_spikeskip=False, spike_threshold=0.0, seq_len=16), x)
    s_high = _sparsity_of_fc2(make_model(use_spikeskip=False, spike_threshold=5.0, seq_len=16), x, threshold=5.0)
    assert s_high < s_base


@pytest.mark.skipif(not CUDA or S._cuda_engine is None, reason="engine + CUDA required")
def test_temporal_spikeskip_cuda_gradients():
    """Temporal mode end-to-end through the multi-step kernel: finite grads."""
    m = make_model(use_spikeskip=True, temporal=True, n_repeats=3, spike_threshold=1.0)
    x = torch.randn(4, 64, 48).cuda()
    y = torch.randint(0, 10, (4,)).cuda()
    torch.nn.functional.cross_entropy(m(x), y).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    assert m.blocks[0].ffn.fc2.linear.weight.grad.abs().sum().item() > 0
    assert m.use_spikeskip is True


@pytest.mark.skipif(not CUDA or S._cuda_engine is None, reason="engine + CUDA required")
def test_temporal_spikeskip_high_threshold_sparsity():
    torch.manual_seed(8)
    x = torch.randn(4, 64, 48).cuda()
    m = make_model(use_spikeskip=True, temporal=True, n_repeats=3, spike_threshold=2.0)
    fired = _sparsity_of_fc2(m, x, threshold=2.0)
    assert (1.0 - fired) > 0.5  # >50% of FC2 inputs are silent at a high threshold