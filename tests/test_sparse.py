import pytest
import torch

from spikestack_lite import sparse as S
from spikestack_lite.sparse import SparseLinear

CUDA = torch.cuda.is_available() and S._cuda_engine is not None


def _pair(seed=7, I=1024, O=512):
    torch.manual_seed(seed)
    dense = torch.nn.Linear(I, O).cuda()
    sp = SparseLinear(I, O, bias=True).cuda()
    sp.linear.load_state_dict(dense.state_dict())
    return dense, sp


@pytest.mark.skipif(not CUDA, reason="CUDA + engine required")
@pytest.mark.parametrize("shape", [(303, 1024), (8, 64, 1024)])
def test_forward_matches_dense(shape):
    dense, sp = _pair()
    x = torch.rand(*shape, device="cuda", requires_grad=True)  # non-negative spikes
    y_dense = dense(x).reshape(-1, 512)
    y_sp = sp(x).reshape(-1, 512)
    assert torch.allclose(y_dense, y_sp, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not CUDA, reason="CUDA + engine required")
@pytest.mark.parametrize("shape", [(303, 1024), (8, 64, 1024)])
def test_backward_matches_dense(shape):
    dense, sp = _pair()
    x = torch.rand(*shape, device="cuda", requires_grad=True)
    target = torch.rand(*shape[:-1], 512, device="cuda")

    def grads(lin, w, b):
        loss = ((lin(x) - target) ** 2).mean()
        return torch.autograd.grad(loss, [x, w, b])

    gd = grads(dense, dense.weight, dense.bias)
    gs = grads(sp, sp.linear.weight, sp.linear.bias)
    for d, s in zip(gd, gs):
        assert torch.allclose(d, s, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not CUDA, reason="CUDA + engine required")
def test_engine_actually_engaged():
    sp = SparseLinear(1024, 512).cuda()
    assert sp.in_features >= sp.sparse_threshold
    # forward() must take the sparse route on an engine-capable layer
    x = torch.rand(3, 1024, device="cuda").round()  # binary spikes
    out = sp(x)
    assert out.shape == (3, 512)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_small_layer_uses_dense_path():
    """Below the sparse threshold the module must fall back to plain nn.Linear."""
    sp = SparseLinear(512, 256).cuda()
    x = torch.rand(2, 512, device="cuda")
    out = sp(x)
    assert out.shape == (2, 256)
    assert sp.linear.weight.grad is None  # no backward run yet


def test_dense_fallback_when_no_engine(monkeypatch):
    monkeypatch.setattr(S, "_cuda_engine", None)
    sp = SparseLinear(2048, 512, bias=True, device="cpu")
    x = torch.rand(2, 2048, requires_grad=True)
    out = sp(x)
    assert out.shape == (2, 512)
    assert torch.allclose(out, x @ sp.linear.weight.t() + sp.linear.bias, atol=1e-5)
    out.sum().backward()
    assert sp.linear.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bias_handled_correctly():
    """Bias must land in the output (regression: was once dropped)."""
    dense, sp = _pair()
    x = torch.zeros(2, 1024, device="cuda")  # no spike contribution: out == bias
    assert torch.allclose(sp(x), dense(x), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_input_never_launches_cuda_kernel():
    """Regression: CPU tensors must never reach the CUDA CSR kernel
    (a CPU-input crash used to poison the whole CUDA context)."""
    sp = SparseLinear(1024, 512, bias=True)  # CPU params by default
    x = torch.rand(2, 1024, requires_grad=True)  # CPU tensor
    out = sp(x)
    assert out.shape == (2, 512)
    assert torch.allclose(out, x @ sp.linear.weight.t() + sp.linear.bias, atol=1e-5)
    out.sum().backward()
    assert sp.linear.weight.grad is not None


# ---------------------------------------------------------------------------
# Multi-timestep kernel path (temporal SpikeSkip)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA, reason="CUDA + engine required")
@pytest.mark.parametrize("shape", [(3, 303, 1024), (3, 8, 64, 1024), (1, 500, 1024)])
def test_forward_multistep_matches_dense(shape):
    dense, sp = _pair()
    x = torch.rand(*shape, device="cuda")  # non-negative spikes
    y_dense = dense(x).reshape(-1, 512)
    y_sp = sp.forward_multistep(x).reshape(-1, 512)
    assert torch.allclose(y_dense, y_sp, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not CUDA, reason="CUDA + engine required")
@pytest.mark.parametrize("shape", [(3, 303, 1024), (3, 8, 64, 1024)])
def test_forward_multistep_backward_matches_dense(shape):
    dense, sp = _pair()
    x = torch.rand(*shape, device="cuda", requires_grad=True)
    target = torch.rand(*shape[:-1], 512, device="cuda")

    def grads(fn, w, b):
        loss = ((fn(x) - target) ** 2).mean()
        return torch.autograd.grad(loss, [x, w, b])

    gd = grads(dense, dense.weight, dense.bias)
    gs = grads(sp.forward_multistep, sp.linear.weight, sp.linear.bias)
    for d, s in zip(gd, gs):
        assert torch.allclose(d, s, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not CUDA, reason="CUDA + engine required")
def test_engine_multistep_raw_matches_dense_and_single():
    """Raw multi-step kernel equals dense, and its first timestep equals the
    single-step kernel (verifies buffer reuse across the tight C++ loop)."""
    torch.manual_seed(11)
    B, T, I, O = 257, 3, 2048, 128
    x = torch.rand(T, B, I, device="cuda").round()
    w = torch.randn(O, I, device="cuda")
    nnz, roff, ci, vals = S.alloc_csr_buffers(B, I, device="cuda")

    out = S._cuda_engine.sparse_multistep_forward(
        x.contiguous(), w.t().contiguous(), nnz, roff, ci, vals, T)
    assert torch.allclose(out, x @ w.t(), atol=1e-3, rtol=1e-3)

    out_single = S._cuda_engine.sparse_forward(
        x[0].contiguous(), w.t().contiguous(), nnz, roff, ci, vals)
    assert torch.allclose(out[0], out_single, atol=1e-3, rtol=1e-3)