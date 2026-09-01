"""Parity tests.

Verifies that the CUDA engines agree with their dense PyTorch references and
that spiking transformers behave identically (up to float tolerance) on CPU
vs GPU -- the property that makes SW/HW experiments interchangeable.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from spikestack_lite import sparse as S
from spikestack_lite._cuda_loader import _load_bundled_extension
from spikestack_lite.models.transformer import SpikingTransformer

CUDA = torch.cuda.is_available()
ENGINE = CUDA and S._cuda_engine is not None


# ------------------------------------------------------------- engine parity

@pytest.mark.skipif(not ENGINE, reason="SpikeSkip CUDA engine required")
def test_sparse_kernel_matches_dense_forward():
    from spikestack_lite.sparse import _SparseMatmulAutograd, alloc_csr_buffers
    torch.manual_seed(0)
    B, Fin, Fout = 16, 1024, 256
    # SpikeSkip operates on non-negative spike activations (spike<=0 == silent),
    # so use sparse, non-negative inputs to match the engine's contract.
    flat = torch.zeros(B, Fin, device="cuda")
    nnz = int(Fin * 0.05)
    flat[:, :nnz] = torch.rand(B, nnz, device="cuda") * 0.5
    w = torch.randn(Fout, Fin, device="cuda") * 0.5
    nnz_per_row, row_offsets, col_indices, values = alloc_csr_buffers(B, Fin, "cuda")
    csr = (nnz_per_row, row_offsets, col_indices, values)
    sparse_out = _SparseMatmulAutograd.apply(flat, w, S._cuda_engine, csr)
    dense_out = flat @ w.t()
    assert torch.allclose(sparse_out, dense_out, atol=1e-3, rtol=1e-3), \
        (sparse_out.detach() - dense_out).detach().abs().max().item()


@pytest.mark.skipif(not ENGINE, reason="SpikeSkip CUDA engine required")
def test_sparse_kernel_backward_matches_dense():
    from spikestack_lite.sparse import _SparseMatmulAutograd, alloc_csr_buffers
    torch.manual_seed(1)
    B, Fin, Fout = 8, 1024, 256
    nnz_per_row, row_offsets, col_indices, values = alloc_csr_buffers(B, Fin, "cuda")
    csr = (nnz_per_row, row_offsets, col_indices, values)

    def run(use_engine):
        torch.manual_seed(5)
        x = torch.zeros(B, Fin, device="cuda")
        x[:, :Fin // 20] = torch.rand(B, Fin // 20, device="cuda") + 1.0  # non-neg spikes
        ww = torch.randn(Fout, Fin, device="cuda", requires_grad=True)
        if use_engine:
            out = _SparseMatmulAutograd.apply(x, ww, S._cuda_engine, csr)
        else:
            out = x @ ww.t()
        out.sum().backward()
        return ww.grad.clone()

    g_sparse, g_dense = run(True), run(False)
    assert torch.allclose(g_sparse, g_dense, atol=1e-2, rtol=1e-2), \
        (g_sparse - g_dense).abs().max().item()


# -------------------------------------------------------------- dev parity

def _tiny_model(device, temporal, use_gsmc, use_exact):
    torch.manual_seed(7)
    return SpikingTransformer(
        d_model=24, seq_len=8, num_classes=64, num_heads=4, num_layers=1,
        n_repeats=3, expansion=4, use_spikeskip=False, spike_readout=True,
        temporal=temporal, use_gsmc=use_gsmc, use_exact=use_exact,
        text_vocab=64, padding_idx=0,
    ).to(device)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("temporal,gsmc,exact", [
    (False, False, False),
    (True, True, True),
])
def test_cpu_gpu_parity_logits(temporal, gsmc, exact):
    if gsmc and not temporal:
        pytest.skip("gsmc requires temporal")
    cpu, gpu = _tiny_model("cpu", temporal, gsmc, exact), _tiny_model("cuda", temporal, gsmc, exact)
    gpu.load_state_dict(cpu.state_dict())
    x_cpu = torch.randint(0, 64, (4, 8))
    x_gpu = x_cpu.to("cuda")
    with torch.no_grad():
        l_cpu = cpu.forward_text(x_cpu)
        l_gpu = gpu.forward_text(x_gpu)
    assert l_cpu.shape == l_gpu.shape
    torch.testing.assert_close(l_gpu.cpu(), l_cpu, atol=1e-3, rtol=1e-3)

    # loss parity through the training objective (teacher forcing)
    ce = torch.nn.functional.cross_entropy
    loss_cpu = ce(l_cpu[:, :-1].reshape(-1, 64), x_cpu[:, 1:].reshape(-1))
    loss_gpu = ce(l_gpu[:, :-1].reshape(-1, 64), x_gpu[:, 1:].reshape(-1))
    assert abs(loss_cpu.item() - loss_gpu.item()) < 1e-3


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_engine_available_after_import():
    assert _load_bundled_extension("spikeskip_cuda_engine") is not None