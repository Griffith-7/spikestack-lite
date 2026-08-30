import torch
import pytest

from spikestack_lite.nn.gsmc import FusedGSMC
from spikestack_lite.nn.attention import AstrocyteHebbianAttention
from spikestack_lite.sparse import SparseLinear


def test_gsmc_forward_and_backward():
    gsmc = FusedGSMC(d_model=64, pool="mean")
    x = torch.randn(3, 2, 16, 64, requires_grad=True)
    out = gsmc(x, T=3)
    assert out.shape == (2, 16, 64)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_attention_forward_and_backward():
    attn = AstrocyteHebbianAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = attn(x)
    assert out.shape == (2, 16, 64)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_sparse_linear_auto_tuning():
    layer = SparseLinear(1024, 512, bias=True)
    x_dense = torch.randn(4, 1024)
    out_dense = layer(x_dense)
    assert out_dense.shape == (4, 512)

    # Sparse tensor (>90% zeros)
    x_sparse = (torch.randn(4, 1024) > 1.5).float()
    out_sparse = layer(x_sparse)
    assert out_sparse.shape == (4, 512)
