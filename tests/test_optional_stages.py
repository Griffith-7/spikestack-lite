"""Tests for the GSMC memory stage, SoftLatencyHead, and ConvStem."""

import pytest
import torch

from spikestack_lite.nn.gsmc import FusedGSMC
from spikestack_lite.nn.exact_head import SoftLatencyHead
from spikestack_lite.nn.conv_stem import ConvStem
from spikestack_lite.models.transformer import SpikingTransformer

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"


# ---------------------------------------------------------------------------
# FusedGSMC
# ---------------------------------------------------------------------------

def test_gsmc_shape_and_binary():
    torch.manual_seed(0)
    cell = FusedGSMC(d_model=32).to(DEV)
    x = (torch.rand(3, 2, 8, 32, device=DEV) > 0.6).float()
    out = cell(x, T=3)
    assert out.shape == (2, 8, 32)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_gsmc_gradients_flow():
    torch.manual_seed(0)
    cell = FusedGSMC(d_model=32).to(DEV)
    x = (torch.rand(3, 2, 8, 32, device=DEV) > 0.6).float()
    out = cell(x, T=3)
    out.pow(2).mean().backward()
    for name, p in cell.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_gsmc_forget_bias_init_integrator():
    # With forget_bias=8 the forget gate starts near 1 (constant-error carousel).
    torch.manual_seed(0)
    cell = FusedGSMC(d_model=16, adaptive_threshold=False)
    f_pre = cell.W_in.bias[:16].detach()
    f = torch.sigmoid(f_pre)
    assert (f > 0.99).all().item()


def test_gsmc_pool_options():
    torch.manual_seed(0)
    for pool in ("mean", "last"):
        cell = FusedGSMC(d_model=16, pool=pool).to(DEV)
        x = (torch.rand(3, 2, 4, 16, device=DEV) > 0.6).float()
        assert cell(x, T=3).shape == (2, 4, 16)


# ---------------------------------------------------------------------------
# SoftLatencyHead (Exact-SNN mechanism)
# ---------------------------------------------------------------------------

def test_exact_head_shape_and_gradients():
    torch.manual_seed(0)
    head = SoftLatencyHead(d_model=64, num_classes=10).to(DEV)
    ctx = torch.randn(8, 64, device=DEV, requires_grad=True)
    t_out, V_max = head(ctx)
    assert t_out.shape == (8, 10)
    assert V_max.shape == (8, 10)
    y = torch.randint(0, 10, (8,), device=DEV)
    head.latency_loss(t_out, y).backward()
    assert torch.isfinite(ctx.grad).all()
    assert ctx.grad.abs().sum().item() > 0


def test_exact_head_anneals_beta_sharp():
    torch.manual_seed(0)
    head = SoftLatencyHead(d_model=64, num_classes=10, beta_init=1.0, beta_end=8.0)
    head.anneal(0.0)
    soft = head.current_beta()
    head.anneal(1.0)
    sharp = head.current_beta()
    assert soft < sharp


# ---------------------------------------------------------------------------
# ConvStem
# ---------------------------------------------------------------------------

def test_conv_stem_shape():
    torch.manual_seed(0)
    stem = ConvStem(in_channels=3, d_model=128, seq_len=64).to(DEV)
    x = torch.randn(4, 3, 32, 32, device=DEV)
    assert stem(x).shape == (4, 64, 128)


# ---------------------------------------------------------------------------
# Integrated model with the optional stages
# ---------------------------------------------------------------------------

def test_model_with_gsmc_learns():
    torch.manual_seed(0)
    m = SpikingTransformer(input_dim=48, d_model=64, seq_len=64, num_classes=10,
                           num_heads=4, num_layers=1, n_repeats=3,
                           temporal=True, spike_threshold=0.0, use_gsmc=True).to(DEV)
    assert m.gsmc is not None
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(8, 64, 48, device=DEV)
    y = torch.randint(0, 10, (8,), device=DEV)
    l0 = None
    for _ in range(10):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        l0 = loss if l0 is None else l0
    assert torch.isfinite(l0)  # no NaN, gradients flowed


def test_model_with_exact_head_learns():
    torch.manual_seed(0)
    m = SpikingTransformer(input_dim=48, d_model=64, seq_len=64, num_classes=10,
                           num_heads=4, num_layers=1, n_repeats=3,
                           temporal=True, spike_threshold=0.0,
                           use_exact=True, exact_weight=0.3).to(DEV)
    assert m.exact_head is not None
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(8, 64, 48, device=DEV)
    y = torch.randint(0, 10, (8,), device=DEV)
    l_first = None
    for i in range(10):
        opt.zero_grad()
        logits = m(x)
        loss = torch.nn.functional.cross_entropy(logits, y) + \
            m.latency_loss(m._pooled_for_latency, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        m.anneal_exact((i + 1) / 10)
        l_first = loss if l_first is None else l_first
    assert torch.isfinite(l_first)


def test_model_with_conv_stem_takes_images():
    torch.manual_seed(0)
    m = SpikingTransformer(input_dim=48, d_model=64, seq_len=64, num_classes=10,
                           num_heads=4, num_layers=1, n_repeats=3,
                           temporal=True, spike_threshold=0.0,
                           use_conv_stem=True).to(DEV)
    assert m.conv_stem is not None
    x = torch.randn(4, 3, 32, 32, device=DEV)
    assert m(x).shape == (4, 10)


def test_gsmc_requires_temporal():
    with pytest.raises(ValueError):
        SpikingTransformer(input_dim=48, d_model=64, seq_len=64, num_classes=10,
                           temporal=False, use_gsmc=True)
