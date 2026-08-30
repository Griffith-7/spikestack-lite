import torch

from spikestack_lite.nn.attention import (
    AstrocyteHebbianAttention,
    AstrocyteHebbianBlock,
    SpikingFFN,
    SurrogateHeaviside,
    spike_fn,
)


def test_surrogate_forward_is_binary():
    x = torch.randn(200)
    y = SurrogateHeaviside.apply(x)
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_surrogate_backward_nonzero():
    x = torch.randn(200, requires_grad=True)
    loss = SurrogateHeaviside.apply(x).mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0.0


def test_attention_shape_preserved():
    attn = AstrocyteHebbianAttention(d_model=128, num_heads=4)
    x = torch.randn(2, 64, 128)
    assert attn(x).shape == (2, 64, 128)


def test_attention_keeps_gradients():
    attn = AstrocyteHebbianAttention(d_model=128, num_heads=4)
    x = torch.randn(2, 64, 128)
    attn(x).sum().backward()
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"


def test_ffn_dense_shape():
    ffn = SpikingFFN(d_model=128, expansion=4, use_spikeskip=False)
    x = torch.randn(2, 64, 128)
    out = ffn(x)
    assert out.shape == (2, 64, 128)
    assert isinstance(ffn.fc2, torch.nn.Linear)


def test_ffn_spikeskip_shape():
    from spikestack_lite.sparse import SparseLinear

    ffn = SpikingFFN(d_model=128, expansion=8, use_spikeskip=True)
    x = torch.randn(2, 64, 128)
    out = ffn(x)
    assert out.shape == (2, 64, 128)
    assert isinstance(ffn.fc2, SparseLinear)


def test_ffn_hidden_is_binary_spike():
    ffn = SpikingFFN(d_model=128, expansion=4, use_spikeskip=False)
    z = []
    # fc2's INPUT is the spiked hidden activation of fc1 (after spike_fn)
    def hook(mod, inp, out):
        z.append(inp[0].detach())
    h = ffn.fc2.register_forward_hook(hook)
    ffn(torch.randn(1, 8, 128) * 0.1)
    h.remove()
    assert z and set(z[0].unique().tolist()) <= {0.0, 1.0}


def test_block_stackable_shape():
    x = torch.randn(2, 64, 128)
    nb = AstrocyteHebbianBlock(d_model=128, num_heads=4)
    for _ in range(3):
        x = nb(x)
    assert x.shape == (2, 64, 128)