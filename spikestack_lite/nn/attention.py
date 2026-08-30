import os
import torch
import torch.nn as nn

from spikestack_lite._cuda_loader import load_cuda_extension

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
_attn_cu_file = os.path.join(_src_dir, "attention_cuda.cu")
_attn_cuda_engine = load_cuda_extension("attention_cuda_engine", [_attn_cu_file])



class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        alpha = 10.0
        sig = torch.sigmoid(alpha * input)
        return grad_output * sig * (1 - sig) * alpha


spike_fn = SurrogateHeaviside.apply


class _ChannelDecayAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, lam, engine):
        N = k.size(-2)
        pos = torch.arange(N - 1, -1, -1, device=k.device, dtype=torch.float32)
        w = lam.unsqueeze(0) ** pos.unsqueeze(1)
        ctx.save_for_backward(k, lam, w, pos)
        if (
            engine is not None
            and k.device.type == "cuda"
            and k.dtype == torch.float32
            and lam.dtype == torch.float32
        ):
            return engine.apply_channel_decay(k.contiguous(), lam.contiguous())
        return k * w.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_k_w):
        k, lam, w, pos = ctx.saved_tensors
        grad_k = grad_k_w * w.unsqueeze(0)
        pos_dim = pos.unsqueeze(1)
        lam_dim = lam.unsqueeze(0)
        d_w_d_lam = pos_dim * torch.pow(lam_dim, (pos_dim - 1.0).clamp(min=0.0))
        grad_lam = (grad_k_w * k * d_w_d_lam.unsqueeze(0)).sum(dim=tuple(range(grad_k_w.dim() - 2)) + (-2,))
        return grad_k, grad_lam, None


class _NormalizeAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, numer, denom, eps, engine):
        denom_eps = denom + eps
        if (
            engine is not None
            and numer.device.type == "cuda"
            and numer.dtype == torch.float32
            and denom.dtype == torch.float32
        ):
            out = engine.fused_normalize_attention(
                numer.contiguous(), denom.expand_as(numer).contiguous(), float(eps)
            )
        else:
            out = numer / denom_eps
        ctx.save_for_backward(numer, denom_eps, out)
        return out


    @staticmethod
    def backward(ctx, grad_out):
        numer, denom_eps, out = ctx.saved_tensors
        grad_numer = grad_out / denom_eps
        grad_denom = -grad_out * out / denom_eps
        return grad_numer, grad_denom, None, None


class AstrocyteHebbianAttention(nn.Module):
    """Multi-head spiking linear attention with learnable per-channel decay."""

    def __init__(self, d_model=128, num_heads=4, v_levels=1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.L = max(1, int(v_levels))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.register_buffer(
            "tau", torch.arange(1, self.L + 1).float() / (self.L + 1)
        )
        self.decay_logit = nn.Parameter(torch.empty(d_model).uniform_(5.0, 8.0))
        self.eps = 1e-6

    def forward(self, x):
        leading = x.shape[:-1]
        N = int(leading[-1])
        d = x.shape[-1]

        q = spike_fn(self.q_proj(x))
        k = spike_fn(self.k_proj(x))

        u01 = (torch.tanh(self.v_proj(x)) + 1.0) / 2.0
        if self.L == 1:
            v = spike_fn(u01 - self.tau.item())
        else:
            spikes = spike_fn(u01.unsqueeze(-1) - self.tau.view(1, 1, 1, -1))
            v = spikes.mean(dim=-1)

        lam = torch.sigmoid(self.decay_logit)
        k_w = _ChannelDecayAutograd.apply(k, lam, _attn_cuda_engine)

        q = q.view(*leading, self.h, self.dh).transpose(-3, -2)
        k_w = k_w.view(*leading, self.h, self.dh).transpose(-3, -2)
        v = v.view(*leading, self.h, self.dh).transpose(-3, -2)

        kv_trace = torch.matmul(k_w.transpose(-2, -1), v)
        key_mass = k_w.sum(dim=-2, keepdim=True).transpose(-2, -1)

        numer = torch.matmul(q, kv_trace)
        denom = torch.matmul(q, key_mass)

        out = _NormalizeAttentionAutograd.apply(numer, denom, self.eps, _attn_cuda_engine)

        out = out.transpose(-3, -2).contiguous().view(*leading, d)
        return self.o_proj(out)




class SpikingFFN(nn.Module):
    def __init__(self, d_model=128, expansion=4, use_spikeskip=False, spike_threshold=0.0):
        super().__init__()
        self.threshold = float(spike_threshold)
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.use_spikeskip = use_spikeskip
        if self.use_spikeskip:
            from spikestack_lite.sparse import SparseLinear
            self.fc2 = SparseLinear(d_model * expansion, d_model, bias=True)
        else:
            self.fc2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x):
        # Temporal 4D inputs (T, B, N, d) route through the multi-timestep
        # kernel so weights stay hot across the T tight-loop steps.
        act = spike_fn(self.fc1(x) - self.threshold)
        if self.use_spikeskip and x.dim() == 4:
            return self.fc2.forward_multistep(act)
        return self.fc2(act)


class AstrocyteHebbianBlock(nn.Module):
    """Pre-norm Spiking Transformer block. Stackable."""

    def __init__(self, d_model=128, num_heads=4, expansion=4, v_levels=1, use_spikeskip=False,
                 spike_threshold=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AstrocyteHebbianAttention(d_model, num_heads, v_levels)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SpikingFFN(d_model, expansion, use_spikeskip=use_spikeskip,
                              spike_threshold=spike_threshold)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AstrocyteHebbianClassifier(nn.Module):
    def __init__(self, input_dim=1, d_model=128, seq_len=784, num_classes=10,
                 num_layers=1, num_heads=4, v_levels=1, use_spikeskip=False,
                 spike_threshold=0.0):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [AstrocyteHebbianBlock(d_model, num_heads, v_levels=v_levels,
                                   use_spikeskip=use_spikeskip,
                                   spike_threshold=spike_threshold)
             for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        for block in self.blocks:
            x = block(x)
        return self.classifier(x[:, -1, :])
