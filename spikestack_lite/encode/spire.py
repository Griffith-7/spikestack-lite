import os
import torch
import torch.nn as nn

from spikestack_lite.nn.attention import SurrogateHeaviside

from spikestack_lite._cuda_loader import load_cuda_extension

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
_spire_cu_file = os.path.join(_src_dir, "spire_cuda.cu")
_spire_cuda_engine = load_cuda_extension("spire_cuda_engine", [_spire_cu_file])



class _SpireDitherAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, dither, engine):
        ctx.save_for_backward(v, dither)
        if engine is not None and v.device.type == "cuda":
            return engine.spire_dither(v, dither.view(-1))
        # PyTorch fallback
        spike_fn = SurrogateHeaviside.apply
        spikes = spike_fn(v.unsqueeze(-1) - dither)
        return spikes.permute(3, 0, 1, 2)

    @staticmethod
    def backward(ctx, grad_out):
        v, dither = ctx.saved_tensors
        # grad_out: (R, B, N, d) -> permute back to (B, N, d, R)
        grad_perm = grad_out.permute(1, 2, 3, 0)
        diff = v.unsqueeze(-1) - dither
        alpha = 10.0
        sig = torch.sigmoid(alpha * diff)
        surr = sig * (1.0 - sig) * alpha
        grad_v = (grad_perm * surr).sum(dim=-1)
        grad_dither = -(grad_perm * surr).sum(dim=(0, 1, 2)).view_as(dither)
        return grad_v, grad_dither, None


class SpireEncoder(nn.Module):
    """
    Spire Time-Repetition (TR) Encoder.

    Translates continuous analog data into (multi-)spike codes using trainable
    per-channel projection frontiers and surrogate thresholding.
    """

    def __init__(self, input_dim: int, d_model: int, seq_len: int = 784, n_repeats: int = 3, threshold: float = 0.5,
                 temporal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_repeats = max(1, int(n_repeats))
        self.temporal = bool(temporal)
        self.projection = nn.Linear(input_dim, d_model)
        self.frontier = nn.Parameter(torch.ones(d_model) * threshold)
        self.temporal_frontier = nn.Parameter(torch.randn(1, seq_len, 1) * 0.02)
        self.dither = nn.Parameter(torch.linspace(-0.05, 0.05, self.n_repeats).view(1, 1, 1, self.n_repeats))
        self.spike_fn = SurrogateHeaviside.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.projection(x)
        v = (u - self.frontier.view(1, 1, self.d_model) - self.temporal_frontier).float()
        if self.n_repeats == 1 and not self.temporal:
            return self.spike_fn(v)
        spikes = _SpireDitherAutograd.apply(v, self.dither.float(), _spire_cuda_engine)

        if self.temporal:
            return spikes
        return spikes.mean(dim=0)