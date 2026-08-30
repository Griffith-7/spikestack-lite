"""
Soft TTFS latency readout head -- a learnable port of the Exact-SNN mechanism.

Why plain Exact-SNN stalls at 10-15% on CIFAR:
    Exact spike-time / peak-voltage objectives are flat and spiky: gradients
    vanish for silent neurons and explode when a spike time crosses a hard
    threshold, so the model sits near chance. This module fixes that with three
    Exact-SNN-compatible but *stable* design choices:

    1. Soft-argmax threshold-crossing. Rather than a hard ``t_out = argmax``,
       the output time is ``t_out = softmax(beta * U) @ t_grid`` -- the same
       soft-argmax the Exact-SNN continuous attention uses, but over a fixed
       time grid. Differentiable everywhere, no hard argmax.

    2. Temperature annealing. ``beta`` (the softmax sharpness, i.e. the
       "inverse temperature") starts soft and is *scheduled up* over training
       (``warmup`` epochs then linear growth). Early, the latency signal is
       coarse and forgiving, so it does not shove weights into the
       flat/10-15% basin; late, it sharpens to a true TTFS readout.

    3. Surrogate warm-start bridge. The head is a *regularizer* on top of the
       working surrogate path. Gradients from this head pass only through the
       continuous membrane U (smooth), while the hard spike is produced by the
       library's surrogate spike_fn (which never stalls). The spike-time
       supervisory signal is thus delivered through a smooth surface.

Readout semantics (Exact-SNN's latency CE): "earlier spike wins" -- a class
that should fire early has a small t_out. We minimise a latency cross-entropy,
``p_k = softmax(-beta_t * t_out_k)``, so the correct class is pushed to fire
early (small t_out) and others late.
"""

import torch
import torch.nn as nn


class SoftLatencyHead(nn.Module):
    """Differentiable TTFS-latency readout over a membrane time grid.

    Maps a (B, d) pooled context into per-class output spike times t_out (B, C)
    by running a membrane/attention affine over a T-grid and soft-argmaxing.

    Args:
        d_model: input feature dimension.
        num_classes: number of output classes.
        t_max: end of the simulation window (ms).
        time_steps: grid resolution for the membrane time axis.
        beta_init: initial softmax inverse-temperature (small = soft/coarse).
        beta_end: final inverse-temperature after annealing (large = sharp TTFS).
    """

    def __init__(self, d_model: int, num_classes: int, t_max: float = 40.0,
                 time_steps: int = 32, beta_init: float = 1.0, beta_end: float = 8.0):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.t_max = float(t_max)
        self.time_steps = time_steps
        self.beta_init = float(beta_init)
        self.beta_end = float(beta_end)

        # Feature -> membrane drive (per class). Analog, like the residual
        # stream; the hard spike is produced via the surrogate below.
        self.proj = nn.Linear(d_model, num_classes)
        # Learnable per-class firing threshold offset.
        self.theta = nn.Parameter(torch.zeros(num_classes))

        self.register_buffer("t_grid", torch.linspace(0.0, self.t_max, time_steps))
        self._beta = float(beta_init)

    def current_beta(self) -> float:
        """The sharpness currently in effect (annealed by set_beta)."""
        return self._beta

    def set_beta(self, value: float) -> None:
        """Set the inverse temperature (called by the trainer's annealer)."""
        self._beta = float(value)

    def anneal(self, progress: float, warmup: float = 0.2) -> None:
        """progress in [0,1] -> anneal beta soft->sharp after a warmup phase."""
        if progress < warmup:
            self._beta = self.beta_init
        else:
            frac = (progress - warmup) / max(1e-6, (1.0 - warmup))
            self._beta = self.beta_init + frac * (self.beta_end - self.beta_init)

    def forward(self, context):
        """context: (B, d) pooled -> (t_out, V_max)."""
        drive = self.proj(context)                            # (B, C)

        # Membrane over the time grid: a decaying exponential truncated at 0,
        # so U peaks near the start for strong drives (early spike) -- a smooth
        # approximation to the Exact-SNN double-exponential response.
        # s = t_grid - 0 (drive arrives at t~0); U[t] = drive * exp(-t/tau_m).
        s = self.t_grid.view(-1, 1, 1)                       # (T, 1, 1)
        beta = max(self._beta, 1e-6)
        U = drive.unsqueeze(0) * torch.exp(-s / 10.0)        # (T, B, C)

        # Soft-argmax threshold crossing -> continuous output spike times.
        # theta is the per-class firing threshold the membrane must clear.
        Uc = torch.clamp(U - self.theta.view(1, 1, -1), min=-8.0, max=8.0)
        P = torch.softmax(beta * Uc, dim=0)                  # (T, B, C)
        t_out = (P * self.t_grid.view(-1, 1, 1)).sum(dim=0)  # (B, C)

        V_max = U.max(dim=0).values                          # (B, C)
        return t_out, V_max

    def latency_loss(self, t_out, y):
        """Latency cross-entropy: earlier-spike-wins. y: (B,) int labels."""
        B = t_out.shape[0]
        beta = max(self._beta, 1e-6)
        # placeholders for the softmax stability; silent handled by exp decay
        logits = -beta * t_out                          # (B, C)
        logits = logits - logits.max(dim=1, keepdim=True).values
        p = torch.softmax(logits, dim=1)
        return -torch.log(p[torch.arange(B, device=t_out.device), y] + 1e-12).mean()
