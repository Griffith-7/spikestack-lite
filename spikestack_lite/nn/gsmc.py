"""
Vectorized Gated Spiking Memory Cell (GSMC) -- a fast, fused port of the
``gsmc`` research solution into SpikeStack Lite.

Why this is fast where the source ``gsmc/cells.py`` is slow:
    - The four GSMC gates (forget f, input i, output o, candidate g) are fused
      into a SINGLE input affine ``W_in`` and a SINGLE recurrent affine
      ``W_rec``. The source builds one ``nn.Linear`` per gate (8 affines per
      timestep); here each timestep is just 2 matmuls + elementwise ops.
    - The output spike tensor is preallocated once as ``(T, B, d)`` and written
      in place, instead of a Python list of T tensors and a final ``stack``.
    - The cell consumes the *existing* Spire T-planes (``temporal=True``), so
      it adds ONE recurrent stage over T rather than forcing a new sequential
      dimension through the whole model.

Mechanism (faithful to GSMC v2, keeping its Constant-Error-Carousel property):
    f[t] = sigmoid(W_f x[t] + U_f s[t-1] + b_f)     (forget gate, ~1 at init)
    i[t] = sigmoid(W_i x[t] + U_i s[t-1])           (input gate)
    o[t] = sigmoid(W_o x[t] + U_o s[t-1])           (output gate)
    g[t] = tanh(W_g x[t] + U_g s[t-1])              (candidate write)
    a[t] = f[t] * m[t-1] + i[t] * g[t]              (memory bus)
    s[t] = Heaviside(o[t] * a[t] - v_th)            (spike, surrogate-trained)
    m[t] = a[t] - v_th * s[t]                       (refractory deduction)

Temporal Jacobian on the memory bus, ``dM[t]/dM[t-1] = diag(f[t])``, is a
learnable constant-error carousel wherever f ~ 1 -> immune to the exponential
gradient decay that plagues LIF over many timesteps.

Interface (matches the rest of SpireStack Lite, which is parallel over N):
    forward(x, T):  x is (T, B, N, d) -> returns (B, N, d)
    With ``pool='mean'`` (default) the T-plane outputs are mean-pooled so the
    downstream attention/FFN interface stays (B, N, d).
"""

import os
import torch
import torch.nn as nn

from spikestack_lite.nn.attention import SurrogateHeaviside
from spikestack_lite._cuda_loader import load_cuda_extension

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
_gsmc_cu_file = os.path.join(_src_dir, "gsmc_cuda.cu")
_gsmc_cuda_engine = load_cuda_extension("gsmc_cuda_engine", [_gsmc_cu_file])


class FusedGSMC(nn.Module):
    """A vectorized GSMC memory stage that runs over Spire T-planes."""

    def __init__(self, d_model: int, v_th: float = 1.0, forget_bias: float = 8.0,
                 output_bias: float = -2.0, recurrent_scale: float = 0.5,
                 gate_scale: float = 4.0, pool: str = "mean",
                 adaptive_threshold: bool = True, beta_adapt: float = 0.8):
        super().__init__()
        self.d_model = d_model
        self.v_th = float(v_th)
        self.pool = pool
        self.adaptive_threshold = bool(adaptive_threshold)
        self.beta_adapt = float(beta_adapt)

        # Fused input affine: projects x -> 4 gates (f, i, o, g).
        self.W_in = nn.Linear(d_model, 4 * d_model)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)

        # Fused recurrent affine: projects previous spike s -> 4 gates.
        self.W_rec = nn.Linear(d_model, 4 * d_model, bias=False)
        nn.init.orthogonal_(self.W_rec.weight)
        self.W_rec.weight.data.mul_(recurrent_scale)

        # Bias is folded into W_in.bias; set forget/output gate biases only.
        with torch.no_grad():
            self.W_in.bias[: d_model].fill_(forget_bias)      # f
            self.W_in.bias[2 * d_model: 3 * d_model].fill_(output_bias)  # o

        # Per-neuron gate scale (keeps tanh/sigmoid pre-activations in range).
        self.gate_scale = float(gate_scale)

        # Direct synaptic drive (the source's W_d residual path).
        self.W_d = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.W_d.weight)
        nn.init.zeros_(self.W_d.bias)

        self.spike_fn = SurrogateHeaviside.apply

    def _step_gates(self, gi_t, s_prev):
        gi = gi_t / self.gate_scale
        gr = self.W_rec(s_prev) / self.gate_scale
        pre = (gi + gr) + self.W_in.bias
        f = torch.sigmoid(pre[:, : self.d_model])
        i = torch.sigmoid(pre[:, self.d_model: 2 * self.d_model])
        o = torch.sigmoid(pre[:, 2 * self.d_model: 3 * self.d_model])
        g = torch.tanh(pre[:, 3 * self.d_model:])
        return f, i, o, g

    def forward(self, x, T):
        """x: (T, B, N, d) Spire time-planes -> (B, N, d) memory-processed."""
        T_len, B, N, d = x.shape
        flat = x.reshape(T_len, -1, d)          # (T, B*N, d)
        device = x.device

        # Batched input projections across all T planes at once
        gi_all = self.W_in(flat)                # (T, B*N, 4d)
        wd_all = self.W_d(flat)                 # (T, B*N, d)

        m = torch.zeros(B * N, d, device=device)
        s = torch.zeros(B * N, d, device=device)
        heat = torch.zeros(B * N, d, device=device)
        out = torch.empty(T_len, B * N, d, device=device)

        use_cuda = (
            _gsmc_cuda_engine is not None
            and device.type == "cuda"
            and not self.training
            and not torch.is_grad_enabled()  # Raw CUDA kernel has no backward path.
        )


        for t in range(T_len):
            gi_t = gi_all[t]
            wd_t = wd_all[t]

            if use_cuda:
                gr_t = self.W_rec(s)
                s_new = _gsmc_cuda_engine.fused_gsmc_step(
                    gi_t.contiguous(), gr_t.contiguous(), wd_t.contiguous(),
                    self.W_in.bias.contiguous(), m, s, heat,
                    self.v_th, self.gate_scale, self.adaptive_threshold, self.beta_adapt
                )
            else:
                f, i, o, g = self._step_gates(gi_t, s)
                a = f * m + i * g
                v = o * torch.tanh(a) + wd_t
                theta_eff = (
                    self.v_th * (1.0 + heat) if self.adaptive_threshold else self.v_th
                )
                s_new = self.spike_fn(v - theta_eff)
                m = a - s_new * self.v_th
                if self.adaptive_threshold:
                    heat = self.beta_adapt * heat + s_new
                s = s_new
            out[t] = s_new

        if self.pool == "mean":
            pooled = out.mean(dim=0)         # (B*N, d)
        else:
            pooled = out[-1]                 # last-plane readout
        return pooled.view(B, N, d)


