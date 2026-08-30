import os
import torch
import torch.nn as nn

from spikestack_lite._cuda_loader import load_cuda_extension

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
_exact_cu_file = os.path.join(_src_dir, "exact_cuda.cu")
_exact_cuda_engine = load_cuda_extension("exact_cuda_engine", [_exact_cu_file])



class _ExactHeadAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, drive, theta, t_grid, beta, engine):
        if engine is not None and drive.device.type == "cuda":
            res = engine.fused_exact_head(drive, theta, t_grid, float(beta))
            t_out, v_max = res[0], res[1]
        else:
            s = t_grid.view(-1, 1, 1)
            U = drive.unsqueeze(0) * torch.exp(-s / 10.0)
            Uc = torch.clamp(U - theta.view(1, 1, -1), min=-8.0, max=8.0)
            P = torch.softmax(beta * Uc, dim=0)
            t_out = (P * t_grid.view(-1, 1, 1)).sum(dim=0)
            v_max = U.max(dim=0).values
        ctx.save_for_backward(drive, theta, t_grid)
        ctx.beta = beta
        return t_out, v_max

    @staticmethod
    def backward(ctx, grad_t_out, grad_v_max):
        drive, theta, t_grid = ctx.saved_tensors
        beta = ctx.beta
        s = t_grid.view(-1, 1, 1)
        decay = torch.exp(-s / 10.0)
        U = drive.unsqueeze(0) * decay
        Uc = torch.clamp(U - theta.view(1, 1, -1), min=-8.0, max=8.0)
        P = torch.softmax(beta * Uc, dim=0)
        t_out_ref = (P * t_grid.view(-1, 1, 1)).sum(dim=0)
        # d(t_out)/d(Uc) = beta * P * (t_grid - t_out)
        # d(Uc)/d(drive) = exp(-s / 10.0)
        # d(Uc)/d(theta) = -1.0
        diff = t_grid.view(-1, 1, 1) - t_out_ref.unsqueeze(0)
        grad_Uc = grad_t_out.unsqueeze(0) * beta * P * diff
        grad_drive = (grad_Uc * decay).sum(dim=0)
        if grad_v_max is not None:
            # v_max = max_t drive * exp(-t / 10); route the gradient to the argmax step.
            arg = U.argmax(dim=0, keepdim=True)
            sel = torch.zeros_like(U).scatter_(0, arg, 1.0)
        grad_theta = -grad_Uc.sum(dim=(0, 1))
        return grad_drive, grad_theta, None, None, None



class SoftLatencyHead(nn.Module):
    """Differentiable TTFS-latency readout over a membrane time grid."""

    def __init__(self, d_model: int, num_classes: int, t_max: float = 40.0,
                 time_steps: int = 32, beta_init: float = 1.0, beta_end: float = 8.0):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.t_max = float(t_max)
        self.time_steps = time_steps
        self.beta_init = float(beta_init)
        self.beta_end = float(beta_end)

        self.proj = nn.Linear(d_model, num_classes)
        self.theta = nn.Parameter(torch.zeros(num_classes))

        self.register_buffer("t_grid", torch.linspace(0.0, self.t_max, time_steps))
        self._beta = float(beta_init)

    def current_beta(self) -> float:
        return self._beta

    def set_beta(self, value: float) -> None:
        self._beta = float(value)

    def anneal(self, progress: float, warmup: float = 0.2) -> None:
        if progress < warmup:
            self._beta = self.beta_init
        else:
            frac = (progress - warmup) / max(1e-6, (1.0 - warmup))
            self._beta = self.beta_init + frac * (self.beta_end - self.beta_init)

    def forward(self, context):
        drive = self.proj(context).float()
        return _ExactHeadAutograd.apply(drive, self.theta.float(), self.t_grid.float(), self._beta, _exact_cuda_engine)


    def latency_loss(self, t_out, y):
        beta = max(self._beta, 1e-6)
        logits = -beta * t_out
        logits = logits - logits.max(dim=1, keepdim=True).values
        p = torch.softmax(logits, dim=1)
        return -torch.log(p[torch.arange(t_out.shape[0], device=t_out.device), y] + 1e-12).mean()

