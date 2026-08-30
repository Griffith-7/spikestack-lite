"""
Spire Time-Repetition (TR) Encoder.

Faithful port of Project SPIRE's champion mechanism (C1/H2 menu_v5):
analog values are mapped to *dithered multi-spike repetition codes* whose
redundancy knob trades rate for robustness, with all "frontiers" (thresholds)
trainable and surrogate-trained.

Core mechanisms carried over from the SPIRE source repository:
    - C1 coarse-to-fine multi-spike family: ``m`` coincident repetitions
      centered on the nominal latency (``self.dither`` plays the offset menu).
    - H2 champion: time-repetition codes (m>1 coincident spikes) selected by a
      channel-aware objective (here learned end-to-end).
    - "Frontiers" = the R(D)-curve vocabulary; this encoder makes them
      per-channel trainable parameters (``self.frontier``) rather than a fixed
      xed-scalar step threshold.
    - Surrogate thresholding (the step is only ever trained via gradient
      surrogates, matching the L1/L2 channel-layer philosophy).

Shape contract (unchanged from the previous design):
    (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
With ``n_repeats=1`` this reduces exactly to a single binary heaviside step.
With ``n_repeats>1`` the output is a graded multi-spike repetition code in
[0, 1], the time-repetition interpretation of SPIRE.
"""

import torch
import torch.nn as nn

from spikestack_lite.nn.attention import SurrogateHeaviside


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
        # Per-channel trainable "frontier" (init from the flat threshold).
        self.frontier = nn.Parameter(torch.ones(d_model) * threshold)
        # Per-token temporal frontier: a learnable position-wise latency shift.
        self.temporal_frontier = nn.Parameter(torch.randn(1, seq_len, 1) * 0.02)
        # Dithered repetition offsets: the C1 redundancy menu (coincident spikes
        # clustered around the nominal latency when |dither| is small).
        self.dither = nn.Parameter(torch.linspace(-0.05, 0.05, self.n_repeats).view(1, 1, 1, self.n_repeats))
        self.spike_fn = SurrogateHeaviside.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) analog input
        Returns:
            spikes: (batch, seq_len, d_model) repetitions-averaged spike code
        """
        # Analog projection (coarse-to-fine begins with the projection front end).
        u = self.projection(x)
        # Membrane = projection minus per-channel frontier minus temporal frontier.
        v = u - self.frontier.view(1, 1, self.d_model) - self.temporal_frontier
        if self.n_repeats == 1 and not self.temporal:
            return self.spike_fn(v)
        # Multi-spike dithered repetition code: m coincident heaviside steps.
        spikes = self.spike_fn(v.unsqueeze(-1) - self.dither)  # (B, N, d, R)
        if self.temporal:
            # Keep the m repetition planes as a time axis: (R, B, N, d) binary planes.
            return spikes.permute(3, 0, 1, 2)
        # Graded single-plane code: mean over repetitions == temporal average.
        return spikes.mean(dim=-1)