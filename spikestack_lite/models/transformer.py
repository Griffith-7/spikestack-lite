import torch
import torch.nn as nn

from spikestack_lite.nn.attention import AstrocyteHebbianBlock, SurrogateHeaviside
from spikestack_lite.encode.spire import SpireEncoder

spike_fn = SurrogateHeaviside.apply


class SpikingTransformer(nn.Module):
    """
    The Speed Demon 3-in-1 SNN Model:
    1. Spire (optimal dithered multi-spike repetition encoding)
    2. AstroHebbian (spiking linear attention for context)
    3. SpikeSkip (CUDA sparse engine on the FFN, when the engine is available)

    Design decision (documented deviation from the source repositories):
    the GSMC sequential memory block is deliberately skipped so that all three
    remaining modules run in parallel over the sequence on GPU -- a direct
    spiking replacement for standard Transformers rather than an RNN.

    Architecture notes:
        - SpireEncoder: (B, N, input_dim) -> (B, N, d_model) spike codes.
        - Blocks: AstroHebbian spiking linear self-attention + spiking FFN.
        - Readout: mean-spike-rate pooling over the sequence.
        - ``spike_readout=True`` binarizes the final residual before the
          readout (strictest spiking regime); set to False to read out the
          analog residual directly (ASTROHEBBIAN's original behavior).
    """

    def __init__(self, input_dim=48, d_model=128, seq_len=64, num_classes=10,
                 num_heads=4, v_levels=1, num_layers=1, n_repeats=3,
                 use_spikeskip=False, expansion=None, spike_readout=True,
                 temporal=False, spike_threshold=0.0):
        super().__init__()
        self.d_model = d_model
        self.spike_readout = spike_readout
        self.temporal = bool(temporal)

        # Phase 1: Optimal Encoding (Spire) -- dithered multi-spike repetition code.
        # In temporal mode the m repetition planes become a time axis (T, B, N, d);
        # their average equals the non-temporal graded code exactly.
        self.embedding = SpireEncoder(input_dim, d_model, seq_len, n_repeats=n_repeats,
                                      temporal=temporal)

        # Phase 2: Biological Attention (AstroHebbian), FFN optionally powered
        # by SpikeSkip. If SpikeSkip was requested but its CUDA engine failed to
        # compile, silently disable it here (dense path) rather than crash.
        if use_spikeskip:
            from spikestack_lite import sparse as _sparse
            if _sparse._cuda_engine is None:
                print("[SpikingTransformer] SpikeSkip engine unavailable; "
                      "falling back to dense FFN.")
                use_spikeskip = False
        if expansion is None:
            # SpikeSkip pays off at >=1024-wide inputs; widen the FFN when the
            # sparse engine is active so forward() actually takes the sparse path.
            expansion = 8 if use_spikeskip else 4

        self.blocks = nn.ModuleList([
            AstrocyteHebbianBlock(d_model=d_model, num_heads=num_heads,
                                  expansion=expansion, v_levels=v_levels,
                                  use_spikeskip=use_spikeskip,
                                  spike_threshold=spike_threshold)
            for _ in range(num_layers)
        ])
        self.use_spikeskip = use_spikeskip
        self.spike_threshold = float(spike_threshold)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Step 1: Encoding (Analog -> dithered multi-spike codes via Spire)
        context = self.embedding(x)  # (B, N, d) or (T, B, N, d)

        # Step 2: Attention (AstroHebbian processing on spikes)
        for block in self.blocks:
            context = block(context)

        # Opt-in strict spiking readout: binarize the residual before pooling.
        if self.spike_readout:
            context = spike_fn(context)

        # Readout (mean spike rate over the sequence for classification).
        # Temporal mode pools the mean firing rate across time and sequence,
        # i.e. the graded-rate readout of the full spike train.
        if self.temporal:
            return self.classifier(context.mean(dim=(0, 2)))
        return self.classifier(context.mean(dim=1))