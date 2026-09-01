import torch
import torch.nn as nn

from spikestack_lite.nn.attention import AstrocyteHebbianBlock, SurrogateHeaviside
from spikestack_lite.encode.spire import SpireEncoder

spike_fn = SurrogateHeaviside.apply


class SpikingTransformer(nn.Module):
    """
    The 5-in-1 SNN Model built from all the source solutions:
    1. Spire (optimal dithered multi-spike repetition encoding)
    2. AstroHebbian (spiking linear attention for context)
    3. SpikeSkip (CUDA sparse engine on the FFN, when the engine is available)
    4. GSMC (optional fused Gated Spiking Memory Cell -- long-range memory),
       opt-in behind ``use_gsmc`` and only meaningful in temporal mode.
    5. Exact-SNN (optional soft TTFS latency head, ``use_exact``) -- a
       temperature-annealed, differentiable latency readout trained as a
       regularizer alongside the surrogate path so it never stalls at chance.

    Core parallelism: the Spire -> AstroHebbian(FFN) path runs fully in
    parallel over the sequence on GPU. The GSMC memory stage is the ONE
    intentionally sequential piece; it runs over the *existing* Spire
    T-planes (temporal=True) as a single recurrent stage, so it adds long-range
    memory at a controlled, measured cost rather than forcing a sequential
    dimension through the whole model.

    Architecture notes:
        - SpireEncoder: (B, N, input_dim) -> (B, N, d_model) / (T, B, N, d) in
          temporal mode, where the m repetition planes are a time axis.
        - Blocks: AstroHebbian spiking linear self-attention + spiking FFN.
        - GSMC (optional): fused memory stage over the T-planes.
        - Exact-SNN (optional): soft-latency TTFS readout head.
        - Readout: mean-spike-rate pooling over the sequence.
        - ``spike_readout=True`` binarizes the final residual before the
          readout (strictest spiking regime); set to False to read out the
          analog residual directly (ASTROHEBBIAN's original behavior).
    """

    def __init__(self, input_dim=48, d_model=128, seq_len=64, num_classes=10,
                 num_heads=4, v_levels=1, num_layers=1, n_repeats=3,
                 use_spikeskip=False, expansion=None, spike_readout=True,
                 temporal=False, spike_threshold=0.0, use_gsmc=False,
                 gsmc_pool="mean", gsmc_readout=False, use_exact=False,
                 exact_weight=1.0, exact_beta_init=1.0, exact_beta_end=8.0,
                 use_conv_stem=False, conv_kernels=32,
                 text_vocab=None, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.spike_readout = spike_readout
        self.temporal = bool(temporal)
        self.use_gsmc = bool(use_gsmc)
        self.use_exact = bool(use_exact)
        self.use_conv_stem = bool(use_conv_stem)
        self.use_text = text_vocab is not None

        # Phase 0 (optional): convolutional stem. Maps raw (B, 3, 32, 32) images
        # to (B, N, d_model) tokens, giving CIFAR-10 the spatial inductive bias
        # an FC-only SNN lacks (which otherwise stalls near the 10% floor).
        self.conv_stem = None
        if self.use_conv_stem:
            from spikestack_lite.nn.conv_stem import ConvStem
            self.conv_stem = ConvStem(in_channels=3, d_model=d_model,
                                      seq_len=seq_len, out_kernels=conv_kernels)

        # Phase 0b (optional): text stem. Maps raw token ids (B, N) to
        # (B, N, d_model) embeddings, the text analog of conv_stem, so the
        # modality-agnostic spiking core consumes text like any token sequence.
        self.text_stem = None
        if self.use_text:
            from spikestack_lite.nn.text_stem import TextStem
            self.text_stem = TextStem(vocab_size=text_vocab, d_model=d_model,
                                      padding_idx=padding_idx)

        # Phase 1: Optimal Encoding (Spire) -- dithered multi-spike repetition code.
        # In temporal mode the m repetition planes become a time axis (T, B, N, d);
        # their average equals the non-temporal graded code exactly.
        # With a conv or text stem, Spire sees the stem's d_model-wide token features.
        spire_in = d_model if (self.use_conv_stem or self.use_text) else input_dim
        self.embedding = SpireEncoder(spire_in, d_model, seq_len, n_repeats=n_repeats,
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

        # Phase 3 (optional): GSMC memory stage over the T-planes. Sequential by
        # design (the cell's recurrence is what buys long-range memory); gated
        # behind use_gsmc so the fully-parallel default is unchanged.
        self.gsmc = None
        self.gsmc_readout = bool(gsmc_readout)
        if self.use_gsmc:
            if not self.temporal:
                raise ValueError("use_gsmc=True requires temporal=True "
                                 "(GSMC consumes the Spire T-planes).")
            from spikestack_lite.nn.gsmc import FusedGSMC
            self.gsmc = FusedGSMC(d_model=d_model, pool=gsmc_pool)

        self.use_spikeskip = use_spikeskip
        self.spike_threshold = float(spike_threshold)
        self.classifier = nn.Linear(d_model, num_classes)

        # Language-modeling head (optional, only with text_vocab): a token-level
        # d_model -> vocab projection for next-token prediction, sharing the same
        # spiking core as classification but reading out per-token.
        self.lm_head = None
        if self.use_text:
            self.lm_head = nn.Linear(d_model, text_vocab)

        # Phase 4 (optional): Exact-SNN soft-latency head. A differentiable
        # TTFS readout supervised with latency cross-entropy (earlier-spike-
        # wins). Deliberately a *regularizer* paired with the surrogate path so
        # it never stalls at chance like a hard time/voltage objective does.
        self.exact_head = None
        self.exact_weight = float(exact_weight)
        if self.use_exact:
            from spikestack_lite.nn.exact_head import SoftLatencyHead
            self.exact_head = SoftLatencyHead(
                d_model=d_model, num_classes=num_classes,
                beta_init=exact_beta_init, beta_end=exact_beta_end)

    def anneal_exact(self, progress: float, warmup: float = 0.2):
        """Anneal the exact head's sharpness; no-op if exact is disabled."""
        if self.exact_head is not None:
            self.exact_head.anneal(progress, warmup)

    def latency_loss(self, context, y, weight=None):
        """Exact-SNN latency loss from a pre-readout pooled ``context`` (B, d).

        ``context`` must be the *pooled* hidden state computed before the readout
        binarizes/spikes it, so gradients flow through the exact head's smooth
        membrane rather than the Heaviside. Returns 0 if exact is disabled.
        """
        if self.exact_head is None:
            return torch.zeros((), device=context.device)
        t_out, _ = self.exact_head(context)
        loss = self.exact_head.latency_loss(t_out, y)
        return (weight if weight is not None else self.exact_weight) * loss

    def forward(self, x):
        # Step 0 (optional): convolutional stem turns raw images into tokens.
        # When active, x is (B, 3, 32, 32); otherwise x is already (B, N, d).
        if self.conv_stem is not None:
            x = self.conv_stem(x)

        # Step 1: Encoding (Analog -> dithered multi-spike codes via Spire)
        context = self.embedding(x)  # (B, N, d) or (T, B, N, d)

        # Step 2: Attention (AstroHebbian processing on spikes)
        for block in self.blocks:
            context = block(context)

        # Step 3 (optional): GSMC memory stage over the T-planes.
        # In temporal mode `context` is (T, B, N, d); GSMC folds it to (B, N, d)
        # via the T-mean-pooled recurrence output, adding sequential memory.
        if self.gsmc is not None:
            context = self.gsmc(context, T=context.shape[0])

        # Pool the hidden states BEFORE the readout spike binarization so the
        # exact latency head sees a smooth (differentiable) quantity.
        pooled = context.mean(dim=(0, 2)) if (self.temporal and self.gsmc is None) \
            else context.mean(dim=1)
        self._pooled_for_latency = pooled

        # Opt-in strict spiking readout: binarize the residual before pooling.
        if self.spike_readout:
            context = spike_fn(context)

        # Readout (mean spike rate over the sequence for classification).
        # Temporal-with-GSMC: context has been folded to (B, N, d), pool over N.
        if self.temporal and self.gsmc is None:
            return self.classifier(context.mean(dim=(0, 2)))
        return self.classifier(context.mean(dim=1))

    def forward_text(self, x):
        """Next-token prediction over token-id input, reusing all 5 mechanisms.

        x: (B, N) long token ids. Returns logits of shape (B, N, vocab) for
        next-token prediction (shifted targets are handled by the caller).

        The core is identical to ``forward``: Spire encoding -> AstroHebbian
        attention/FFN -> optional GSMC memory -> optional spike readout. The
        only difference is the readout: per-token LM logits instead of the
        mean-pooled classifier, exposing the spiking transformer's abilities
        to a language-modeling task.
        """
        if self.text_stem is None:
            raise RuntimeError("forward_text requires text_vocab to be set "
                               "(SpikingTransformer(text_vocab=...))")
        # Step 0: token ids -> token embeddings.
        x = self.text_stem(x)                      # (B, N, d_model)

        # Step 1: Spire encoding -> dithered multi-spike codes.
        context = self.embedding(x)                # (B, N, d) or (T, B, N, d)

        # Step 2: Attention (AstroHebbian processing on spikes).
        for block in self.blocks:
            context = block(context)

        # Step 3 (optional): GSMC memory stage over the T-planes.
        if self.gsmc is not None:
            context = self.gsmc(context, T=context.shape[0])

        # Pool pre-readout hidden states so the exact latency head (if enabled)
        # sees a smooth, differentiable quantity on this token task.
        if self.temporal and self.gsmc is None:
            self._pooled_for_latency = context.mean(dim=0).mean(dim=1)  # (B, d)
        else:
            self._pooled_for_latency = context.mean(dim=1)              # (B, d)

        # Step 4 (optional): spike readout binarizes the residual.
        if self.spike_readout:
            context = spike_fn(context)

        # Readout: per-token LM logits.
        # Temporal-with-GSMC: context folded to (B, N, d).
        if self.temporal and self.gsmc is None:
            context = context.mean(dim=0)          # fold T -> (B, N, d)
        return self.lm_head(context)               # (B, N, vocab)