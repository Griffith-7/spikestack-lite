"""
Text stem for SpikeStack Lite.

The library's core (Spire -> AstroHebbian attention + FFN, GSMC, SpikeSkip,
Exact-SNN) is modality-agnostic: it consumes token sequences (B, seq_len, d).
The optional conv_stem feeds raw images in; this text stem feeds raw text in by
converting token ids into the same (B, N, d_model) embedding space.

This is the text analog of ``conv_stem``: a small learnable token embedding that
maps integer token ids (from any tokenizer/BPE) to dense vectors, so the rest of
the parallel spiking transformer is unchanged.
"""

import torch
import torch.nn as nn


class TextStem(nn.Module):
    """Token-id -> (B, N, d_model) embedding front-end.

    Args:
        vocab_size: number of distinct tokens in the tokenizer vocabulary.
        d_model: output token feature dimension (must match the transformer).
        padding_idx: optional index used for padding (its embedding stays zero).
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.padding_idx = int(padding_idx)
        self.embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.padding_idx)

    def forward(self, x):
        # x: (B, N) long token ids -> (B, N, d_model)
        return self.embed(x)
