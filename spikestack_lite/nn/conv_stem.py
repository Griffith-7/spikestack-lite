"""
Convolutional stem for SpikeStack Lite.

CIFAR-10 (and real vision generally) needs spatial inductive bias: an FC-only
SNN caps at ~the 10% chance baseline (the Exact-SNN docs note FC-only nets
cannot learn CIFAR-10 spatial structure). A tiny conv stem maps raw (B,3,32,32)
images into the library's patch/token representation *before* Spire encoding,
providing translation-invariant features without changing the downstream
parallel spiking transformer.

Design:
    conv(3->32,3x3) -> ReLU -> conv(32->32,3x3) -> ReLU -> maxpool(2) ->
    flatten to tokens -> linear -> (B, N, d_model)
The output is a token sequence matching SpikingTransformer's expected
(B, seq_len, d_model) input so the rest of the pipeline is unchanged.
"""

import torch
import torch.nn as nn


class ConvStem(nn.Module):
    """Small convolutional front-end (3,32,32) -> (B, N, d_model) tokens.

    Args:
        in_channels: image channels (3 for RGB).
        d_model: output token feature dimension.
        seq_len: number of output tokens (N). 32x32 with two 3x3 convs + a 2x2
            maxpool yields 8x8=64 spatial tokens -> N=64 by default.
        out_kernels: channels for the two conv layers.
    """

    def __init__(self, in_channels: int = 3, d_model: int = 128, seq_len: int = 64,
                 out_kernels: int = 32):
        super().__init__()
        c1, c2 = out_kernels, out_kernels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),   # 8x8 = 64 tokens
        )
        self.proj = nn.Linear(c2, d_model)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (B, 3, 32, 32)
        f = self.features(x)                       # (B, c2, 8, 8)
        B = f.shape[0]
        tokens = f.flatten(2).transpose(1, 2)      # (B, 64, c2)
        tokens = self.proj(tokens)                 # (B, 64, d_model)
        return tokens
