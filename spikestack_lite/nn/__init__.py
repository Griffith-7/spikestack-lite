"""Astrocyte-Hebbian spiking attention building blocks + optional stages."""

from spikestack_lite.nn.attention import (
    AstrocyteHebbianAttention,
    AstrocyteHebbianBlock,
    AstrocyteHebbianClassifier,
    SpikingFFN,
    SurrogateHeaviside,
    spike_fn,
)
from spikestack_lite.nn.gsmc import FusedGSMC
from spikestack_lite.nn.exact_head import SoftLatencyHead
from spikestack_lite.nn.conv_stem import ConvStem

__all__ = [
    "SurrogateHeaviside",
    "spike_fn",
    "AstrocyteHebbianAttention",
    "SpikingFFN",
    "AstrocyteHebbianBlock",
    "AstrocyteHebbianClassifier",
    "FusedGSMC",
    "SoftLatencyHead",
    "ConvStem",
]