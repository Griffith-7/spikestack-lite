"""Astrocyte-Hebbian spiking attention building blocks."""

from spikestack_lite.nn.attention import (
    AstrocyteHebbianAttention,
    AstrocyteHebbianBlock,
    AstrocyteHebbianClassifier,
    SpikingFFN,
    SurrogateHeaviside,
    spike_fn,
)

__all__ = [
    "SurrogateHeaviside",
    "spike_fn",
    "AstrocyteHebbianAttention",
    "SpikingFFN",
    "AstrocyteHebbianBlock",
    "AstrocyteHebbianClassifier",
]