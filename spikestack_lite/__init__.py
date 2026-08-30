"""SpikeStack Lite: a 5-in-1 spiking transformer library.

Bundles all five solutions from the source repository set:
    - Spire       -> dithered multi-spike repetition encoder (``encode.spire``)
    - AstroHebbian-> 100% spiking linear attention (``nn.attention``)
    - SpikeSkip   -> CUDA sparse inference engine (``sparse``)
    - GSMC        -> fused gated spiking memory cell (``nn.gsmc``, opt-in)
    - Exact-SNN   -> soft TTFS latency head (``nn.exact_head``, opt-in)

GSMC and Exact-SNN are opt-in ("mem" and "exact" flags) because GSMC is a
sequential memory stage (adds long-range memory at a controlled cost over the
Spire T-planes) and the Exact-SNN head is a temperature-annealed latency
regularizer trained alongside the surrogate path. The core Spire ->
AstroHebbian(FFN) path stays fully parallel over the sequence.
"""

__version__ = "1.1.0"