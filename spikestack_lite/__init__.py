"""SpikeStack Lite: a 3-in-1 spiking transformer library.

Bundles three solutions from the source repository set:
    - Spire       -> dithered multi-spike repetition encoder (``encode.spire``)
    - AstroHebbian-> 100% spiking linear attention (``nn.attention``)
    - SpikeSkip   -> CUDA sparse inference engine (``sparse``)

Explicitly NOT included by design: GSMC (sequential memory cell) and
Exact-SNN (exact IFT gradients). See README.md for the rationale.
"""

__version__ = "1.0.0"