"""Container package for precompiled CUDA engines bundled into wheels.

Production wheels ship the compiled ``*.pyd`` / ``*.so`` engine modules here
(one per CUDA/torch build), so installed packages never need to JIT-compile
the CUDA sources at import time.

In a source checkout this package is empty: the loader falls back to either
the torch JIT-build cache or an on-demand JIT compile.
"""