"""Production build: bundles the 5 CUDA engines INTO the wheel.

Precompiled engines are emitted as dotted submodules
(``spikestack_lite._engine.<name>.pyd``) so an installed wheel never needs to
JIT-compile CUDA at import time. Control the CUDA build with environment vars:

    SKIP_CUDA_BUILD=1   force a pure-Python wheel (no CUDA engines bundled);
                        the loader will JIT-fallback at runtime when needed.
    SPIKESTACK_ARCHS=gpu-list   e.g. "8.0 8.6 9.0" to override -gencode archs.

When CUDA_HOME + a C++ toolchain are present, the 5 engines are compiled.
"""
import os
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup

SKIP_CUDA_BUILD = os.environ.get("SKIP_CUDA_BUILD", "0") in ("1", "true", "True")
WHEEL_ARCHS = os.environ.get("SPIKESTACK_ARCHS", "").split()


def _version():
    m = re.search(r'__version__\s*=\s*"([^"]+)"', Path("spikestack_lite/_version.py").read_text("utf-8"))
    return m.group(1) if m else "0.0.0"


ext_modules = []
cmdclass = {}

if not SKIP_CUDA_BUILD:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

        from spikestack_lite._cuda_loader import _CXX_FLAGS, _NVCC_FLAGS, _setup_toolchain

        _setup_toolchain()
        if CUDA_HOME:
            nvcc = _NVCC_FLAGS[:]

            def _norm(a):
                # "8.6" -> "86", "compute_86" -> "compute_86"
                a = a.strip()
                if a.startswith("compute_"):
                    a = a[len("compute_"):]
                return a.replace(".", "")

            if WHEEL_ARCHS:
                nvcc += [f"-gencode=arch=compute_{_norm(a)},code=sm_{_norm(a)}" for a in WHEEL_ARCHS]
            else:
                # Build for the local GPU only (typical `pip install -e .` / dev).
                try:
                    import torch
                    maj, min_ = torch.cuda.get_device_capability(0)
                    nvcc += [f"-gencode=arch=compute_{maj}{min_},code=sm_{maj}{min_}"]
                except Exception:
                    pass
            engines = {
                # module name (MUST match the names passed to load_cuda_extension
                # by sparse/__init__.py, nn/attention.py, nn/gsmc.py,
                # encode/spire.py, nn/exact_head.py) -> source
                "spikeskip_cuda_engine": "spikestack_lite/sparse/src/sparse_linear.cu",
                "gsmc_cuda_engine": "spikestack_lite/nn/src/gsmc_cuda.cu",
                "attention_cuda_engine": "spikestack_lite/nn/src/attention_cuda.cu",
                "spire_cuda_engine": "spikestack_lite/encode/src/spire_cuda.cu",
                "exact_cuda_engine": "spikestack_lite/nn/src/exact_cuda.cu",
            }
            ext_modules = [
                CUDAExtension(
                    name=f"spikestack_lite._engine.{name}",
                    sources=[src],
                    extra_compile_args={"cxx": _CXX_FLAGS, "nvcc": nvcc},
                )
                for name, src in engines.items()
            ]
            cmdclass = {"build_ext": BuildExtension}
        else:
            print("CUDA_HOME not found - building pure-Python wheel "
                  "(engines will JIT-compile at runtime).")
    except Exception as e:  # noqa: BLE001
        print(f"CUDA extension setup skipped ({e}); building pure-Python wheel.")


setup(
    name="spikestack-lite",
    version=_version(),
    description="Production-grade 5-in-1 spiking transformer: Spire + AstroHebbian + SpikeSkip + GSMC + Exact-SNN",
    long_description=(Path("README.md").read_text("utf-8") if Path("README.md").exists() else ""),
    long_description_content_type="text/markdown",
    url="https://github.com/Griffith-7/spikestack-lite",
    packages=find_packages(include=["spikestack_lite", "spikestack_lite.*"]),
    package_data={"spikestack_lite": ["**/*.cu"]},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "torchvision>=0.15"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    include_package_data=True,
)