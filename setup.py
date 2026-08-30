import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from spikestack_lite._cuda_loader import _setup_toolchain, _CXX_FLAGS, _NVCC_FLAGS

_setup_toolchain()

def _cuda_ext(name, source):
    return CUDAExtension(
        name=name,
        sources=[source],
        extra_compile_args={
            "cxx": _CXX_FLAGS,
            "nvcc": _NVCC_FLAGS,
        },
    )

has_cuda = bool(CUDA_HOME)

ext_modules = [
    _cuda_ext("spikeskip_cuda_engine", "spikestack_lite/sparse/src/sparse_linear.cu"),
    _cuda_ext("gsmc_cuda_engine", "spikestack_lite/nn/src/gsmc_cuda.cu"),
    _cuda_ext("attention_cuda_engine", "spikestack_lite/nn/src/attention_cuda.cu"),
    _cuda_ext("spire_cuda_engine", "spikestack_lite/encode/src/spire_cuda.cu"),
    _cuda_ext("exact_cuda_engine", "spikestack_lite/nn/src/exact_cuda.cu"),
] if has_cuda else []

setup(
    name="spikestack-lite",
    version="0.1.0",
    description="A lightweight 5-in-1 spiking transformer library built for standard GPUs.",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if has_cuda else {},
)

