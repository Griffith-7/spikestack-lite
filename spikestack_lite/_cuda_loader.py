import os
import sys
import glob
import warnings

_IS_WIN = sys.platform == "win32"
_CXX_FLAGS = ["/O2", "/Zc:preprocessor"] if _IS_WIN else ["-O3"]
_NVCC_FLAGS = ["-O3", "-Xcompiler", "/Zc:preprocessor"] if _IS_WIN else ["-O3"]


def _setup_toolchain():
    if not _IS_WIN:
        return
    if "CUDA_HOME" not in os.environ:
        for cand_i in [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        ]:
            if os.path.exists(cand_i):
                os.environ["CUDA_HOME"] = cand_i
                break

    cuda_bin = os.path.join(os.environ.get("CUDA_HOME", ""), "bin")
    if os.path.exists(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

    for vs_root in [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ]:
        if not os.path.exists(vs_root):
            continue
        for cl in glob.glob(
            os.path.join(vs_root, "*", "*", "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe")
        ):
            cl_dir = os.path.dirname(cl)
            if cl_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = cl_dir + os.pathsep + os.environ.get("PATH", "")
            return


def load_cuda_extension(name, sources, build_directory=None):
    """Load a CUDA extension with cross-platform compiler flags."""
    try:
        import torch
        from torch.utils.cpp_extension import load, CUDA_HOME
        if not torch.cuda.is_available() or CUDA_HOME is None:
            return None

        _setup_toolchain()
        kwargs = {
            "name": name,
            "sources": sources,
            "extra_cflags": _CXX_FLAGS,
            "extra_cuda_cflags": _NVCC_FLAGS,
            "verbose": False,
        }
        if build_directory is not None:
            kwargs["build_directory"] = build_directory
        return load(**kwargs)
    except Exception as e:
        warnings.warn(f"Failed to load CUDA extension '{name}': {e}. Falling back to PyTorch implementation.")
        return None
