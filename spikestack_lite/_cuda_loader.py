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


def _cached_extension_dir(name):
    """Return the torch JIT cache dir holding a previously-built engine, if any."""
    try:
        import torch
        from torch.utils.cpp_extension import _get_build_directory
        return _get_build_directory(name, verbose=False)
    except Exception:
        return None


def _load_cached_extension(name):
    """Fast-path: load a previously-built .pyd/.so from torch's JIT cache.

    Calling torch.utils.cpp_extension.load() on every import re-hashes the
    source files and, when the .cu sources are newer than the cached build
    (e.g. right after a fresh git clone), triggers a full nvcc/cl rebuild that
    can hang silently on Windows when the toolchain env isn't fully set up.
    Here we reuse the already-built engine directly, setting up torch's DLL
    search paths, which is instant and avoids any recompilation.
    """
    try:
        import torch
        from torch.utils.cpp_extension import _import_module_from_library
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    build_dir = _cached_extension_dir(name)
    if build_dir is None:
        return None
    # The built artifact is <name>.pyd (Windows) / <name>.so (Linux).
    if not any(
        os.path.exists(os.path.join(build_dir, f))
        for f in (name + ".pyd", name + ".so")
    ):
        return None
    try:
        return _import_module_from_library(name, build_dir, True)
    except Exception as e:
        warnings.warn(f"Failed to load cached CUDA extension '{name}': {e}.")
        return None


def load_cuda_extension(name, sources, build_directory=None, _try_cache=True):
    """Load a CUDA extension with cross-platform compiler flags.

    Tries the fast cached-build path first; only falls back to a JIT
    build when no prebuilt engine exists.
    """
    if _try_cache:
        cached = _load_cached_extension(name)
        if cached is not None:
            return cached
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
