"""
SpikeSkip: Bypass the Von Neumann Memory Wall.

Custom CUDA sparse inference engine for PyTorch that skips weight memory fetches
for silent neurons using CSR format on standard NVIDIA GPUs.

Usage:
    from spikestack_lite.sparse import SparseLinear, sparse_linear_forward

    # High-level module (drop-in replacement for nn.Linear)
    layer = SparseLinear(4096, 4096)
    output = layer(input_spikes)

    # Low-level function
    output = sparse_linear_forward(input, weightT, preallocated_buffers)

The CUDA engine lives in ``sparse_linear.cu`` (packaged under ``src/``). It is
JIT-compiled on first import (cached by torch afterwards). If CUDA, nvcc, or a
C++ toolchain is unavailable, the module degrades gracefully to dense matmul.
"""

import glob
import os

import torch

_MSG_CTX = "spikestack_lite.sparse"


def _setup_toolchain():
    """Auto-detect CUDA_HOME + MSVC bin and prepend to PATH (harmless if absent)."""
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


_setup_toolchain()

_cuda_engine = None
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
_cu_file = os.path.join(_src_dir, "sparse_linear.cu")

if torch.cuda.is_available() and os.path.exists(_cu_file):
    try:
        from torch.utils.cpp_extension import load

        _cuda_engine = load(
            name="spikeskip_cuda_engine",
            sources=[_cu_file],
            extra_cuda_cflags=["-O3", "-Xcompiler", "/Zc:preprocessor"],
            extra_cflags=["-O3", "/Zc:preprocessor"],
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001 - any toolchain/compile failure -> fallback
        print(
            f"[{_MSG_CTX}] Warning: SpikeSkip CUDA JIT compilation failed. "
            f"SpikeSkip acceleration will be disabled (falling back to dense). "
            f"Error: {e}"
        )
        _cuda_engine = None
else:
    if not os.path.exists(_cu_file):
        print(
            f"[{_MSG_CTX}] Warning: {_cu_file} not found. "
            f"SpikeSkip acceleration will be disabled (falling back to dense)."
        )
    elif not torch.cuda.is_available():
        print(
            f"[{_MSG_CTX}] Note: no CUDA device detected. "
            f"SpikeSkip acceleration will be disabled (falling back to dense)."
        )


def sparse_linear_forward(input, weightT, nnz_per_row, row_offsets, col_indices, values):
    """Fused sparse linear: GPU CSR conversion + coalesced multiply in ONE call."""
    if _cuda_engine is None:
        # Fallback to standard dense matmul if CUDA SpikeSkip failed to compile
        return torch.matmul(input, weightT)

    return _cuda_engine.sparse_forward(input, weightT, nnz_per_row, row_offsets, col_indices, values)


def sparse_multistep_forward(dense_input, weightT, nnz_per_row, row_offsets, col_indices, values, T):
    """Multi-timestep sparse linear: CSR conversion + multiply for T timesteps in a tight C++ loop."""
    if _cuda_engine is None:
        return torch.matmul(dense_input, weightT)
    return _cuda_engine.sparse_multistep_forward(dense_input, weightT, nnz_per_row, row_offsets, col_indices, values, T)


def sparse_conv2d_forward(input, weight, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    """Sparse Conv2d: zero-skipping convolution via implicit im2col."""
    if _cuda_engine is None:
        raise RuntimeError("SpikeSkip CUDA engine is not available (sparse_conv2d has no dense fallback).")
    return _cuda_engine.sparse_conv2d_forward(input, weight, stride_h, stride_w, pad_h, pad_w)


class _SparseMatmulAutograd(torch.autograd.Function):
    """
    Autograd-aware wrapper around the SpikeSkip CUDA kernel.

    The raw CUDA ops in ``sparse_linear.cu`` are not registered with PyTorch's
    autograd, so calling them directly detaches the graph. This Function keeps
    the sparse kernel for the forward pass and supplies the *exact* gradients
    with dense matmuls in backward (backprop needs no sparsity to be correct).
    """

    @staticmethod
    def forward(ctx, flat, weight, engine, csr):
        # out[b, o] = sum_f flat[b, f] * weight[o, f]   (CSR multiply over non-zero f)
        if engine is None:
            out = flat @ weight.t()
        else:
            nnz_per_row, row_offsets, col_indices, values = csr
            out = engine.sparse_forward(
                flat, weight.t().contiguous(),
                nnz_per_row, row_offsets, col_indices, values,
            )
        ctx.save_for_backward(flat, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        flat, weight = ctx.saved_tensors
        grad_flat = grad_out @ weight
        grad_weight = grad_out.t() @ flat
        return grad_flat, grad_weight, None, None


class _SparseMultistepAutograd(torch.autograd.Function):
    """
    Autograd-aware wrapper around the SpikeSkip *multi-timestep* kernel.

    Same contract as ``_SparseMatmulAutograd`` but for batched temporal inputs
    of shape (T, *, in_features): the kernel packs/CSR-converts once and reuses
    weights across all T timesteps (one tight C++ loop). Backward is exact and
    dense.
    """

    @staticmethod
    def forward(ctx, inp, weight, engine, csr, T):
        # out[t, ..., o] = sum_f inp[t, ..., f] * weight[o, f]
        if engine is None:
            out = torch.matmul(inp, weight.t())
        else:
            nnz_per_row, row_offsets, col_indices, values = csr
            out = engine.sparse_multistep_forward(
                inp, weight.t().contiguous(),
                nnz_per_row, row_offsets, col_indices, values, int(T),
            )
        ctx.save_for_backward(inp, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight = ctx.saved_tensors
        grad_inp = torch.matmul(grad_out, weight)
        grad_weight = torch.einsum("...bo,...bf->of", grad_out, inp)
        return grad_inp, grad_weight, None, None, None


def alloc_csr_buffers(batch_size, in_features, device=None):
    """Allocate reusable CSR buffers for sparse_linear_forward."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return (
        torch.empty(batch_size, dtype=torch.int32, device=device),
        torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        torch.empty(batch_size * in_features, dtype=torch.int32, device=device),
        torch.empty(batch_size * in_features, dtype=torch.float32, device=device),
    )


class SparseLinear(torch.nn.Module):
    """
    Drop-in replacement for torch.nn.Linear using sparse computation.

    Zero-valued activations skip weight memory fetches entirely.
    Best performance at >99% sparsity (typical for SNNs and deep ReLU networks).

    Example:
        layer = SparseLinear(4096, 4096)
        output = layer.forward_single(input)
    """

    def __init__(self, in_features, out_features, bias=False, device=None, sparse_threshold=1024):
        super().__init__()
        # nn.Linear convention: parameters live on CPU unless the caller moves
        # the module (.cuda()). The engine then engages once inputs are on CUDA
        # and the layer is wide enough.
        device = device or "cpu"
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias, device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.sparse_threshold = sparse_threshold
        self._buffers_allocated = False
        self._buf_device = None
        self._buf_cap = 0

    def _ensure_buffers(self, device, batch):
        if not self._buffers_allocated or self._buf_device != device or batch > self._buf_cap:
            self._buf_cap = max(int(batch), 2048)
            self._nnz_per_row, self._row_offsets, self._col_indices, self._values = \
                alloc_csr_buffers(self._buf_cap, self.in_features, device)
            self._buf_device = device
            self._buffers_allocated = True

    def forward_single(self, input):
        """Single-timestep sparse forward pass. input: (*, in_features)."""
        leading = input.shape[:-1]
        flat = input.reshape(-1, self.in_features).contiguous()
        # The CUDA kernel only runs on GPU; never launch it on CPU tensors.
        if _cuda_engine is None or flat.device.type != "cuda":
            return self.linear(input)
        B = flat.size(0)
        self._ensure_buffers(flat.device, B)
        csr = (self._nnz_per_row, self._row_offsets, self._col_indices, self._values)
        out = _SparseMatmulAutograd.apply(flat, self.linear.weight, _cuda_engine, csr)
        if self.linear.bias is not None:
            out = out + self.linear.bias
        return out.view(*leading, self.out_features)

    def forward_multistep(self, dense_input, T=None):
        """Multi-timestep sparse forward. Supports any leading dims.

        * 2D ``(*, in_features)``           -> single step (forward_single).
        * 3D+ ``(T, ..., in_features)``     -> T timesteps; the kernel packs,
          CSR-converts once and reuses weights across all T (one tight C++ loop).
        """
        if dense_input.dim() == 2:
            return self.forward_single(dense_input)
        if T is None:
            T = dense_input.size(0)
        leading = dense_input.shape[1:-1]
        flat = dense_input.reshape(T, -1, self.in_features).contiguous()
        B = flat.size(1)
        engine = _cuda_engine if flat.device.type == "cuda" else None
        csr = None
        if engine is not None:
            self._ensure_buffers(flat.device, B)
            csr = (self._nnz_per_row, self._row_offsets, self._col_indices, self._values)
        out = _SparseMultistepAutograd.apply(flat, self.linear.weight, engine, csr, T)
        if self.linear.bias is not None:
            out = out + self.linear.bias
        return out.view(T, *leading, self.out_features)

    def forward(self, x):
        """Default forward: sparse path for large CUDA layers, dense otherwise."""
        if (_cuda_engine is not None and x.device.type == "cuda"
                and self.in_features >= self.sparse_threshold):
            return self.forward_single(x)
        return self.linear(x)