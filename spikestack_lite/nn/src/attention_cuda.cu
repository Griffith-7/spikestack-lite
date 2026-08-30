#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void apply_channel_decay_kernel(
    const float* __restrict__ k,          // (B_total, N, d)
    const float* __restrict__ lam,        // (d)
    float* __restrict__ k_w,              // (B_total, N, d)
    int B_total,
    int N,
    int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B_total * N * d;
    if (idx >= total) return;

    int c = idx % d;
    int n = (idx / d) % N;

    float l = lam[c];
    float pos = (float)(N - 1 - n);
    float weight = powf(l, pos);

    k_w[idx] = k[idx] * weight;
}

__global__ void fused_normalize_attention_kernel(
    const float* __restrict__ numer,      // (total_elements)
    const float* __restrict__ denom,      // (total_elements)
    float* __restrict__ out,              // (total_elements)
    float eps,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    out[idx] = numer[idx] / (denom[idx] + eps);
}

torch::Tensor apply_channel_decay_cuda(
    torch::Tensor k,
    torch::Tensor lam
) {
    TORCH_CHECK(k.is_cuda() && lam.is_cuda(), "k and lam must be CUDA tensors");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(lam.scalar_type() == torch::kFloat32, "lam must be float32");
    TORCH_CHECK(k.dim() >= 2, "k must have at least 2 dimensions");

    if (k.numel() == 0) return torch::empty_like(k);

    k = k.contiguous();
    lam = lam.contiguous();

    int d = k.size(-1);
    int N = k.size(-2);
    TORCH_CHECK(N > 0 && d > 0, "Sequence length N and channel dimension d must be positive");
    TORCH_CHECK(lam.dim() == 1 && lam.size(0) == d, "lam must be 1D tensor of size d matching k.size(-1)");

    int B_total = k.numel() / (N * d);

    auto k_w = torch::empty_like(k);
    int total = B_total * N * d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    apply_channel_decay_kernel<<<blocks, threads>>>(
        k.data_ptr<float>(),
        lam.data_ptr<float>(),
        k_w.data_ptr<float>(),
        B_total, N, d
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return k_w;
}

torch::Tensor fused_normalize_attention_cuda(
    torch::Tensor numer,
    torch::Tensor denom,
    float eps
) {
    TORCH_CHECK(numer.is_cuda() && denom.is_cuda(), "numer and denom must be CUDA tensors");
    TORCH_CHECK(numer.scalar_type() == torch::kFloat32, "numer must be float32");
    TORCH_CHECK(denom.scalar_type() == torch::kFloat32, "denom must be float32");
    TORCH_CHECK(denom.numel() >= numer.numel(), "denom must have at least as many elements as numer");

    numer = numer.contiguous();
    denom = denom.contiguous();


    auto out = torch::empty_like(numer);
    int total = numer.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_normalize_attention_kernel<<<blocks, threads>>>(
        numer.data_ptr<float>(),
        denom.data_ptr<float>(),
        out.data_ptr<float>(),
        eps, total
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_channel_decay", &apply_channel_decay_cuda, "Apply Channel Decay CUDA Kernel");
    m.def("fused_normalize_attention", &fused_normalize_attention_cuda, "Fused Normalize Attention CUDA Kernel");
}

