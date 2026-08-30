#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void spire_dither_kernel(
    const float* __restrict__ v,       // (B, N, d)
    const float* __restrict__ dither,  // (R)
    float* __restrict__ out,           // (R, B, N, d)
    int B,
    int N,
    int d,
    int R
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int BNd = B * N * d;
    int total = R * BNd;
    if (idx >= total) return;

    int r = idx / BNd;
    int rem = idx % BNd;

    float val = v[rem] - dither[r];
    out[idx] = (val > 0.0f) ? 1.0f : 0.0f;
}

torch::Tensor spire_dither_cuda(
    torch::Tensor v,
    torch::Tensor dither
) {
    TORCH_CHECK(v.is_cuda() && dither.is_cuda(), "v and dither must be CUDA tensors");
    TORCH_CHECK(v.dim() == 3, "v must be (B, N, d)");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(dither.scalar_type() == torch::kFloat32, "dither must be float32");
    v = v.contiguous();
    dither = dither.contiguous();
    int B = v.size(0);
    int N = v.size(1);
    int d = v.size(2);
    int R = dither.numel();

    auto out = torch::empty({R, B, N, d}, v.options());

    int total = R * B * N * d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    spire_dither_kernel<<<blocks, threads>>>(
        v.data_ptr<float>(),
        dither.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, d, R
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spire_dither", &spire_dither_cuda, "Spire Dither CUDA Kernel");
}
