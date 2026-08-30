#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void exact_head_kernel(
    const float* __restrict__ drive,     // (B, C)
    const float* __restrict__ theta,     // (C)
    const float* __restrict__ t_grid,    // (T_grid)
    float* __restrict__ t_out,           // (B, C)
    float* __restrict__ v_max,           // (B, C)
    int B,
    int C,
    int T_grid,
    float beta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;

    int c = idx % C;
    float drv = drive[idx];
    float th = theta[c];

    float max_u = -1e9f;
    float max_arg = -1e30f;

    // Pass 1: find max_u and max of beta * uc for numerical stability.
    for (int t = 0; t < T_grid; ++t) {
        float s = t_grid[t];
        float u = drv * expf(-s / 10.0f);
        if (u > max_u) max_u = u;

        float uc = fminf(fmaxf(u - th, -8.0f), 8.0f);
        max_arg = fmaxf(max_arg, beta * uc);
    }

    // Pass 2: compute softmax weighted sum over t_grid.
    float sum_exp = 0.0f;
    float sum_weighted = 0.0f;
    for (int t = 0; t < T_grid; ++t) {
        float s = t_grid[t];
        float u = drv * expf(-s / 10.0f);
        float uc = fminf(fmaxf(u - th, -8.0f), 8.0f);

        float w = expf(beta * uc - max_arg);
        sum_exp += w;
        sum_weighted += w * s;
    }

    t_out[idx] = sum_weighted / (sum_exp + 1e-12f);
    v_max[idx] = max_u;
}

std::vector<torch::Tensor> fused_exact_head_cuda(
    torch::Tensor drive,
    torch::Tensor theta,
    torch::Tensor t_grid,
    float beta
) {
    TORCH_CHECK(drive.is_cuda() && theta.is_cuda() && t_grid.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(drive.dim() == 2, "drive must be 2D tensor (B, C)");
    TORCH_CHECK(drive.scalar_type() == torch::kFloat32, "drive must be float32");
    TORCH_CHECK(theta.scalar_type() == torch::kFloat32, "theta must be float32");
    TORCH_CHECK(t_grid.scalar_type() == torch::kFloat32, "t_grid must be float32");

    int B = drive.size(0);
    int C = drive.size(1);
    TORCH_CHECK(theta.dim() == 1 && theta.size(0) == C, "theta must be 1D tensor of size C matching drive.size(1)");
    TORCH_CHECK(t_grid.dim() == 1 && t_grid.numel() > 0, "t_grid must be a non-empty 1D tensor");

    drive = drive.contiguous();
    theta = theta.contiguous();
    t_grid = t_grid.contiguous();

    int T_grid = t_grid.size(0);

    auto t_out = torch::empty({B, C}, drive.options());

    auto v_max = torch::empty({B, C}, drive.options());

    int total = B * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    exact_head_kernel<<<blocks, threads>>>(
        drive.data_ptr<float>(),
        theta.data_ptr<float>(),
        t_grid.data_ptr<float>(),
        t_out.data_ptr<float>(),
        v_max.data_ptr<float>(),
        B, C, T_grid, beta
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return {t_out, v_max};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_exact_head", &fused_exact_head_cuda, "Fused Exact Head CUDA Kernel");
}

