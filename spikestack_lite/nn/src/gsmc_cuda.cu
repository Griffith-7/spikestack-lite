#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float sigmoid_fn(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gsmc_step_kernel(
    const float* __restrict__ gi,         // (M, 4d)
    const float* __restrict__ gr,         // (M, 4d)
    const float* __restrict__ wd,         // (M, d)
    const float* __restrict__ bias,       // (4d)
    float* __restrict__ m,                // (M, d) (in-place)
    float* __restrict__ s,                // (M, d) (in-place)
    float* __restrict__ heat,             // (M, d) (in-place)
    float* __restrict__ out_t,            // (M, d)
    int M,
    int d,
    float v_th,
    float gate_scale,
    bool adaptive_threshold,
    float beta_adapt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * d;
    if (idx >= total) return;

    int row = idx / d;
    int col = idx % d;

    // Load pre-activations for 4 gates: f, i, o, g
    float gi_f = gi[row * (4 * d) + col];
    float gi_i = gi[row * (4 * d) + d + col];
    float gi_o = gi[row * (4 * d) + 2 * d + col];
    float gi_g = gi[row * (4 * d) + 3 * d + col];

    float gr_f = gr[row * (4 * d) + col];
    float gr_i = gr[row * (4 * d) + d + col];
    float gr_o = gr[row * (4 * d) + 2 * d + col];
    float gr_g = gr[row * (4 * d) + 3 * d + col];

    float bias_f = bias[col];
    float bias_i = bias[d + col];
    float bias_o = bias[2 * d + col];
    float bias_g = bias[3 * d + col];

    float pre_f = (gi_f + gr_f) / gate_scale + bias_f;
    float pre_i = (gi_i + gr_i) / gate_scale + bias_i;
    float pre_o = (gi_o + gr_o) / gate_scale + bias_o;
    float pre_g = (gi_g + gr_g) / gate_scale + bias_g;

    float f = sigmoid_fn(pre_f);
    float i = sigmoid_fn(pre_i);
    float o = sigmoid_fn(pre_o);
    float g = tanhf(pre_g);

    float m_prev = m[idx];
    float a = f * m_prev + i * g;

    float wd_val = wd[idx];
    float v = o * tanhf(a) + wd_val;

    float heat_prev = heat[idx];
    float theta_eff = adaptive_threshold ? (v_th * (1.0f + heat_prev)) : v_th;

    float s_new = (v > theta_eff) ? 1.0f : 0.0f;
    float m_new = a - s_new * v_th;
    float heat_new = adaptive_threshold ? (beta_adapt * heat_prev + s_new) : 0.0f;

    m[idx] = m_new;
    s[idx] = s_new;
    heat[idx] = heat_new;
    out_t[idx] = s_new;
}

torch::Tensor fused_gsmc_step_cuda(
    torch::Tensor gi,
    torch::Tensor gr,
    torch::Tensor wd,
    torch::Tensor bias,
    torch::Tensor m,
    torch::Tensor s,
    torch::Tensor heat,
    float v_th,
    float gate_scale,
    bool adaptive_threshold,
    float beta_adapt
) {
    TORCH_CHECK(gi.is_cuda() && gr.is_cuda() && wd.is_cuda() && bias.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(m.is_cuda() && s.is_cuda() && heat.is_cuda(), "State tensors m, s, heat must be CUDA tensors");
    TORCH_CHECK(gi.scalar_type() == torch::kFloat32 && gr.scalar_type() == torch::kFloat32, "gi and gr must be float32");
    TORCH_CHECK(wd.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32, "wd and bias must be float32");

    TORCH_CHECK(gi.is_contiguous() && gr.is_contiguous() && wd.is_contiguous() && bias.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(m.is_contiguous() && s.is_contiguous() && heat.is_contiguous(), "State tensors must be contiguous");

    int M = gi.size(0);
    int d = wd.size(1);

    TORCH_CHECK(gi.size(1) == 4 * d && gr.size(1) == 4 * d, "gi and gr must be of shape (M, 4*d)");
    TORCH_CHECK(bias.numel() == 4 * d, "bias must have 4*d elements");
    TORCH_CHECK(m.numel() == M * d && s.numel() == M * d && heat.numel() == M * d, "m, s and heat must be of size (M, d)");

    auto out_t = torch::empty({M, d}, gi.options());

    int total = M * d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    gsmc_step_kernel<<<blocks, threads>>>(
        gi.data_ptr<float>(),
        gr.data_ptr<float>(),
        wd.data_ptr<float>(),
        bias.data_ptr<float>(),
        m.data_ptr<float>(),
        s.data_ptr<float>(),
        heat.data_ptr<float>(),
        out_t.data_ptr<float>(),
        M, d, v_th, gate_scale, adaptive_threshold, beta_adapt
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return out_t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gsmc_step", &fused_gsmc_step_cuda,
          "Fused GSMC Step CUDA Kernel. NOTE: Mutates state tensors m, s, heat in place.");
}

