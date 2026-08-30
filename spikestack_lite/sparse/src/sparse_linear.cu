#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// =============================================================================
// KERNEL 1: Naive dense (baseline)
// =============================================================================
template <typename scalar_t>
__global__ void naive_sparse_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_features, int out_features) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && o < out_features) {
        scalar_t sum = 0;
        for (int i = 0; i < in_features; ++i) {
            scalar_t spike = input[b * in_features + i];
            if (spike > 0) {
                sum += spike * weight[o * in_features + i];
            }
        }
        output[b * out_features + o] = sum;
    }
}

// =============================================================================
// KERNEL 2: CSR Sparse — iterates only over non-zero spike positions
// =============================================================================
template <typename scalar_t>
__global__ void csr_sparse_forward_kernel(
    const int*   __restrict__ row_offsets,
    const int*   __restrict__ col_indices,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_features, int out_features) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && o < out_features) {
        const int row_start = row_offsets[b];
        const int row_end   = row_offsets[b + 1];
        scalar_t sum = 0;
        for (int k = row_start; k < row_end; ++k) {
            sum += values[k] * weight[o * in_features + col_indices[k]];
        }
        output[b * out_features + o] = sum;
    }
}

// =============================================================================
// KERNEL 3: CSR + Coalesced Weight Access (transposed weight layout)
// =============================================================================
template <typename scalar_t>
__global__ void csr_coalesced_kernel(
    const int*   __restrict__ row_offsets,
    const int*   __restrict__ col_indices,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ weightT,
    scalar_t* __restrict__ output,
    int batch_size, int in_features, int out_features) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && o < out_features) {
        const int row_start = row_offsets[b];
        const int row_end   = row_offsets[b + 1];
        scalar_t sum = 0;
        for (int k = row_start; k < row_end; ++k) {
            sum += values[k] * weightT[col_indices[k] * out_features + o];
        }
        output[b * out_features + o] = sum;
    }
}

// =============================================================================
// KERNEL 4: CSR + Shared Memory Weight-Row
// =============================================================================
template <typename scalar_t>
__global__ void csr_weightrow_kernel(
    const int*   __restrict__ row_offsets,
    const int*   __restrict__ col_indices,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_features, int out_features) {

    const int o = blockIdx.x;
    const int tid = threadIdx.x;
    if (o >= out_features) return;

    extern __shared__ char smem_raw[];
    scalar_t* sm_weight = reinterpret_cast<scalar_t*>(smem_raw);

    for (int i = tid; i < in_features; i += blockDim.x)
        sm_weight[i] = weight[o * in_features + i];
    __syncthreads();

    for (int b = tid; b < batch_size; b += blockDim.x) {
        scalar_t sum = 0;
        for (int k = row_offsets[b]; k < row_offsets[b + 1]; ++k)
            sum += values[k] * sm_weight[col_indices[k]];
        output[b * out_features + o] = sum;
    }
}

// =============================================================================
// KERNEL 5: Multi-Timestep with Temporal Weight Reuse
// =============================================================================
template <typename scalar_t>
__global__ void csr_multistep_kernel(
    const int*   __restrict__ packed_row_offsets,
    const int*   __restrict__ packed_col_indices,
    const scalar_t* __restrict__ packed_values,
    const int*   __restrict__ step_nnz_prefix,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int T, int batch_size, int in_features, int out_features) {

    const int o = blockIdx.x;
    const int tid = threadIdx.x;
    if (o >= out_features) return;

    extern __shared__ char smem_raw[];
    scalar_t* sm_weight = reinterpret_cast<scalar_t*>(smem_raw);
    for (int i = tid; i < in_features; i += blockDim.x)
        sm_weight[i] = weight[o * in_features + i];
    __syncthreads();

    for (int t = 0; t < T; t++) {
        const int nnz_offset = step_nnz_prefix[t];
        const int base_ro = t * (batch_size + 1);
        for (int b = tid; b < batch_size; b += blockDim.x) {
            const int rs = packed_row_offsets[base_ro + b] - packed_row_offsets[base_ro];
            const int re = packed_row_offsets[base_ro + b + 1] - packed_row_offsets[base_ro];
            scalar_t sum = 0;
            for (int k = nnz_offset + rs; k < nnz_offset + re; ++k)
                sum += packed_values[k] * sm_weight[packed_col_indices[k]];
            output[(t * batch_size + b) * out_features + o] = sum;
        }
    }
}

// =============================================================================
// KERNEL 6: Fused Dense-to-CSR + Sparse Multiply (NO CSR conversion overhead)
//
// Takes a dense input tensor, discovers non-zeros on the fly, and multiplies
// with coalesced transposed weights — all in ONE kernel launch.
// This eliminates the Python CSR conversion overhead entirely.
// =============================================================================
template <typename scalar_t>
__global__ void fused_dense_sparse_kernel(
    const scalar_t* __restrict__ input,     // (batch_size, in_features)
    const scalar_t* __restrict__ weightT,   // (in_features, out_features) transposed
    scalar_t* __restrict__ output,          // (batch_size, out_features)
    int batch_size, int in_features, int out_features) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && o < out_features) {
        scalar_t sum = 0;
        const scalar_t* row = input + b * in_features;
        for (int i = 0; i < in_features; ++i) {
            scalar_t spike = row[i];
            if (spike > 0) {
                // COALESCED: weightT[i * out_features + o]
                sum += spike * weightT[i * out_features + o];
            }
        }
        output[b * out_features + o] = sum;
    }
}

// =============================================================================
// GPU-side Dense-to-CSR Conversion Kernels
// =============================================================================

// Step 1: Count non-zeros per row
template <typename scalar_t>
__global__ void count_nnz_kernel(
    const scalar_t* __restrict__ input,
    int* __restrict__ nnz_per_row,
    int batch_size, int in_features) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int count = 0;
    for (int i = 0; i < in_features; i++) {
        if (input[b * in_features + i] > 0) count++;
    }
    nnz_per_row[b] = count;
}

template <typename scalar_t>
__global__ void write_csr_kernel(
    const scalar_t* __restrict__ input,
    const int* __restrict__ row_offsets,
    int* __restrict__ col_indices,
    scalar_t* __restrict__ values,
    int batch_size, int in_features) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int pos = row_offsets[b];
    for (int i = 0; i < in_features; i++) {
        scalar_t spike = input[b * in_features + i];
        if (spike > 0) {
            col_indices[pos] = i;
            values[pos] = spike;
            pos++;
        }
    }
}

// =============================================================================
// Coalesced CSR build.
//
// The old count_nnz_kernel / write_csr_kernel gave one *row* to one thread, so
// a warp touched 32 rows = 32 separate 4KB segments -> ~32x uncoalesced read
// amplification every timestep (the dominant cost of the sparse engine).
//
// csr_build_kernel: one WARP per row, loads the row cooperatively (32
// consecutive elements per iteration -> fully coalesced), ballots to find the
// nonzeros and writes them tightly packed (no holes) into per-row scratch
// slots, recording the row's nnz count.
//
// csr_compact_kernel: after the parallel prefix sum gives absolute row
// offsets, move each row's packed nonzeros to its final CSR position. Reads
// and writes only the nonzero entries.
// =============================================================================
template <typename scalar_t>
__global__ void csr_build_kernel(
    const scalar_t* __restrict__ input,
    int* __restrict__ nnz_per_row,
    scalar_t* __restrict__ packed_values,
    int* __restrict__ packed_cols,
    int batch_size, int in_features) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= batch_size) return;
    const scalar_t* r = input + (size_t)row * in_features;
    const unsigned full = 0xffffffffu;
    int total = 0;
    for (int it = 0; it < in_features; it += 32) {
        const int idx = it + lane;
        const scalar_t v = (idx < in_features) ? r[idx] : (scalar_t)0;
        const unsigned flags = __ballot_sync(full, v > 0);
        if (v > 0) {
            const unsigned below = flags & ((1u << lane) - 1);
            const int pos = total + __popc(below);
            packed_cols[(size_t)row * in_features + pos] = idx;
            packed_values[(size_t)row * in_features + pos] = v;
        }
        total += __popc(flags);
    }
    nnz_per_row[row] = total;
}

template <typename scalar_t>
__global__ void csr_compact_kernel(
    const scalar_t* __restrict__ packed_values,
    const int* __restrict__ packed_cols,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ row_offsets,
    scalar_t* __restrict__ values,
    int* __restrict__ col_indices,
    int batch_size, int in_features) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    const int n = nnz_per_row[b];
    const int dst = row_offsets[b];
    const scalar_t* pv = packed_values + (size_t)b * in_features;
    const int* pc = packed_cols + (size_t)b * in_features;
    for (int k = 0; k < n; ++k) {
        col_indices[dst + k] = pc[k];
        values[dst + k] = pv[k];
    }
}

// =============================================================================
// Parallel prefix sum (replaces the single-thread version).
//
// The single-threaded scan cost O(B) serial dependent global reads/writes each
// timestep: for B=8192 that is an ~8k-deep latency chain (~1ms/step), the
// dominant cost of the multi-timestep kernel. Instead:
//   1. one thread per row-Hillis-Steele-scan per 256-row block (grid-wide
//      parallel), writing INCLUSIVE local offsets row_offsets[i+1] and each
//      block's total;
//   2. a tiny single-block scan of the NB <= 256 block totals into exclusive
//      block starting offsets;
//   3. add each block's start offset to its rows.
// Each step goes from an 8192-deep serial chain to a ~(log2 256 + NB) one.
// =============================================================================
__global__ void scan_blocks_kernel(
    const int* __restrict__ nnz_per_row,
    int* __restrict__ row_offsets,
    int* __restrict__ block_totals,
    int batch_size) {
    __shared__ int sm[256];
    const int t = threadIdx.x;
    const int i = blockIdx.x * 256 + t;
    sm[t] = (i < batch_size) ? nnz_per_row[i] : 0;
    __syncthreads();
#pragma unroll
    for (int d = 1; d < 256; d <<= 1) {
        int add = 0;
        __syncthreads();
        if (t >= d) add = sm[t - d];
        __syncthreads();
        sm[t] += add;
    }
    __syncthreads();
    if (i < batch_size) row_offsets[i + 1] = sm[t];
    if (t == 255) block_totals[blockIdx.x] = sm[t];
    if (blockIdx.x == 0 && t == 0) row_offsets[0] = 0;
}

__global__ void offscan_add_kernel(
    int* __restrict__ row_offsets,
    const int* __restrict__ block_totals,
    int NB, int batch_size) {
    __shared__ int sm[256];
    const int t = threadIdx.x;
    if (t < NB) sm[t] = block_totals[t];
    __syncthreads();
    if (t < NB) {
        int acc = 0;
        for (int k = 0; k < t; k++) acc += sm[k];
        const int base = t * 256;
#pragma unroll
        for (int i = 0; i < 256; i++) {
            const int idx = base + i;
            if (idx < batch_size) row_offsets[idx + 1] += acc;
        }
    }
}

// Launch the 2-kernel parallel prefix sum into pre-allocated row_offsets.
// block_totals: scratch of size (B + 255) / 256 ints, reused across timesteps.
inline void launch_prefix_sum(
    const int* __restrict__ nnz_per_row,
    int* __restrict__ row_offsets,
    int* __restrict__ block_totals,
    int batch_size) {
    const int NB = (batch_size + 255) / 256;
    scan_blocks_kernel<<<NB, 256>>>(nnz_per_row, row_offsets, block_totals, batch_size);
    offscan_add_kernel<<<1, 256>>>(row_offsets, block_totals, NB, batch_size);
}

// =============================================================================
// KERNEL 7: Fused Multi-Timestep Dense-to-Sparse (NO CSR conversion at all)
//
// Reads dense input directly, skips zeros inline, weights cached in shared memory
// across ALL timesteps. Single kernel launch for entire temporal loop.
// This eliminates CSR conversion overhead entirely — the branch skip is free
// when the weight is already in smem.
// =============================================================================
template <typename scalar_t>
__global__ void fused_multistep_kernel(
    const scalar_t* __restrict__ input,     // (T, batch_size, in_features)
    const scalar_t* __restrict__ weightT,   // (in_features, out_features) transposed
    scalar_t* __restrict__ output,          // (T, batch_size, out_features)
    int T, int batch_size, int in_features, int out_features) {

    const int o = blockIdx.x;
    const int tid = threadIdx.x;
    if (o >= out_features) return;

    extern __shared__ char smem_raw[];
    scalar_t* sm_weight = reinterpret_cast<scalar_t*>(smem_raw);

    // Load weight row for output feature o into shared memory ONCE
    for (int i = tid; i < in_features; i += blockDim.x)
        sm_weight[i] = weightT[i * out_features + o];
    __syncthreads();

    // Process all timesteps — weight stays in smem the entire time
    for (int t = 0; t < T; t++) {
        for (int b = tid; b < batch_size; b += blockDim.x) {
            const scalar_t* in_row = input + (t * batch_size + b) * in_features;
            scalar_t sum = 0;
            for (int i = 0; i < in_features; i++) {
                scalar_t spike = in_row[i];
                if (spike > 0) {
                    sum += spike * sm_weight[i];
                }
            }
            output[(t * batch_size + b) * out_features + o] = sum;
        }
    }
}

// =============================================================================
// KERNEL 8: Sparse Conv2d via implicit im2col
//
// Treats each (batch, spatial_position) as a CSR row.
// For each output channel o, loads filter row into shared memory.
// Each thread computes one (batch, h, w) position's output for channel o.
// Input is already sparse — we skip zero activations during im2col traversal.
//
// weightT: (C_out, C_in * kH * kW) transposed to (C_in * kH * kW, C_out)
// input:   (B, C_in, H, W)
// output:  (B, C_out, oH, oW)
// =============================================================================
template <typename scalar_t>
__global__ void sparse_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weightT,
    scalar_t* __restrict__ output,
    int B, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int oH, int oW, int sH, int sW,
    int pH, int pW) {

    const int o = blockIdx.x;
    const int tid = threadIdx.x;
    if (o >= C_out) return;

    const int spatial = B * oH * oW;
    const int filter_size = C_in * kH * kW;

    extern __shared__ char smem_raw[];
    scalar_t* sm_weight = reinterpret_cast<scalar_t*>(smem_raw);

    // Load filter for output channel o into shared memory ONCE
    for (int i = tid; i < filter_size; i += blockDim.x)
        sm_weight[i] = weightT[i * C_out + o];
    __syncthreads();

    // Each thread computes multiple spatial positions
    for (int pos = tid; pos < spatial; pos += blockDim.x) {
        const int ow = pos % oW;
        const int oh = (pos / oW) % oH;
        const int b  = pos / (oH * oW);

        scalar_t sum = 0;
        // im2col: iterate over filter window
        for (int ci = 0; ci < C_in; ci++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int ih = oh * sH - pH + kh;
                    int iw = ow * sW - pW + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        scalar_t spike = input[((b * C_in + ci) * H + ih) * W + iw];
                        if (spike > 0) {
                            int fi = (ci * kH + kh) * kW + kw;
                            sum += spike * sm_weight[fi];
                        }
                    }
                }
            }
        }
        output[((b * C_out + o) * oH + oh) * oW + ow] = sum;
    }
}

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor naive_sparse_forward(torch::Tensor input, torch::Tensor weight) {
    const int B = input.size(0), I = input.size(1), O = weight.size(0);
    auto output = torch::zeros({B, O}, input.options());
    const int t = 16;
    dim3 blocks((B+t-1)/t, (O+t-1)/t);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "naive_cuda", ([&] {
        naive_sparse_forward_kernel<scalar_t><<<blocks, dim3(t,t)>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), B, I, O);
    }));
    return output;
}

torch::Tensor csr_sparse_forward(
    torch::Tensor ro, torch::Tensor ci, torch::Tensor vl,
    torch::Tensor weight, int B, int I, int O) {
    auto output = torch::zeros({B, O}, weight.options());
    const int t = 16;
    dim3 blocks((B+t-1)/t, (O+t-1)/t);
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "csr_cuda", ([&] {
        csr_sparse_forward_kernel<scalar_t><<<blocks, dim3(t,t)>>>(
            ro.data_ptr<int>(), ci.data_ptr<int>(), vl.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), B, I, O);
    }));
    return output;
}

torch::Tensor csr_coalesced_forward(
    torch::Tensor ro, torch::Tensor ci, torch::Tensor vl,
    torch::Tensor weightT, int B, int I, int O) {
    auto output = torch::zeros({B, O}, weightT.options());
    const int t = 16;
    dim3 blocks((B+t-1)/t, (O+t-1)/t);
    AT_DISPATCH_FLOATING_TYPES(weightT.scalar_type(), "csr_coal_cuda", ([&] {
        csr_coalesced_kernel<scalar_t><<<blocks, dim3(t,t)>>>(
            ro.data_ptr<int>(), ci.data_ptr<int>(), vl.data_ptr<scalar_t>(),
            weightT.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), B, I, O);
    }));
    return output;
}

torch::Tensor csr_shared_mem_forward(
    torch::Tensor ro, torch::Tensor ci, torch::Tensor vl,
    torch::Tensor weight, int B, int I, int O) {
    auto output = torch::zeros({B, O}, weight.options());
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "csr_sm_cuda", ([&] {
        csr_weightrow_kernel<scalar_t><<<O, 256, I * weight.element_size()>>>(
            ro.data_ptr<int>(), ci.data_ptr<int>(), vl.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), B, I, O);
    }));
    return output;
}

torch::Tensor csr_multistep_forward(
    torch::Tensor pro, torch::Tensor pci, torch::Tensor pvl,
    torch::Tensor snp, torch::Tensor weight,
    int T, int B, int I, int O) {
    auto output = torch::zeros({T, B, O}, weight.options());
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "csr_ms_cuda", ([&] {
        csr_multistep_kernel<scalar_t><<<O, 256, I * weight.element_size()>>>(
            pro.data_ptr<int>(), pci.data_ptr<int>(), pvl.data_ptr<scalar_t>(),
            snp.data_ptr<int>(), weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), T, B, I, O);
    }));
    return output;
}

torch::Tensor fused_dense_sparse_forward(
    torch::Tensor input, torch::Tensor weightT) {
    const int B = input.size(0), I = input.size(1), O = weightT.size(1);
    auto output = torch::zeros({B, O}, input.options());
    const int t = 16;
    dim3 blocks((B+t-1)/t, (O+t-1)/t);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_cuda", ([&] {
        fused_dense_sparse_kernel<scalar_t><<<blocks, dim3(t,t)>>>(
            input.data_ptr<scalar_t>(), weightT.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), B, I, O);
    }));
    return output;
}

std::vector<torch::Tensor> dense_to_csr_gpu(torch::Tensor dense_input) {
    const int B = dense_input.size(0), I = dense_input.size(1);
    const int max_nnz = B * I;
    const int t = 256;
    const int grid = (B + t - 1) / t;

    auto nnz_per_row = torch::empty(B, torch::dtype(torch::kInt32).device(dense_input.device()));
    auto row_offsets = torch::empty(B + 1, torch::dtype(torch::kInt32).device(dense_input.device()));
    auto col_indices = torch::empty(max_nnz, torch::dtype(torch::kInt32).device(dense_input.device()));
    auto values = torch::empty(max_nnz, dense_input.options());

    AT_DISPATCH_FLOATING_TYPES(dense_input.scalar_type(), "dense_to_csr_gpu", ([&] {
        count_nnz_kernel<scalar_t><<<grid, t>>>(
            dense_input.data_ptr<scalar_t>(), nnz_per_row.data_ptr<int>(), B, I);
    }));
    auto block_totals = torch::empty({grid}, torch::dtype(torch::kInt32).device(dense_input.device()));
    launch_prefix_sum(nnz_per_row.data_ptr<int>(), row_offsets.data_ptr<int>(),
                      block_totals.data_ptr<int>(), B);
    AT_DISPATCH_FLOATING_TYPES(dense_input.scalar_type(), "dense_to_csr_write", ([&] {
        write_csr_kernel<scalar_t><<<grid, t>>>(
            dense_input.data_ptr<scalar_t>(), row_offsets.data_ptr<int>(),
            col_indices.data_ptr<int>(), values.data_ptr<scalar_t>(), B, I);
    }));

    return {row_offsets, col_indices, values};
}

// =============================================================================
// Fused single-call: CSR conversion + coalesced multiply in ONE C++ call
// Takes pre-allocated temp buffers to avoid per-call allocation overhead.
// =============================================================================
torch::Tensor sparse_linear_forward(
    torch::Tensor input, torch::Tensor weightT,
    torch::Tensor nnz_per_row, torch::Tensor row_offsets,
    torch::Tensor col_indices, torch::Tensor values) {
    const int B = input.size(0), I = input.size(1), O = weightT.size(1);
    const int t = 256;
    const int grid = (B + 255) / 256;
    const int wgrid = (B + 7) / 8;  // warp-per-row: 256 threads = 8 warps/block

    auto packed_cols = torch::empty({B, I}, torch::dtype(torch::kInt32).device(input.device()));
    auto packed_vals = torch::empty({B, I}, input.options());
    auto block_totals = torch::empty({grid}, torch::dtype(torch::kInt32).device(input.device()));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sl_build", ([&] {
        csr_build_kernel<scalar_t><<<wgrid, t>>>(
            input.data_ptr<scalar_t>(), nnz_per_row.data_ptr<int>(),
            packed_vals.data_ptr<scalar_t>(), packed_cols.data_ptr<int>(), B, I);
    }));
    launch_prefix_sum(nnz_per_row.data_ptr<int>(), row_offsets.data_ptr<int>(),
                      block_totals.data_ptr<int>(), B);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sl_compact", ([&] {
        csr_compact_kernel<scalar_t><<<grid, t>>>(
            packed_vals.data_ptr<scalar_t>(), packed_cols.data_ptr<int>(),
            nnz_per_row.data_ptr<int>(), row_offsets.data_ptr<int>(),
            values.data_ptr<scalar_t>(), col_indices.data_ptr<int>(), B, I);
    }));

    auto output = torch::empty({B, O}, input.options());
    dim3 blocks((B + 15) / 16, (O + 15) / 16);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sl_multiply", ([&] {
        csr_coalesced_kernel<scalar_t><<<blocks, dim3(16, 16)>>>(
            row_offsets.data_ptr<int>(), col_indices.data_ptr<int>(),
            values.data_ptr<scalar_t>(), weightT.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), B, I, O);
    }));

    return output;
}

// =============================================================================
// C++ Interface: Multi-Timestep with CSR (tight C++ loop, zero Python overhead)
//
// For each of T timesteps:
//   1. Count non-zeros (GPU kernel)
//   2. Prefix sum (GPU kernel)
//   3. Write CSR columns+values (GPU kernel)
//   4. Coalesced CSR multiply (GPU kernel)
//
// All pre-allocated buffers are reused across timesteps. No Python calls between steps.
// Input: (T, B, I), Output: (T, B, O)
// =============================================================================
torch::Tensor sparse_multistep_forward(
    torch::Tensor dense_input,
    torch::Tensor weightT,
    torch::Tensor nnz_per_row,
    torch::Tensor row_offsets,
    torch::Tensor col_indices,
    torch::Tensor values,
    int T) {

    const int B = dense_input.size(1);
    const int I = dense_input.size(2);
    const int O = weightT.size(1);
    const int max_nnz = B * I;
    const int grid = (B + 255) / 256;
    const int wgrid = (B + 7) / 8;  // warp-per-row: 256 threads = 8 warps/block

    auto output = torch::empty({T, B, O}, dense_input.options());
    auto packed_cols = torch::empty({B, I}, torch::dtype(torch::kInt32).device(dense_input.device()));
    auto packed_vals = torch::empty({B, I}, dense_input.options());
    auto block_totals = torch::empty({grid}, torch::dtype(torch::kInt32).device(dense_input.device()));

    AT_DISPATCH_FLOATING_TYPES(dense_input.scalar_type(), "sparse_ms_loop", ([&] {
        for (int t = 0; t < T; t++) {
            const scalar_t* inp_t = dense_input.data_ptr<scalar_t>() + t * B * I;
            scalar_t* out_t = output.data_ptr<scalar_t>() + t * B * O;

            csr_build_kernel<scalar_t><<<wgrid, 256>>>(
                inp_t, nnz_per_row.data_ptr<int>(),
                packed_vals.data_ptr<scalar_t>(), packed_cols.data_ptr<int>(), B, I);

            launch_prefix_sum(nnz_per_row.data_ptr<int>(), row_offsets.data_ptr<int>(),
                              block_totals.data_ptr<int>(), B);

            csr_compact_kernel<scalar_t><<<grid, 256>>>(
                packed_vals.data_ptr<scalar_t>(), packed_cols.data_ptr<int>(),
                nnz_per_row.data_ptr<int>(), row_offsets.data_ptr<int>(),
                values.data_ptr<scalar_t>(), col_indices.data_ptr<int>(), B, I);

            dim3 blocks((B + 15) / 16, (O + 15) / 16);
            csr_coalesced_kernel<scalar_t><<<blocks, dim3(16, 16)>>>(
                row_offsets.data_ptr<int>(), col_indices.data_ptr<int>(),
                values.data_ptr<scalar_t>(), weightT.data_ptr<scalar_t>(),
                out_t, B, I, O);
        }
    }));
    return output;
}

// =============================================================================
// C++ Interface: Sparse Conv2d
// =============================================================================
torch::Tensor sparse_conv2d_forward(
    torch::Tensor input, torch::Tensor weight,
    int stride_h, int stride_w, int pad_h, int pad_w) {

    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = weight.size(0);
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    const int oH = (H + 2 * pad_h - kH) / stride_h + 1;
    const int oW = (W + 2 * pad_w - kW) / stride_w + 1;

    // weightT: reshape (C_out, C_in*kH*kW) -> transpose -> (C_in*kH*kW, C_out)
    auto w_flat = weight.reshape({C_out, C_in * kH * kW});
    auto weightT = w_flat.t().contiguous();

    auto output = torch::empty({B, C_out, oH, oW}, input.options());
    const int filter_size = C_in * kH * kW;
    const int smem_bytes = filter_size * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sparse_conv2d_cuda", ([&] {
        sparse_conv2d_kernel<scalar_t><<<C_out, 256, smem_bytes>>>(
            input.data_ptr<scalar_t>(), weightT.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, C_in, H, W, C_out, kH, kW, oH, oW,
            stride_h, stride_w, pad_h, pad_w);
    }));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &naive_sparse_forward, "Naive (dense loop + branch skip)");
    m.def("csr_forward", &csr_sparse_forward, "CSR sparse");
    m.def("csr_coalesced_forward", &csr_coalesced_forward, "CSR + coalesced weight");
    m.def("csr_shared_forward", &csr_shared_mem_forward, "CSR + shared memory");
    m.def("csr_multistep_forward", &csr_multistep_forward, "CSR + temporal reuse");
    m.def("fused_forward", &fused_dense_sparse_forward, "Fused dense->sparse (no CSR overhead)");
    m.def("dense_to_csr_gpu", &dense_to_csr_gpu, "GPU-side dense-to-CSR conversion");
    m.def("sparse_forward", &sparse_linear_forward, "Fused: GPU CSR conversion + coalesced multiply (single call)");
    m.def("sparse_multistep_forward", &sparse_multistep_forward, "Multi-timestep CSR: tight C++ loop, pre-allocated buffers");
    m.def("sparse_conv2d_forward", &sparse_conv2d_forward, "Sparse Conv2d: implicit im2col with zero-skipping");
}
