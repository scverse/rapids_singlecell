from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

normalize_kernel_optimized = r"""
({0} * X, long long rows, long long cols) {
    // Shared memory for warp-level partial sums (max 32 warps = 1024 threads)
    __shared__ {0} warp_sums[32];

    long long row = blockIdx.x;  // One block per row
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    if (row >= rows) return;

    {0}* row_ptr = X + row * cols;

    // Step 1: Each thread accumulates its portion
    {0} acc = {0}(0);
    for (long long col = tid; col < cols; col += blockDim.x) {
        acc += fabs(row_ptr[col]);
    }

    // Step 2: Warp-level reduction using shuffle (on register value)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    // Step 3: First warp reduces all warp results
    if (tid < 32) {
        {0} val = (tid < num_warps) ? warp_sums[tid] : {0}(0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            {0} final_norm = fmax(val, {0}(1e-12));
            warp_sums[0] = {0}(1) / final_norm;
        }
    }
    __syncthreads();

    // Step 4: Normalize the row
    {0} scale = warp_sums[0];
    for (long long col = tid; col < cols; col += blockDim.x) {
        row_ptr[col] *= scale;
    }
}
"""


def _get_normalize_kernel_optimized(dtype):
    return cuda_kernel_factory(
        normalize_kernel_optimized, (dtype,), "normalize_kernel_optimized"
    )
