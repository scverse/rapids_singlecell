from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

normalize_kernel_optimized = r"""
({0} * X, long long rows, long long cols) {
    __shared__ {0}  shared[32];  // Shared memory for partial sums (one per thread)

    long long row = blockIdx.x;  // One block per row
    long long tid = threadIdx.x;  // Thread index within the block

    // Ensure we're within matrix bounds
    if (row >= rows) return;

    // Step 1: Compute partial sums within each thread
    {0}  norm = 0.0;
    for (long long col = tid; col < cols; col += blockDim.x) {
        norm += fabs(X[row * cols + col]);// Manhattan norm

    }

    // Store partial sum in shared memory
    shared[tid] = norm;
    __syncthreads();

    // Step 2: Perform shared memory reduction using warp shuffle
    #pragma unroll
    for (long long offset = 16; offset > 0; offset /= 2) {
        shared[tid] += __shfl_down_sync(0xFFFFFFFF, shared[tid], offset);
    }
    __syncthreads();

    // First thread calculates the final norm
    if (tid == 0) {
        {0}  final_norm = shared[0];
        final_norm = fmaxf(final_norm, 1e-12);
        shared[0] = 1.0 / final_norm;  // Store reciprocal for normalization
    }
    __syncthreads();

    // Step 3: Normalize the row
    for (long long col = tid; col < cols; col += blockDim.x) {
        X[row * cols + col] *= shared[0];
    }
}
"""


def _get_normalize_kernel_optimized(dtype):
    return cuda_kernel_factory(
        normalize_kernel_optimized, (dtype,), "normalize_kernel_optimized"
    )
