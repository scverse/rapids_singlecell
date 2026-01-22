"""
Fallback kernel for computing pairwise group distances.

No shared memory tiling - works for any n_features but slower.
Used when n_features is too large for tiled kernels.
"""

from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory


def get_compute_group_distances_kernel_fallback(dtype):
    """
    Compiles fallback kernel - no shared memory, works for any n_features.

    Slower than tiled kernels but handles large feature counts.

    Parameters
    ----------
    dtype : dtype
        Data type for embeddings

    Returns
    -------
    kernel : callable
        Compiled CUDA kernel
    """

    kernel_code = """
(const {0}* __restrict__ embedding,
 const int* __restrict__ cat_offsets,
 const int* __restrict__ cell_indices,
 const int* __restrict__ pair_left,
 const int* __restrict__ pair_right,
 {0}* __restrict__ pairwise_sums,
 int k,
 int n_features,
 int blocks_per_pair) {

    const int thread_id = threadIdx.x;
    const int pair_id = blockIdx.x;
    const int block_in_pair = blockIdx.y;
    const int block_size = blockDim.x;

    {0} local_sum = ({0})(0.0);

    const int a = pair_left[pair_id];
    const int b = pair_right[pair_id];

    const int start_a = cat_offsets[a];
    const int end_a = cat_offsets[a + 1];
    const int start_b = cat_offsets[b];
    const int end_b = cat_offsets[b + 1];

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    // Distribute A cells across blocks_per_pair
    const int total_threads_for_pair = blocks_per_pair * block_size;
    const int global_thread_in_pair = block_in_pair * block_size + thread_id;
    const int n_iters_a = (n_a + total_threads_for_pair - 1) / total_threads_for_pair;

    for (int iter_a = 0; iter_a < n_iters_a; ++iter_a) {
        const int ia = start_a + iter_a * total_threads_for_pair + global_thread_in_pair;
        const bool valid_a = (ia < end_a);
        const int idx_i = valid_a ? cell_indices[ia] : 0;
        const int i_local = ia - start_a;

        if (valid_a) {
            // Iterate over all B cells (no tiling)
            for (int jb = 0; jb < n_b; ++jb) {
                int j_local = jb;

                // Skip lower triangle for diagonal blocks
                if (a == b && i_local >= j_local) continue;

                int idx_j = cell_indices[start_b + jb];

                // Compute Euclidean distance
                {0} dist_sq = ({0})(0.0);
                for (int feat = 0; feat < n_features; ++feat) {
                    {0} diff = embedding[idx_i * n_features + feat]
                             - embedding[idx_j * n_features + feat];
                    dist_sq = fma(diff, diff, dist_sq);
                }
                local_sum += sqrt(dist_sq);
            }
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    // Block reduction
    static __shared__ {0} warp_sums[32];
    if ((thread_id & 31) == 0) warp_sums[thread_id >> 5] = local_sum;
    __syncthreads();

    if (thread_id < 32) {
        {0} val = (thread_id < (block_size >> 5)) ? warp_sums[thread_id] : ({0})(0.0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);

        // Atomic add to accumulate across all blocks for this pair
        if (thread_id == 0) {
            atomicAdd(&pairwise_sums[a * k + b], val);
            if (a != b) {
                atomicAdd(&pairwise_sums[b * k + a], val);
            }
        }
    }
}
"""

    kernel = cuda_kernel_factory(
        kernel_code,
        (dtype,),
        "compute_group_distances_fallback",
    )

    return kernel
