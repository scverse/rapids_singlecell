from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_compute_group_distances_kernel = r"""
(const {0}* __restrict__ embedding,
 const int* __restrict__ cat_offsets,
 const int* __restrict__ cell_indices,
 const int* __restrict__ pair_left,
 const int* __restrict__ pair_right,
 {0}* __restrict__ pairwise_means,
 int k,
 int n_features,
 int blocks_per_pair) {

    const int thread_id = threadIdx.x;
    const int pair_id = blockIdx.x;
    const int block_id_in_pair = blockIdx.y;
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
    const int total_pairs = n_a * n_b;

    // Each thread computes distance for cell pairs using grid-stride loop
    const int global_tid = block_id_in_pair * block_size + thread_id;
    const int stride = blocks_per_pair * block_size;

    for (int cell_pair_id = global_tid; cell_pair_id < total_pairs; cell_pair_id += stride) {
        // Map thread to (i, j) cell pair
        const int i_local = cell_pair_id / n_b;
        const int j_local = cell_pair_id % n_b;

        // Get actual cell indices (lexsorted, so sequential access within group!)
        const int idx_i = cell_indices[start_a + i_local];
        const int idx_j = cell_indices[start_b + j_local];

        // Compute Euclidean distance with FMA
        {0} dist_sq = ({0})(0.0);
        #pragma unroll 4
        for (int feat = 0; feat < n_features; ++feat) {
            {0} diff = embedding[idx_i * n_features + feat] -
                          embedding[idx_j * n_features + feat];
            dist_sq = fma(diff, diff, dist_sq);
        }

        local_sum += sqrt(dist_sq);
    }

    // --- warp-shuffle reduction (FAST - register level) -------------
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    // --- cross-warp reduction via shared memory -----------------------
    static __shared__ {0} s[32];              // one per warp
    if ((threadIdx.x & 31) == 0) s[threadIdx.x>>5] = local_sum;
    __syncthreads();

    // Final reduction within block
    if (threadIdx.x < 32) {
        {0} val = (threadIdx.x < (blockDim.x>>5)) ? s[threadIdx.x] : ({0})(0.0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);

        // One atomic per block to accumulate partial sum
        if (threadIdx.x == 0) {
            atomicAdd(&pairwise_means[a * k + b], val);
            if (a != b) {
                atomicAdd(&pairwise_means[b * k + a], val);
            }
        }
    }
}
"""


def get_compute_group_distances_kernel(dtype):
    return cuda_kernel_factory(
        _compute_group_distances_kernel,
        (dtype,),
        "compute_group_distances",
    )
