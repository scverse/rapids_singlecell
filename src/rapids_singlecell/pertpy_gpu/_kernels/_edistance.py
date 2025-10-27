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
 int n_features) {

    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    {0} local_sum = ({0})(0.0);

    const int a = pair_left[block_id];
    const int b = pair_right[block_id];

    const int start_a = cat_offsets[a];
    const int end_a = cat_offsets[a + 1];
    const int start_b = cat_offsets[b];
    const int end_b = cat_offsets[b + 1];

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    for (int ia = start_a + thread_id; ia < end_a; ia += block_size) {
        const int idx_i = cell_indices[ia];

        for (int jb = start_b; jb < end_b; ++jb) {
            const int idx_j = cell_indices[jb];

            {0} dist_sq = ({0})(0.0);
            for (int feat = 0; feat < n_features; ++feat) {
                {0} diff = embedding[idx_i * n_features + feat] -
                              embedding[idx_j * n_features + feat];
                dist_sq += diff * diff;
            }

            local_sum += sqrt(dist_sq);
        }
    }

    // --- warp-shuffle reduction -------------
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    // --- block reduce -----------------------
    static __shared__ {0} s[32];              // one per warp
    if ((threadIdx.x & 31) == 0) s[threadIdx.x>>5] = local_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        {0} val = (threadIdx.x < (blockDim.x>>5)) ? s[threadIdx.x] : ({0})(0.0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) {
            {0} mean = val/(({0})(n_a) * ({0})(n_b));
            pairwise_means[a * k + b] = mean;
            pairwise_means[b * k + a] = mean;
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
