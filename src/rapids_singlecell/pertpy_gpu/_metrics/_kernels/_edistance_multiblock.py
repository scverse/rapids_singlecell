"""
Multi-block kernel for computing pairwise group distances.

Uses a 2D grid with multiple blocks per pair to increase GPU utilization
for datasets with many small groups.
"""

from __future__ import annotations

import cupy as cp
from cuml.common.kernel_utils import cuda_kernel_factory


def get_compute_group_distances_kernel_multiblock(dtype, n_features, tile_size=32):
    """
    Compiles multi-block kernel with 2D grid for better GPU utilization.

    Uses multiple blocks per pair to increase parallelism for small groups.
    Returns sums (caller must normalize to means).

    Parameters
    ----------
    dtype : dtype
        Data type for embeddings
    n_features : int
        Number of features per cell
    tile_size : int
        Tile size for B group (default 32)

    Returns
    -------
    kernel : callable
        Compiled CUDA kernel
    shared_mem_bytes : int
        Required shared memory in bytes
    """

    kernel_code = f"""
#define TILE_SIZE {tile_size}

(const {{0}}* __restrict__ embedding,
 const int* __restrict__ cat_offsets,
 const int* __restrict__ cell_indices,
 const int* __restrict__ pair_left,
 const int* __restrict__ pair_right,
 {{0}}* __restrict__ pairwise_sums,
 int k,
 int n_features,
 int blocks_per_pair) {{

    extern __shared__ {{0}} smem_b[];

    const int thread_id = threadIdx.x;
    const int pair_id = blockIdx.x;
    const int block_in_pair = blockIdx.y;
    const int block_size = blockDim.x;

    {{0}} local_sum = ({{0}})(0.0);

    const int a = pair_left[pair_id];
    const int b = pair_right[pair_id];

    const int start_a = cat_offsets[a];
    const int end_a = cat_offsets[a + 1];
    const int start_b = cat_offsets[b];
    const int end_b = cat_offsets[b + 1];

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    // Distribute A cells across blocks_per_pair using iteration-based approach
    // to ensure all threads hit syncthreads together
    const int total_threads_for_pair = blocks_per_pair * block_size;
    const int global_thread_in_pair = block_in_pair * block_size + thread_id;
    const int n_iters_a = (n_a + total_threads_for_pair - 1) / total_threads_for_pair;

    for (int iter_a = 0; iter_a < n_iters_a; ++iter_a) {{
        const int ia = start_a + iter_a * total_threads_for_pair + global_thread_in_pair;
        const bool valid_a = (ia < end_a);
        const int idx_i = valid_a ? cell_indices[ia] : 0;
        const int i_local = ia - start_a;

        {{0}} dist_sum_for_i = ({{0}})(0.0);

        for (int jb_base = 0; jb_base < n_b; jb_base += TILE_SIZE) {{

            int items_in_tile = min(TILE_SIZE, n_b - jb_base);

            // Cooperatively load B tile into shared memory (ALL threads participate)
            int total_floats = items_in_tile * n_features;
            for (int i = thread_id; i < total_floats; i += block_size) {{
                int cell_idx = i / n_features;
                int feat_idx = i % n_features;
                int global_b_idx = cell_indices[start_b + jb_base + cell_idx];
                smem_b[cell_idx * n_features + feat_idx] =
                    embedding[global_b_idx * n_features + feat_idx];
            }}

            __syncthreads();

            // Compute distances (only valid threads)
            if (valid_a) {{
                {{0}} tile_acc[TILE_SIZE];

                #pragma unroll
                for (int t = 0; t < TILE_SIZE; ++t) tile_acc[t] = ({{0}})(0.0);

                for (int feat = 0; feat < n_features; ++feat) {{
                    {{0}} val_a = embedding[idx_i * n_features + feat];

                    #pragma unroll
                    for (int t = 0; t < items_in_tile; ++t) {{
                        {{0}} val_b = smem_b[t * n_features + feat];
                        {{0}} diff = val_a - val_b;
                        tile_acc[t] = fmaf(diff, diff, tile_acc[t]);
                    }}
                }}

                // Accumulate distances (skip lower triangle for diagonal blocks)
                #pragma unroll
                for (int t = 0; t < items_in_tile; ++t) {{
                    int j_local = jb_base + t;

                    if (a == b && i_local >= j_local) continue;

                    dist_sum_for_i += sqrtf(tile_acc[t]);
                }}
            }}

            __syncthreads();
        }}

        if (valid_a) {{
            local_sum += dist_sum_for_i;
        }}
    }}

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    // Block reduction
    static __shared__ {{0}} warp_sums[32];
    if ((thread_id & 31) == 0) warp_sums[thread_id >> 5] = local_sum;
    __syncthreads();

    if (thread_id < 32) {{
        {{0}} val = (thread_id < (block_size >> 5)) ? warp_sums[thread_id] : ({{0}})(0.0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);

        // Atomic add to accumulate across all blocks for this pair
        if (thread_id == 0) {{
            atomicAdd(&pairwise_sums[a * k + b], val);
            if (a != b) {{
                atomicAdd(&pairwise_sums[b * k + a], val);
            }}
        }}
    }}
}}
"""

    kernel = cuda_kernel_factory(
        kernel_code,
        (dtype,),
        f"compute_group_distances_multiblock_{tile_size}",
    )

    dtype_size = cp.dtype(dtype).itemsize
    shared_mem_bytes = tile_size * n_features * dtype_size

    return kernel, shared_mem_bytes
