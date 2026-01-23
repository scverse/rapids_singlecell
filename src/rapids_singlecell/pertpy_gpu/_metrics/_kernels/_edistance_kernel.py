"""
GPU kernel for computing pairwise group distances.

Tiles over both cells and features for efficient shared memory usage.
Shared memory is constant: CELL_TILE * FEAT_TILE * dtype_size.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
from cuml.common.kernel_utils import cuda_kernel_factory


def _choose_feat_tile(n_features: int) -> int:
    """
    Choose optimal feat_tile based on n_features to minimize wasted work.

    Prioritizes exact divisibility to avoid partially-filled final tiles,
    which incur full synchronization overhead for little work.

    Parameters
    ----------
    n_features : int
        Number of features in the embedding

    Returns
    -------
    int
        Optimal feat_tile (32, 50, or 64)
    """
    # Check exact divisibility - prefer larger tiles (fewer iterations)
    # 64 first since it gives fewest tiles when it divides evenly
    if n_features % 64 == 0:
        return 64
    if n_features % 50 == 0:
        return 50
    if n_features % 32 == 0:
        return 32

    # No exact match - minimize (n_tiles, waste)
    # Start with 50 as baseline (common PCA default)
    best_tile = 50
    best_n_tiles = (n_features + 49) // 50
    best_waste = 50 - (n_features % 50 or 50)

    for tile in [32, 64]:
        n_tiles = (n_features + tile - 1) // tile
        remainder = n_features % tile
        waste = 0 if remainder == 0 else tile - remainder

        # Prefer fewer tiles, then less waste
        if n_tiles < best_n_tiles or (n_tiles == best_n_tiles and waste < best_waste):
            best_tile = tile
            best_n_tiles = n_tiles
            best_waste = waste

    return best_tile


def get_compute_group_distances_kernel(
    dtype: np.dtype, n_features: int
) -> tuple[object, int]:
    """
    Compile GPU kernel for computing pairwise group distances.

    Parameters
    ----------
    dtype : dtype
        Data type for embeddings (float32 or float64)
    n_features : int
        Number of features per cell

    Returns
    -------
    kernel : callable
        Compiled CUDA kernel
    shared_mem_bytes : int
        Required shared memory in bytes
    """
    dtype = np.dtype(dtype)
    is_double = dtype == np.float64
    sqrt_fn = "sqrt" if is_double else "sqrtf"

    # Cell tile based on dtype to avoid register pressure
    cell_tile = 16 if is_double else 32

    # Auto-select feat_tile based on n_features
    feat_tile = _choose_feat_tile(n_features)

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

    // Shared memory for B tile: [FEAT_TILE][CELL_TILE] in row-major
    // But we store as [feat_chunk][cell] for coalesced access
    extern __shared__ {0} smem_b[];

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

        // Tile over B cells
        for (int jb_base = 0; jb_base < n_b; jb_base += CELL_TILE) {
            const int cells_in_tile = min(CELL_TILE, n_b - jb_base);

            // Accumulate squared distances for this cell tile
            {0} dist_sq[CELL_TILE];
            #pragma unroll
            for (int c = 0; c < CELL_TILE; ++c) dist_sq[c] = ({0})(0.0);

            // Tile over features
            for (int feat_base = 0; feat_base < n_features; feat_base += FEAT_TILE) {
                const int feats_in_tile = min(FEAT_TILE, n_features - feat_base);

                // Cooperatively load B tile [feat_chunk x cell_chunk] into shared memory
                const int total_elems = FEAT_TILE * CELL_TILE;
                for (int i = thread_id; i < total_elems; i += block_size) {
                    int cell_idx = i / FEAT_TILE;
                    int feat_idx = i % FEAT_TILE;
                    {0} val = ({0})(0.0);
                    if (cell_idx < cells_in_tile && feat_idx < feats_in_tile) {
                        int global_b_idx = cell_indices[start_b + jb_base + cell_idx];
                        val = embedding[global_b_idx * n_features + feat_base + feat_idx];
                    }
                    // Store as smem_b[feat][cell] for sequential access when iterating cells
                    smem_b[feat_idx * CELL_TILE + cell_idx] = val;
                }

                __syncthreads();

                // Compute partial squared differences for this feature chunk
                if (valid_a) {
                    for (int f = 0; f < feats_in_tile; ++f) {
                        {0} val_a = embedding[idx_i * n_features + feat_base + f];

                        #pragma unroll
                        for (int c = 0; c < CELL_TILE; ++c) {
                            {0} val_b = smem_b[f * CELL_TILE + c];
                            {0} diff = val_a - val_b;
                            dist_sq[c] += diff * diff;
                        }
                    }
                }

                __syncthreads();
            }

            // Now dist_sq[c] contains full squared distance for cell c in this tile
            // Accumulate sqrt(dist_sq) into local_sum
            if (valid_a) {
                #pragma unroll
                for (int c = 0; c < CELL_TILE; ++c) {
                    if (c >= cells_in_tile) break;
                    int j_local = jb_base + c;

                    // Skip lower triangle for diagonal blocks
                    if (a == b && i_local >= j_local) continue;

                    local_sum += SQRT_FN(dist_sq[c]);
                }
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

        if (thread_id == 0) {
            atomicAdd(&pairwise_sums[a * k + b], val);
            if (a != b) {
                atomicAdd(&pairwise_sums[b * k + a], val);
            }
        }
    }
}
"""
    # Replace compile-time constants
    kernel_code = kernel_code.replace("CELL_TILE", str(cell_tile))
    kernel_code = kernel_code.replace("FEAT_TILE", str(feat_tile))
    kernel_code = kernel_code.replace("SQRT_FN", sqrt_fn)

    precision_suffix = "f64" if is_double else "f32"
    kernel = cuda_kernel_factory(
        kernel_code,
        (dtype,),
        f"edistance_kernel_{cell_tile}x{feat_tile}_{precision_suffix}",
    )

    dtype_size = cp.dtype(dtype).itemsize
    shared_mem_bytes = cell_tile * feat_tile * dtype_size

    return kernel, shared_mem_bytes
