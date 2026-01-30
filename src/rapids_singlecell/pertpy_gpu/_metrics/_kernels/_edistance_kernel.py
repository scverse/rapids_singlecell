"""
GPU kernel for computing pairwise group distances.

Tiles over both cells and features for efficient shared memory usage.
Shared memory is constant: CELL_TILE * FEAT_TILE * dtype_size.
"""

from __future__ import annotations

import numpy as np
from cuml.common.kernel_utils import cuda_kernel_factory

from rapids_singlecell._utils import _get_device_attrs

# Common tile sizes for feature dimension
# Larger tiles = fewer iterations but more shared memory
TILE_SIZES = [32, 50, 64]

# Feature tile sizes for CELL_TILE=64 configuration (Ampere+)
# 25 is optimal for common PC counts (50, 100), 16 for better coalescing otherwise
FEAT_TILE_64_PREFERRED = 25
FEAT_TILE_64_COALESCED = 16


def _choose_feat_tile_64(n_features: int) -> int:
    """
    Choose feat_tile for CELL_TILE=64 configuration.

    Uses 25 for common PC counts (divisible by 25), otherwise 16 for coalescing.

    Parameters
    ----------
    n_features
        Number of features in the embedding

    Returns
    -------
    int
        Optimal feat_tile (25 or 16)
    """
    if n_features % FEAT_TILE_64_PREFERRED == 0:
        return FEAT_TILE_64_PREFERRED
    return FEAT_TILE_64_COALESCED


def _choose_feat_tile(
    n_features: int, max_shared_bytes: int, cell_tile: int, dtype_size: int
) -> int:
    """
    Choose optimal feat_tile based on n_features and shared memory limits.

    Prioritizes exact divisibility to avoid partially-filled final tiles,
    which incur full synchronization overhead for little work.

    Parameters
    ----------
    n_features
        Number of features in the embedding
    max_shared_bytes
        Maximum shared memory available per block
    cell_tile
        Cell tile size (affects shared memory usage)
    dtype_size
        Size of data type in bytes (4 for float32, 8 for float64)

    Returns
    -------
    int
        Optimal feat_tile that fits in shared memory
    """
    # Filter tile sizes that fit in shared memory
    # Shared memory usage: cell_tile * feat_tile * dtype_size + overhead for warp_sums
    warp_sums_overhead = 32 * dtype_size  # warp_sums[32] in kernel
    available_shared = max_shared_bytes - warp_sums_overhead

    valid_tiles = [
        t for t in TILE_SIZES if cell_tile * t * dtype_size <= available_shared
    ]

    if not valid_tiles:
        # Fallback: compute max possible tile size
        max_feat_tile = available_shared // (cell_tile * dtype_size)
        return max(16, max_feat_tile)  # Minimum tile of 16

    # Check exact divisibility - prefer larger tiles (fewer iterations)
    for tile in reversed(valid_tiles):
        if n_features % tile == 0:
            return tile

    # No exact match - minimize (n_tiles, waste)
    best_tile = valid_tiles[0]
    best_n_tiles = (n_features + best_tile - 1) // best_tile
    best_waste = best_tile - (n_features % best_tile or best_tile)

    for tile in valid_tiles[1:]:
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
) -> tuple[object, int, int]:
    """
    Compile GPU kernel for computing pairwise group distances.

    Parameters
    ----------
    dtype
        Data type for embeddings (float32 or float64)
    n_features
        Number of features per cell

    Returns
    -------
    kernel
        Compiled CUDA kernel
    shared_mem_bytes
        Required shared memory in bytes
    block_size
        Recommended block size (threads per block)
    """
    dtype = np.dtype(dtype)
    is_double = dtype == np.float64
    sqrt_fn = "sqrt" if is_double else "sqrtf"
    dtype_size = dtype.itemsize

    device_attrs = _get_device_attrs()
    max_shared = device_attrs["max_shared_mem"]
    is_ampere_plus = device_attrs["cc_major"] >= 8

    # Cell tile and block size based on dtype and compute capability
    # Register pressure per block: cell_tile * block_size (for dist_sq array)
    # 64 * 512 = 32 * 1024 = 32,768 registers/block
    if is_double:
        cell_tile = 16
        block_size = 256 if not is_ampere_plus else 1024
        feat_tile = _choose_feat_tile(n_features, max_shared, cell_tile, dtype_size)
    else:
        # float32: use CELL_TILE=64 with block_size=512
        # Same register pressure as CELL_TILE=32 with block_size=1024, but faster
        cell_tile = 64
        block_size = 512
        feat_tile = _choose_feat_tile_64(n_features)

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

    shared_mem_bytes = cell_tile * feat_tile * dtype_size

    return kernel, shared_mem_bytes, block_size
