from __future__ import annotations

import cupy as cp
import numpy as np
from cuml.common.kernel_utils import cuda_kernel_factory

from rapids_singlecell._utils import _get_device_attrs

# Cell tile sizes for B-tile caching (2 floats per cell = 8 bytes)
# Larger tiles reduce iterations but increase shared memory usage
CELL_TILE_SIZES = [1024, 512, 256, 128, 64, 32, 16]

# Size constants
FLOAT32_SIZE = 4
INT32_SIZE = 4
WARP_SIZE = 32

# Block size thresholds based on cells per category (from benchmarking)
# Larger categories need more thread parallelism per block
# Smaller categories benefit from more grid parallelism (more blocks)
CELLS_PER_CAT_THRESHOLDS = [
    (10000, 1024),  # >= 10k cells/cat: use block_size=1024
    (5000, 512),  # >= 5k cells/cat: use block_size=512
    (2500, 256),  # >= 2.5k cells/cat: use block_size=256
    (0, 128),  # < 2.5k cells/cat: use block_size=128
]


def _compute_shared_mem(cell_tile: int, l_pad: int, warps_per_block: int) -> int:
    """Compute total shared memory needed for given configuration."""
    b_tile_bytes = cell_tile * 2 * FLOAT32_SIZE
    warp_hist_bytes = warps_per_block * l_pad * INT32_SIZE
    return b_tile_bytes + warp_hist_bytes


def _choose_block_size(n_cells: int, k: int) -> int:
    """Choose optimal block size based on cells per category.

    Parameters
    ----------
    n_cells
        Total number of cells
    k
        Number of categories

    Returns
    -------
    int
        Optimal block size
    """
    cells_per_cat = n_cells / k
    for threshold, block_size in CELLS_PER_CAT_THRESHOLDS:
        if cells_per_cat >= threshold:
            return block_size
    return 128  # fallback


def get_co_occurrence_kernel(
    l_val: int,
    n_cells: int,
    k: int,
    device_id: int | None = None,
) -> tuple[object, int, int, int, int] | None:
    """Compile GPU kernel for computing co-occurrence counts.

    Parameters
    ----------
    l_val
        Number of threshold bins
    n_cells
        Total number of cells (used for block size heuristic)
    k
        Number of categories (used for block size heuristic)
    device_id
        CUDA device ID. If None, uses current device.

    Returns
    -------
    kernel
        Compiled CUDA kernel
    shared_mem_bytes
        Required shared memory in bytes
    block_size
        Recommended block size (threads per block)
    cell_tile
        Cell tile size for B-tile caching
    l_pad
        Padded l_val for aligned histogram access
    """
    device_attrs = _get_device_attrs(device_id)
    max_shared = device_attrs["max_shared_mem"]
    l_pad = ((l_val + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE

    # Choose block size based on cells per category heuristic
    target_block_size = _choose_block_size(n_cells, k)

    # Find best configuration starting from target block size
    # Try target first, then fall back to smaller sizes if needed
    block_sizes = [target_block_size]
    for bs in (1024, 512, 256, 128, 64, 32):
        if bs not in block_sizes:
            block_sizes.append(bs)

    best_config = None
    for block_size in block_sizes:
        warps_per_block = block_size // WARP_SIZE
        for cell_tile in CELL_TILE_SIZES:
            total_shared = _compute_shared_mem(cell_tile, l_pad, warps_per_block)
            if total_shared <= max_shared:
                best_config = (block_size, cell_tile, total_shared)
                break  # Found best cell_tile for this block_size
        if best_config is not None:
            break  # Found valid config

    if best_config is None:
        return None

    block_size, cell_tile, total_shared = best_config

    kernel_code = f"""
(const float* __restrict__ spatial,
 const float* __restrict__ thresholds,
 const int* __restrict__ cat_offsets,
 const int* __restrict__ cell_indices,
 const int* __restrict__ pair_left,
 const int* __restrict__ pair_right,
 int* __restrict__ counts,
 int k,
 int l_val,
 int blocks_per_pair) {{

    // Shared memory layout:
    // - B-tile coords: CELL_TILE * 2 floats
    // - Warp histograms: warps_per_block * l_pad ints
    extern __shared__ char smem[];
    float* smem_b = (float*)smem;
    int* shared_hist = (int*)(smem + {cell_tile} * 2 * sizeof(float));

    const int l_pad = {l_pad};
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    int* warp_hist = shared_hist + warp_id * l_pad;

    // Zero per-warp histograms
    for (int r = lane; r < l_pad; r += 32) {{
        warp_hist[r] = 0;
    }}
    __syncthreads();

    const int pair_id = blockIdx.x;
    const int block_in_pair = blockIdx.y;

    const int a = pair_left[pair_id];
    const int b = pair_right[pair_id];

    const int start_a = cat_offsets[a];
    const int end_a = cat_offsets[a + 1];
    const int start_b = cat_offsets[b];
    const int end_b = cat_offsets[b + 1];

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    // Distribute A cells across blocks_per_pair
    const int total_threads_for_pair = blocks_per_pair * blockDim.x;
    const int global_thread_in_pair = block_in_pair * blockDim.x + threadIdx.x;

    const int is_diagonal = (a == b);

    // Tile over B cells with shared memory caching
    for (int jb_base = 0; jb_base < n_b; jb_base += {cell_tile}) {{
        const int cells_in_tile = min({cell_tile}, n_b - jb_base);

        // Cooperatively load B tile coords into shared memory
        for (int i = threadIdx.x; i < {cell_tile} * 2; i += blockDim.x) {{
            int cell = i / 2;
            int coord = i % 2;
            if (cell < cells_in_tile) {{
                int idx = cell_indices[start_b + jb_base + cell];
                smem_b[cell * 2 + coord] = spatial[idx * 2 + coord];
            }}
        }}
        __syncthreads();

        // Each thread processes a subset of A cells
        for (int ia = global_thread_in_pair; ia < n_a; ia += total_threads_for_pair) {{
            const int idx_i = cell_indices[start_a + ia];
            const float xi = spatial[idx_i * 2];
            const float yi = spatial[idx_i * 2 + 1];

            for (int c = 0; c < cells_in_tile; ++c) {{
                // For diagonal blocks (a == b), skip lower triangle + diagonal
                if (is_diagonal && ia >= jb_base + c) continue;

                const float dx = xi - smem_b[c * 2];
                const float dy = yi - smem_b[c * 2 + 1];
                const float dist_sq = dx * dx + dy * dy;

                // Binary search for threshold bin
                int lo = 0, hi = l_val;
                while (lo < hi) {{
                    int mid = (lo + hi) >> 1;
                    if (dist_sq <= thresholds[mid]) hi = mid;
                    else lo = mid + 1;
                }}
                if (lo < l_val) {{
                    atomicAdd(&warp_hist[lo], 1);
                }}
            }}
        }}
        __syncthreads();
    }}

    // Reduce warp histograms into block result
    if (warp_id == 0) {{
        // Sum each bin across warps into warp0's histogram
        for (int r = lane; r < l_pad; r += 32) {{
            int sum = 0;
            for (int w = 0; w < warps_per_block; ++w) {{
                sum += shared_hist[w * l_pad + r];
            }}
            shared_hist[r] = sum;
        }}
        __syncwarp();

        // Inclusive scan (cumulative) along thresholds
        if (threadIdx.x == 0) {{
            int acc = 0;
            for (int r = 0; r < l_val; ++r) {{
                acc += shared_hist[r];
                shared_hist[r] = acc;
            }}
        }}
        __syncwarp();

        // Atomic add to global counts (partial results from each block)
        for (int r = lane; r < l_val; r += 32) {{
            atomicAdd(&counts[a * (k * l_val) + b * l_val + r], shared_hist[r]);
        }}
    }}
}}
"""

    kernel = cuda_kernel_factory(
        kernel_code,
        (np.dtype("float32"),),
        f"co_oc_kernel_tile{cell_tile}_l{l_pad}",
    )

    return kernel, total_shared, block_size, cell_tile, l_pad


kernel_code_pairwise = r"""
extern "C" __global__
void occur_count_kernel_pairwise(const float* __restrict__ spatial,
                        const float* __restrict__ thresholds,
                        const int* __restrict__ label_idx,
                        int* __restrict__ result,
                        int n,
                        int k,
                        int l_val)
{
    int i = blockIdx.x;  // grid is 1D over n*n
    int s = i % 2;
    if (i >= n)
        return;
    int offset = (i % 4 < 2) ? 0 : l_val;
    float spx = spatial[i * 2];
    float spy = spatial[i * 2 + 1];
    int label_i = label_idx[i];

    for (int j = i + 1; j < n; j++) {
        float dx = spx - spatial[j * 2];
        float dy = spy - spatial[j * 2 + 1];
        float dist_sq = dx * dx + dy * dy;

        // Get labels for both points
        int low = label_i;
        int high = label_idx[j];

        // Sort labels if needed
        if (high < low) {
            int tmp = low;
            low = high;
            high = tmp;
        }

        // Swap based on s flag
        if (s == 0) {
            int tmp = low;
            low = high;
            high = tmp;
        }

        // Process each threshold in parallel within the block
        for (int r = threadIdx.x; r < l_val; r += blockDim.x) {
            if (dist_sq <= thresholds[r]) {
                int index = low * (k * l_val * 2) + high * l_val * 2 + r + offset;
                atomicAdd(&result[index], 1);
            }
        }
    }
}
"""
occur_count_kernel_pairwise = cp.RawKernel(
    kernel_code_pairwise, "occur_count_kernel_pairwise"
)


occur_reduction_kernel_code_shared = r"""
extern "C" __global__
void occur_reduction_kernel_shared(const int* __restrict__ result,
                            float* __restrict__ out,
                            int k,
                            int l_val,
                            int format)
{
    // Each block handles one threshold index.
    int r_th = blockIdx.x;  // threshold index

    // Shared memory allocation
    extern __shared__ float shared[];
    float* Y = shared;
    float* col_sum = shared + (k * k);

    int total_elements = k * k;

    // Initialize shared memory
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        Y[i] = 0.0f;
    }
    __syncthreads();

    // --- Load counts for this threshold and convert to float---
    if (format == 0){
        for (int i = threadIdx.x; i < k; i += blockDim.x){
            for (int j = 0; j<k;j++){
                Y[i * k + j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
                Y[j * k + i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
                Y[i * k + j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
                Y[j * k + i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
            }
        }
    }
    else{
        for (int i = threadIdx.x; i < k; i += blockDim.x){
            for (int j = 0; j<k;j++){
                float v = float(result[i * (k * l_val) + j * l_val + r_th]);
                Y[i * k + j] += v;
                Y[j * k + i] += v;
            }
        }
    }
    __syncthreads();

    // Compute total sum of the counts
    __shared__ float total;
    float sum_val = 0.0f;
    for (int idx = threadIdx.x; idx < k * k; idx += blockDim.x) {
        sum_val += Y[idx];
    }

    // Warp-level reduction
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }

    if (threadIdx.x == 0) {
        total = sum_val;
    }
    __syncthreads();

    // Normalize the matrix Y = Y / total (if total > 0)
    if (total > 0.0f) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            Y[idx] = Y[idx] / total;
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                out[i * (k * l_val) + j * l_val + r_th] = 0.0f;
            }
        }
        return;
    }
    __syncthreads();

    // Compute column sums of the normalized matrix
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float sum_col = 0.0f;
        for (int i = 0; i < k; i++) {
            sum_col += Y[i * k + j];
        }
        col_sum[j] = sum_col;
    }
    __syncthreads();

    // Compute conditional probabilities
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float row_sum = 0.0f;
        for (int j = 0; j < k; j++) {
            row_sum += Y[i * k + j];
        }

        for (int j = 0; j < k; j++) {
            float cond = 0.0f;
            if (row_sum != 0.0f) {
                cond = Y[i * k + j] / row_sum;
            }

            float final_val = 0.0f;
            if (col_sum[j] != 0.0f) {
                final_val = cond / col_sum[j];
            }

            // Write to output with (row, column, threshold) ordering
            out[i * (k * l_val) + j * l_val + r_th] = final_val;
        }
    }
    __syncthreads();
}
"""
occur_reduction_kernel_shared = cp.RawKernel(
    occur_reduction_kernel_code_shared, "occur_reduction_kernel_shared"
)

occur_reduction_kernel_code_global = r"""
extern "C" __global__
void occur_reduction_kernel_global(const int* __restrict__ result,
                            float* __restrict__ inter_out,
                            float* __restrict__ out,
                            int k,
                            int l_val,
                            int format)
{
    // Each block handles one threshold index.
    int r_th = blockIdx.x;  // threshold index
    if (r_th >= l_val)
        return;
    // Shared memory allocation
    extern __shared__ float shared[];
    float* Y = inter_out + r_th*k*k;
    float* col_sum = shared;

    int total_elements = k * k;

    // --- Load counts for this threshold and convert to float---
    if (format == 0){
        for (int i = threadIdx.x; i < k; i += blockDim.x){
            for (int j = 0; j<k;j++){
                Y[i * k + j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
                Y[j * k + i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th]);
                Y[i * k + j] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
                Y[j * k + i] += float(result[i * (k * l_val*2) + j * l_val*2 + r_th+l_val]);
            }
        }
    }
    else{
        for (int i = threadIdx.x; i < k; i += blockDim.x){
            for (int j = 0; j<k;j++){
                float v = float(result[i * (k * l_val) + j * l_val + r_th]);
                Y[i * k + j] += v;
                Y[j * k + i] += v;
            }
        }
    }
    __syncthreads();

    // Compute total sum of the counts
    __shared__ float total;
    float sum_val = 0.0f;
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        sum_val += Y[idx];
    }
    __syncthreads();
    // Warp-level reduction
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        total = sum_val;
    }
    __syncthreads();

    // Normalize the matrix Y = Y / total (if total > 0)
    if (total > 0.0f) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            Y[idx] = Y[idx] / total;
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                out[i * (k * l_val) + j * l_val + r_th] = 0.0f;
            }
        }
        return;
    }
    __syncthreads();

    // Compute column sums of the normalized matrix
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float sum_col = 0.0f;
        for (int i = 0; i < k; i++) {
            sum_col += Y[i * k + j];
        }
        col_sum[j] = sum_col;
    }
    __syncthreads();

    // Compute conditional probabilities
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float row_sum = 0.0f;
        for (int j = 0; j < k; j++) {
            row_sum += Y[i * k + j];
        }

        for (int j = 0; j < k; j++) {
            float cond = 0.0f;
            if (row_sum != 0.0f) {
                cond = Y[i * k + j] / row_sum;
            }

            float final_val = 0.0f;
            if (col_sum[j] != 0.0f) {
                final_val = cond / col_sum[j];
            }

            // Write to output with (row, column, threshold) ordering
            out[i * (k * l_val) + j * l_val + r_th] = final_val;
        }
    }
    __syncthreads();
}
"""
occur_reduction_kernel_global = cp.RawKernel(
    occur_reduction_kernel_code_global, "occur_reduction_kernel_global"
)
