from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numba.types as nt
import numpy as np
from cuml.metrics import pairwise_distances
from spatialdata import SpatialData
from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_spatial_basis,
    _save_data,
)

it = nt.int32
ft = nt.float32
tt = nt.UniTuple
ip = np.int32
fp = np.float32
bl = nt.boolean

if TYPE_CHECKING:
    from anndata import AnnData
    from squidpy._utils import NDArrayA


def co_occurrence(
    adata: AnnData | SpatialData,
    cluster_key: str,
    spatial_key: str = Key.obsm.spatial,
    interval: int | NDArrayA = 50,
    copy: bool = False,
) -> tuple[NDArrayA, NDArrayA] | None:
    """
    Compute co-occurrence probability of clusters.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(spatial_key)s
    interval
        Distances interval at which co-occurrence is computed. If :class:`int`, uniformly spaced interval
        of the given size will be used.
    %(copy)s
    n_splits
        Number of splits in which to divide the spatial coordinates in
        :attr:`anndata.AnnData.obsm` ``['{spatial_key}']``.
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns the co-occurrence probability and the distance thresholds intervals.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['occ']`` - the co-occurrence probabilities
          across interval thresholds.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['interval']`` - the distance thresholds
          computed at ``interval``.
    """

    if isinstance(adata, SpatialData):
        adata = adata.table
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)

    spatial = cp.array(adata.obsm[spatial_key]).astype(np.float32)
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    labs = cp.array([clust_map[c] for c in original_clust], dtype=ip)

    # create intervals thresholds
    if isinstance(interval, int):
        thresh_min, thresh_max = _find_min_max(spatial)
        interval = cp.linspace(thresh_min, thresh_max, num=interval, dtype=fp)
    else:
        interval = cp.array(sorted(interval), dtype=fp, copy=True)
    if len(interval) <= 1:
        raise ValueError(
            f"Expected interval to be of length `>= 2`, found `{len(interval)}`."
        )

    spatial_x = spatial[:, 0].copy()
    spatial_y = spatial[:, 1].copy()

    # Compute co-occurrence probabilities using the fast numba routine.
    out = _co_occurrence_helper(spatial_x, spatial_y, interval, labs)

    if copy:
        return out, interval

    _save_data(
        adata,
        attr="uns",
        key=Key.uns.co_occurrence(cluster_key),
        data={"occ": out, "interval": interval},
    )


def _find_min_max(spatial: NDArrayA) -> tuple[float, float]:
    coord_sum = cp.sum(spatial, axis=1)
    min_idx, min_idx2 = cp.argpartition(coord_sum, 2)[:2]
    max_idx = cp.argmax(coord_sum)
    # fmt: off
    thres_max = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1))[0, 0] / 2.0
    thres_min = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[min_idx2, :].reshape(1, -1))[0, 0]
    # fmt: on

    return thres_min.astype(fp), thres_max.astype(fp)


def _co_occurrence_helper(
    v_x: NDArrayA, v_y: NDArrayA, v_radium: NDArrayA, labs: NDArrayA
) -> NDArrayA:
    """
    Fast co-occurrence probability computation using the new numba-accelerated counting.

    Parameters
    ----------
    v_x : np.ndarray, float64
         x–coordinates.
    v_y : np.ndarray, float64
         y–coordinates.
    v_radium : np.ndarray, float64
         Distance thresholds (in ascending order).
    labs : np.ndarray
         Cluster labels (as integers).

    Returns
    -------
    occ_prob : np.ndarray
         A 3D array of shape (k, k, len(v_radium)-1) containing the co-occurrence probabilities.
    labs_unique : np.ndarray
         Array of unique labels.
    """
    n = len(v_x)
    labs_unique = cp.unique(labs)
    k = len(labs_unique)
    # l_val is the number of bins; here we assume the thresholds come from v_radium[1:].
    l_val = len(v_radium) - 1
    # Compute squared thresholds from the interval (skip the first value)
    thresholds = (v_radium[1:]) ** 2

    grid = (n, n)
    block = (32,)
    # Allocate the result array as a 3D array with shape (k, k, l_val)
    counts = cp.zeros((l_val, k, k), dtype=cp.int32)
    # Use its flat view for the kernel.
    occur_count_kernel(grid, block, (v_x, v_y, thresholds, labs, counts, n, k, l_val))
    occ_prob = cp.empty((k, k, l_val), dtype=np.float32)
    shared_mem_size = (k * k + k) * cp.dtype("float32").itemsize
    blocks = l_val
    threads = 32
    occur_count_kernel2(
        (blocks,), (threads,), (counts, occ_prob, k, l_val), shared_mem=shared_mem_size
    )
    return occ_prob


kernel_code = r"""
extern "C" __global__
void occur_count_kernel(const float* spatial_x, const float* spatial_y,
                          const float* thresholds, const int* label_idx,
                          int* result, int n, int k, int l_val)
{
    // Each block corresponds to a unique point pair (i, j).
    int i = blockIdx.x;
    int j = blockIdx.y;

    // Skip self-comparisons or out-of-bound indices.
    if (i >= n || j >= n || i == j)
        return;

    // Compute squared distance.
    float dx = spatial_x[i] - spatial_x[j];
    float dy = spatial_y[i] - spatial_y[j];
    float dist_sq = dx * dx + dy * dy;

    // Get labels for both points.
    int label_i = label_idx[i];
    int label_j = label_idx[j];

    // Each thread loops over thresholds with a stride.
    for (int r = threadIdx.x; r < l_val; r += blockDim.x) {
        if (dist_sq <= thresholds[r]) {
            // Compute the flat index corresponding to result[label_i, label_j, r]
            int index = r * (k * k) + label_idx[i] * k + label_idx[j];
            atomicAdd(&result[index], 1);
        }
    }
}
"""

# Compile the kernel.
occur_count_kernel = cp.RawKernel(kernel_code, "occur_count_kernel")


kernel_code2 = r"""
extern "C" __global__
void coocur_part2(const int* result, float *out, int k, int l_val)
{
    // Each block handles one threshold index.
    int r_th = blockIdx.x;  // threshold index

    // Shared memory:
    //  - First k*k floats: will hold the counts matrix (converted to float).
    //  - Next k floats: used to store column sums.
    extern __shared__ float shared[];
    float* Y = shared;            // normalized matrix (will overwrite counts with counts/total)
    float* col_sum = shared + (k * k);  // column sums of Y

    int total_elements = k * k;

    // --- Load counts for this threshold and convert to float ---
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        Y[idx] = result[r_th * total_elements + idx];
    }
    __syncthreads();

    // --- Compute total sum of the counts ---
    __shared__ float total;
    float sum_val = 0.0f;
    for (int idx = threadIdx.x; idx < k*k; idx += blockDim.x) {
        sum_val += Y[idx];
    }

    // Use warp-level reduction; since blockDim is 32, all threads are in one warp.
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }

    // Now, thread 0 in the warp holds the total.
    if (threadIdx.x == 0) {
        total = sum_val;
    }
    __syncthreads();

    // --- Normalize the matrix Y = Y / total (if total > 0) ---
    if (total > 0.0f) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            Y[idx] = Y[idx] / total;
        }
    }
    else{
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                out[i * (k * l_val) + j * l_val + r_th] = 0.0f;
            }
        }
        return;
    }
    __syncthreads();

    // --- Compute column sums of the normalized matrix ---
    // For each column j, sum over rows i: col_sum[j] = sum_{i=0}^{k-1} Y[i*k + j]
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float sum_col = 0.0f;
        for (int i = 0; i < k; i++) {
            sum_col += Y[i * k + j];
        }
        col_sum[j] = sum_col;
    }
    __syncthreads();

    // --- Compute conditional probabilities ---
    // For each row i, compute the row sum (over columns) then for each column j:
    //   cond = (row sum != 0) ? (Y[i*k+j] / row_sum) : 0.0f
    //   final = (col_sum[j] != 0) ? (cond / col_sum[j]) : 0.0f
    // We then store the result into out with layout (k, k, l_val) in C-order.
    // The correct linear index for element (i, j, r_th) is: i*(k*l_val) + j*l_val + r_th.
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float row_sum = 0.0f;
        for (int j = 0; j < k; j++) {
            row_sum += Y[i * k + j];
        }
        for (int j = 0; j < k; j++) {
            float cond = 0.0f;
            if (row_sum != 0.0f)
                cond = Y[i * k + j] / row_sum;
            float final_val = 0.0f;
            if (col_sum[j] != 0.0f)
                final_val = cond / col_sum[j];
            // Write to output: note the ordering (row, column, threshold)
            out[i * (k * l_val) + j * l_val + r_th] = final_val;
        }
    }
}
"""
occur_count_kernel2 = cp.RawKernel(kernel_code2, "coocur_part2")
