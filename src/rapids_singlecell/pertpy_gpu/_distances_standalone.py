from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData

from ..preprocessing._harmony._helper import _create_category_index_mapping
from ..squidpy_gpu._utils import _assert_categorical_obs

# CUDA kernel for edistance computation - simplified direct approach
edistance_kernel_code = r"""
extern "C" __global__
void edistance_pairwise_kernel(
    const float* __restrict__ embedding,
    const int* __restrict__ cat_offsets,
    const int* __restrict__ cell_indices,
    const int* __restrict__ pair_left,
    const int* __restrict__ pair_right,
    float* __restrict__ edistances,
    int k,
    int n_features)
{
    extern __shared__ float shared_sums[];

    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    // Each thread accumulates partial sums for [within_A, within_B, between_AB]
    float local_within_A = 0.0f;
    float local_within_B = 0.0f;
    float local_between = 0.0f;

    const int a = pair_left[block_id];
    const int b = pair_right[block_id];

    const int start_a = cat_offsets[a];
    const int end_a = cat_offsets[a + 1];
    const int start_b = cat_offsets[b];
    const int end_b = cat_offsets[b + 1];

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    if (a == b) {
        // Same group: edistance = 0 by definition, but we still need to compute
        // within-group sum to match cuml behavior
        if (thread_id == 0) {
            edistances[block_id] = 0.0f;
        }

    } else {
        // Different groups: compute all three components and final edistance

        // 1. Compute within-group A distances (ALL pairs including symmetric)
        for (int ia = start_a + thread_id; ia < end_a; ia += block_size) {
            const int idx_i = cell_indices[ia];

            for (int ja = start_a; ja < end_a; ++ja) {
                const int idx_j = cell_indices[ja];

                if (idx_i != idx_j) {
                    float dist_sq = 0.0f;
                    for (int feat = 0; feat < n_features; ++feat) {
                        float diff = embedding[idx_i * n_features + feat] -
                                    embedding[idx_j * n_features + feat];
                        dist_sq += diff * diff;
                    }
                    local_within_A += sqrtf(dist_sq);
                }
            }
        }

        // 2. Compute within-group B distances (ALL pairs including symmetric)
        for (int ib = start_b + thread_id; ib < end_b; ib += block_size) {
            const int idx_i = cell_indices[ib];

            for (int jb = start_b; jb < end_b; ++jb) {
                const int idx_j = cell_indices[jb];

                if (idx_i != idx_j) {
                    float dist_sq = 0.0f;
                    for (int feat = 0; feat < n_features; ++feat) {
                        float diff = embedding[idx_i * n_features + feat] -
                                    embedding[idx_j * n_features + feat];
                        dist_sq += diff * diff;
                    }
                    local_within_B += sqrtf(dist_sq);
                }
            }
        }

        // 3. Compute between-group distances (ALL cross-pairs)
        for (int ia = start_a + thread_id; ia < end_a; ia += block_size) {
            const int idx_i = cell_indices[ia];

            for (int jb = start_b; jb < end_b; ++jb) {
                const int idx_j = cell_indices[jb];

                float dist_sq = 0.0f;
                for (int feat = 0; feat < n_features; ++feat) {
                    float diff = embedding[idx_i * n_features + feat] -
                                embedding[idx_j * n_features + feat];
                    dist_sq += diff * diff;
                }
                local_between += sqrtf(dist_sq);
            }
        }

        // Reduce within_A
        shared_sums[thread_id] = local_within_A;
        __syncthreads();
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (thread_id < stride) {
                shared_sums[thread_id] += shared_sums[thread_id + stride];
            }
            __syncthreads();
        }
        float total_within_A = shared_sums[0];
        __syncthreads();

        // Reduce within_B
        shared_sums[thread_id] = local_within_B;
        __syncthreads();
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (thread_id < stride) {
                shared_sums[thread_id] += shared_sums[thread_id + stride];
            }
            __syncthreads();
        }
        float total_within_B = shared_sums[0];
        __syncthreads();

        // Reduce between
        shared_sums[thread_id] = local_between;
        __syncthreads();
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (thread_id < stride) {
                shared_sums[thread_id] += shared_sums[thread_id + stride];
            }
            __syncthreads();
        }
        float total_between = shared_sums[0];

        // Compute final edistance directly
        if (thread_id == 0) {
            // Normalize by total matrix elements (matching cuml behavior)
            // cast to float
            float mean_within_A = total_within_A / ((float)(n_a * n_a));
            float mean_within_B = total_within_B / ((float)(n_b * n_b));
            float mean_between = total_between / ((float)(n_a * n_b) * 0.5f);

            // Edistance formula: 2*δ - σ_A - σ_B
            edistances[block_id] = mean_between - mean_within_A - mean_within_B;
        }
    }
}
"""

edistance_pairwise_kernel = cp.RawKernel(
    edistance_kernel_code, "edistance_pairwise_kernel"
)


def _fill_distance_matrix(
    edistances: cp.ndarray,
    pairwise_distances: cp.ndarray,
    pair_left: cp.ndarray,
    pair_right: cp.ndarray,
):
    """Fill the symmetric distance matrix from kernel output"""

    for i in range(pair_left.size):
        a, b = int(pair_left[i]), int(pair_right[i])
        edist = float(edistances[i])

        # Fill symmetric matrix
        pairwise_distances[a, b] = edist
        pairwise_distances[b, a] = edist


def _edistance_pairwise_helper(
    embedding: cp.ndarray, cat_offsets: cp.ndarray, cell_indices: cp.ndarray, k: int
) -> cp.ndarray:
    """
    Fast pairwise edistance computation using CUDA kernels.

    Parameters
    ----------
    embedding : cp.ndarray
        Cell embeddings [n_cells, n_features]
    cat_offsets : cp.ndarray
        Group start/end indices (from harmony helper)
    cell_indices : cp.ndarray
        Sorted cell indices by group (from harmony helper)
    k : int
        Number of groups

    Returns
    -------
    pairwise_distances : cp.ndarray
        Pairwise edistance matrix [k, k]
    """

    n_cells, n_features = embedding.shape

    # Build group pairs (same pattern as co_occurrence)
    pair_left = []
    pair_right = []
    for a in range(k):
        for b in range(a, k):  # Upper triangle
            pair_left.append(a)
            pair_right.append(b)
    pair_left = cp.asarray(pair_left, dtype=cp.int32)
    pair_right = cp.asarray(pair_right, dtype=cp.int32)

    # Allocate output for final edistances (one per pair)
    edistances = cp.zeros(pair_left.size, dtype=np.float32)

    # Choose optimal block size (same logic as co_occurrence)
    props = cp.cuda.runtime.getDeviceProperties(0)
    max_smem = int(props.get("sharedMemPerBlock", 48 * 1024))

    chosen_threads = None
    for tpb in (1024, 512, 256, 128, 64, 32):
        # Each thread needs one float for shared memory reduction
        required = tpb * cp.dtype(cp.float32).itemsize
        if required <= max_smem:
            chosen_threads = tpb
            shared_mem_size = required
            break

    # Launch kernel (similar pattern to co_occurrence)
    grid = (pair_left.size,)  # One block per group pair
    block = (chosen_threads,)
    edistance_pairwise_kernel(
        grid,
        block,
        (
            embedding,
            cat_offsets,
            cell_indices,
            pair_left,
            pair_right,
            edistances,
            k,
            n_features,
        ),
        shared_mem=shared_mem_size,
    )

    # Fill symmetric distance matrix
    pairwise_distances = cp.zeros((k, k), dtype=np.float32)
    _fill_distance_matrix(edistances, pairwise_distances, pair_left, pair_right)

    return pairwise_distances


def pairwise_edistance_gpu(
    adata: AnnData,
    groupby: str,
    *,
    obsm_key: str = "X_pca",
    groups: list[str] | None = None,
    copy: bool = False,
) -> pd.DataFrame | None:
    """GPU-accelerated pairwise edistance computation"""

    # 1. Prepare data (exactly like co_occurrence)
    _assert_categorical_obs(adata, key=groupby)  # Reuse validation

    embedding = cp.array(adata.obsm[obsm_key]).astype(np.float32)
    original_groups = adata.obs[groupby]
    group_map = {v: i for i, v in enumerate(original_groups.cat.categories.values)}
    group_labels = cp.array([group_map[c] for c in original_groups], dtype=np.int32)

    # 2. Use harmony's category mapping (same as co_occurrence)
    k = len(group_map)  # number of groups
    cat_offsets, cell_indices = _create_category_index_mapping(group_labels, k)

    # 3. Compute pairwise edistances using GPU kernels
    pairwise_distances = _edistance_pairwise_helper(
        embedding, cat_offsets, cell_indices, k
    )

    # 4. Create output DataFrame (same pattern as pertpy)
    groups_list = (
        list(original_groups.cat.categories.values) if groups is None else groups
    )
    df = pd.DataFrame(pairwise_distances.get(), index=groups_list, columns=groups_list)
    df.index.name = groupby
    df.columns.name = groupby
    df.name = "pairwise edistance"

    if copy:
        return df

    # Store in adata like co_occurrence does
    adata.uns[f"{groupby}_pairwise_edistance"] = {"distances": df}
    return df
