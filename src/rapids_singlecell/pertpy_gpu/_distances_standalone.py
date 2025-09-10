from __future__ import annotations

import os
from pathlib import Path
import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData

from ..preprocessing._harmony._helper import _create_category_index_mapping
from ..squidpy_gpu._utils import _assert_categorical_obs

# Load CUDA kernels from separate file
def _load_edistance_kernels():
    """Load CUDA kernels from separate .cu file"""
    kernel_dir = Path(__file__).parent / "kernels"
    kernel_file = kernel_dir / "edistance_kernels.cu"
    
    if not kernel_file.exists():
        raise FileNotFoundError(f"CUDA kernel file not found: {kernel_file}")
    
    with open(kernel_file, 'r') as f:
        kernel_code = f.read()
    
    # Compile kernels
    compute_group_distances_kernel = cp.RawKernel(kernel_code, "compute_group_distances")
    
    return compute_group_distances_kernel

# Load kernels at module import time
compute_group_distances_kernel = _load_edistance_kernels()



def compute_d_other_gpu(
    embedding: cp.ndarray, cat_offsets: cp.ndarray, cell_indices: cp.ndarray, k: int
) -> cp.ndarray:
    """
    Compute between-group mean distances for all group pairs.
    
    Parameters
    ----------
    embedding : cp.ndarray
        Cell embeddings [n_cells, n_features]
    cat_offsets : cp.ndarray
        Group start/end indices
    cell_indices : cp.ndarray
        Sorted cell indices by group
    k : int
        Number of groups
        
    Returns
    -------
    d_other : cp.ndarray
        Between-group mean distances [k, k]
    """
    _, n_features = embedding.shape
    
    pair_left = []
    pair_right = []
    pair_indices = []     
    # only upper triangle
    for a in range(k):
        for b in range(a, k):
            pair_left.append(a)
            pair_right.append(b)
            pair_indices.append(a * k + b)  # Flatten matrix index
    
    pair_left = cp.asarray(pair_left, dtype=cp.int32)
    pair_right = cp.asarray(pair_right, dtype=cp.int32)
    pair_indices = cp.asarray(pair_indices, dtype=cp.int32)
    
    num_pairs = len(pair_left)  # k * (k-1) pairs instead of k²

    # Allocate output for off-diagonal distances only
    d_other_offdiag = cp.zeros(num_pairs, dtype=np.float32)
    
    # Choose optimal block size
    props = cp.cuda.runtime.getDeviceProperties(0)
    max_smem = int(props.get("sharedMemPerBlock", 48 * 1024))

    chosen_threads = None
    shared_mem_size = 0 # TODO: think of a better way to do this
    for tpb in (1024, 512, 256, 128, 64, 32):
        required = tpb * cp.dtype(cp.float32).itemsize
        if required <= max_smem:
            chosen_threads = tpb
            shared_mem_size = required
            break

    # Launch kernel - one block per OFF-DIAGONAL group pair only
    grid = (num_pairs,)
    block = (chosen_threads,)
    compute_group_distances_kernel(
        grid,
        block,
        (embedding, cat_offsets, cell_indices, pair_left, pair_right, d_other_offdiag, k, n_features),
        shared_mem=shared_mem_size,
    )
    
    # Build full k x k matrix
    pairwise_means = cp.zeros((k, k), dtype=np.float32)
    
    # Fill the full matrix
    for i, idx in enumerate(pair_indices.get()):
        a, b = divmod(idx, k)
        pairwise_means[a, b] = d_other_offdiag[i]
        pairwise_means[b, a] = d_other_offdiag[i]
    
    
    return pairwise_means


def pairwise_edistance_gpu(
    adata: AnnData,
    groupby: str,
    *,
    obsm_key: str = "X_pca",
    groups: list[str] | None = None,
) -> pd.DataFrame:
    """
    GPU-accelerated pairwise edistance computation with decomposed components.
    
    Returns d_itself, d_other arrays and final edistance DataFrame where:
    df[a,b] = 2*d_other[a,b] - d_itself[a] - d_itself[b]
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    groupby : str
        Key in adata.obs for grouping
    obsm_key : str
        Key in adata.obsm for embeddings
    groups : list[str] | None
        Specific groups to compute (if None, use all)
    copy : bool
        Whether to return a copy
        
    Returns
    -------
    d_itself : cp.ndarray
        Within-group mean distances [k]
    d_other : cp.ndarray  
        Between-group mean distances [k, k]
    df : pd.DataFrame
        Final edistance matrix
    """
    # 1. Prepare data (same as original)
    _assert_categorical_obs(adata, key=groupby)

    embedding = cp.array(adata.obsm[obsm_key]).astype(np.float32)
    original_groups = adata.obs[groupby]
    group_map = {v: i for i, v in enumerate(original_groups.cat.categories.values)}
    group_labels = cp.array([group_map[c] for c in original_groups], dtype=cp.int32)

    # 2. Use harmony's category mapping
    k = len(group_map)
    cat_offsets, cell_indices = _create_category_index_mapping(group_labels, k)

    # 3. Compute decomposed components
    # d_itself = compute_d_itself_gpu(embedding, cat_offsets, cell_indices, k)
    pairwise_means = compute_d_other_gpu(embedding, cat_offsets, cell_indices, k)
    
    # 4. Compute final edistance: df[a,b] = 2*d_other[a,b] - d_itself[a] - d_itself[b]
    edistance_matrix = cp.zeros((k, k), dtype=np.float32)
    for a in range(k):
        for b in range(a+1, k):
            edistance_matrix[a, b] = 2 * pairwise_means[a, b] - pairwise_means[a, a] - pairwise_means[b, b]
            edistance_matrix[b, a] = edistance_matrix[a, b]

    # 5. Create output DataFrame
    groups_list = (
        list(original_groups.cat.categories.values) if groups is None else groups
    )
    df = pd.DataFrame(edistance_matrix.get(), index=groups_list, columns=groups_list)
    df.index.name = groupby
    df.columns.name = groupby
    df.name = "pairwise edistance"


    # Store in adata
    adata.uns[f"{groupby}_pairwise_edistance"] = {
        "distances": df,
    }
    
    return df
