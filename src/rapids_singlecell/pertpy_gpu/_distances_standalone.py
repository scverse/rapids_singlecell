from __future__ import annotations

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

    with open(kernel_file) as f:
        kernel_code = f.read()

    # Compile kernels
    compute_group_distances_kernel = cp.RawKernel(
        kernel_code, "compute_group_distances"
    )

    return compute_group_distances_kernel


# Load kernels at module import time
compute_group_distances_kernel = _load_edistance_kernels()


def compute_pairwise_means_gpu(
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
    d_other_offdiag = cp.zeros(num_pairs, dtype=np.float64)

    # Choose optimal block size
    props = cp.cuda.runtime.getDeviceProperties(0)
    max_smem = int(props.get("sharedMemPerBlock", 48 * 1024))

    chosen_threads = None
    shared_mem_size = 0  # TODO: think of a better way to do this
    for tpb in (1024, 512, 256, 128, 64, 32):
        required = tpb * cp.dtype(cp.float64).itemsize
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
        (
            embedding,
            cat_offsets,
            cell_indices,
            pair_left,
            pair_right,
            d_other_offdiag,
            k,
            n_features,
        ),
        shared_mem=shared_mem_size,
    )

    # Build full k x k matrix
    pairwise_means = cp.zeros((k, k), dtype=np.float64)

    # Fill the full matrix
    for i, idx in enumerate(pair_indices.get()):
        a, b = divmod(idx, k)
        pairwise_means[a, b] = d_other_offdiag[i]
        pairwise_means[b, a] = d_other_offdiag[i]

    return pairwise_means


def generate_bootstrap_indices(
    cat_offsets: cp.ndarray,
    k: int,
    n_bootstrap: int = 100,
    random_state: int = 0,
) -> list[list[cp.ndarray]]:
    """
    Generate bootstrap indices for all groups and all bootstrap iterations.
    This matches the CPU implementation's random sampling logic for reproducibility.
    
    Parameters
    ----------
    cat_offsets : cp.ndarray
        Group start/end indices
    k : int
        Number of groups
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    bootstrap_indices : list[list[cp.ndarray]]
        For each bootstrap iteration, list of indices arrays for each group
        Shape: [n_bootstrap][k] where each element is cp.ndarray of group_size
    """
    import numpy as np
    
    # Use same RNG logic as CPU code
    rng = np.random.default_rng(random_state)
    
    # Convert to numpy for CPU-based random generation
    cat_offsets_np = cat_offsets.get()
    
    bootstrap_indices = []
    
    for bootstrap_iter in range(n_bootstrap):
        group_indices = []
        
        for group_idx in range(k):
            start_idx = cat_offsets_np[group_idx]
            end_idx = cat_offsets_np[group_idx + 1]
            group_size = end_idx - start_idx
            
            if group_size > 0:
                # Generate bootstrap indices using same logic as CPU code
                # rng.choice(a=X.shape[0], size=X.shape[0], replace=True)
                bootstrap_group_indices = rng.choice(
                    group_size, size=group_size, replace=True
                )
                # Convert to CuPy array
                group_indices.append(cp.array(bootstrap_group_indices, dtype=cp.int32))
            else:
                # Empty group
                group_indices.append(cp.array([], dtype=cp.int32))
        
        bootstrap_indices.append(group_indices)
    
    return bootstrap_indices


def _bootstrap_sample_cells_from_indices(
    *,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    k: int,
    bootstrap_group_indices: list[cp.ndarray],
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Bootstrap sample cells using pre-generated indices.
    
    Parameters
    ----------
    cat_offsets : cp.ndarray
        Group start/end indices
    cell_indices : cp.ndarray
        Sorted cell indices by group
    k : int
        Number of groups
    bootstrap_group_indices : list[cp.ndarray]
        Pre-generated bootstrap indices for each group
        
    Returns
    -------
    new_cat_offsets, new_cell_indices : tuple[cp.ndarray, cp.ndarray]
        New category structure with bootstrapped cells
    """
    new_cell_indices = []
    new_cat_offsets = cp.zeros(k + 1, dtype=cp.int32)
    
    for group_idx in range(k):
        start_idx = cat_offsets[group_idx]
        end_idx = cat_offsets[group_idx + 1]
        group_size = end_idx - start_idx
        
        if group_size > 0:
            # Get original cell indices for this group
            group_cells = cell_indices[start_idx:end_idx]
            
            # Use pre-generated bootstrap indices
            bootstrap_indices = bootstrap_group_indices[group_idx]
            bootstrap_cells = group_cells[bootstrap_indices]
            
            new_cell_indices.extend(bootstrap_cells.get().tolist())
        
        new_cat_offsets[group_idx + 1] = len(new_cell_indices)
    
    return new_cat_offsets, cp.array(new_cell_indices, dtype=cp.int32)


def compute_pairwise_means_gpu_bootstrap(
    embedding: cp.ndarray,
    *,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    k: int,
    n_bootstrap: int = 100,
    random_state: int = 0,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute bootstrap statistics for between-group distances.
    Uses CPU-compatible random generation for reproducibility.
    
    Returns:
        means: [k, k] matrix of bootstrap means
        variances: [k, k] matrix of bootstrap variances
    """
    # Generate all bootstrap indices upfront using CPU-compatible logic
    bootstrap_indices = generate_bootstrap_indices(
        cat_offsets, k, n_bootstrap, random_state
    )
    
    bootstrap_results = []
    
    for bootstrap_iter in range(n_bootstrap):
        # Use pre-generated indices for this bootstrap iteration
        boot_cat_offsets, boot_cell_indices = _bootstrap_sample_cells_from_indices(
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            bootstrap_group_indices=bootstrap_indices[bootstrap_iter],
        )
        
        # Compute distances with bootstrapped samples
        pairwise_means = compute_pairwise_means_gpu(
            embedding=embedding,
            cat_offsets=boot_cat_offsets,
            cell_indices=boot_cell_indices,
            k=k,
        )
        bootstrap_results.append(pairwise_means.get())
    
    # Compute statistics across bootstrap samples
    bootstrap_stack = cp.array(bootstrap_results)  # [n_bootstrap, k, k]
    means = cp.mean(bootstrap_stack, axis=0)
    variances = cp.var(bootstrap_stack, axis=0)
    
    return means, variances


def pairwise_edistance_gpu(
    adata: AnnData,
    groupby: str,
    *,
    obsm_key: str = "X_pca",
    groups: list[str] | None = None,
    inplace: bool = False,
    bootstrap: bool = False,
    n_bootstrap: int = 100,
    random_state: int = 0,
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

    embedding = cp.array(adata.obsm[obsm_key]).astype(np.float64)
    original_groups = adata.obs[groupby]
    group_map = {v: i for i, v in enumerate(original_groups.cat.categories.values)}
    group_labels = cp.array([group_map[c] for c in original_groups], dtype=cp.int32)

    # 2. Use harmony's category mapping
    k = len(group_map)
    cat_offsets, cell_indices = _create_category_index_mapping(group_labels, k)

    groups_list = (
        list(original_groups.cat.categories.values) if groups is None else groups
    )
    if not bootstrap:
        df = compute_pairwise_means_gpu_edistance(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            groups_list=groups_list,
            groupby=groupby,
        )
        if inplace:
            adata.uns[f"{groupby}_pairwise_edistance"] = {
                "distances": df,
            }
        return df

    else:
        df, df_var = compute_pairwise_means_gpu_edistance_bootstrap(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            groups_list=groups_list,
            groupby=groupby,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        if inplace:
            adata.uns[f"{groupby}_pairwise_edistance"] = {
                "distances": df,
                "distances_var": df_var,
            }
        return df, df_var


def compute_pairwise_means_gpu_edistance_bootstrap(
    embedding: cp.ndarray,
    *,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    k: int,
    groups_list: list[str],
    groupby: str,
    n_bootstrap: int = 100,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Bootstrap computation
    pairwise_means_boot, pairwise_vars_boot = compute_pairwise_means_gpu_bootstrap(
        embedding=embedding,
        cat_offsets=cat_offsets,
        cell_indices=cell_indices,
        k=k,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # 4. Compute final edistance for means and variances
    edistance_means = cp.zeros((k, k), dtype=np.float32)
    edistance_vars = cp.zeros((k, k), dtype=np.float32)

    for a in range(k):
        for b in range(a + 1, k):
            # Bootstrap mean edistance
            edistance_means[a, b] = (
                2 * pairwise_means_boot[a, b]
                - pairwise_means_boot[a, a]
                - pairwise_means_boot[b, b]
            )
            edistance_means[b, a] = edistance_means[a, b]

            # Bootstrap variance edistance (using delta method approximation)
            # Var(2*X - Y - Z) = 4*Var(X) + Var(Y) + Var(Z) (assuming independence)
            edistance_vars[a, b] = (
                4 * pairwise_vars_boot[a, b]
                + pairwise_vars_boot[a, a]
                + pairwise_vars_boot[b, b]
            )
            edistance_vars[b, a] = edistance_vars[a, b]

    # 5. Create output DataFrames

    df_mean = pd.DataFrame(
        edistance_means.get(), index=groups_list, columns=groups_list
    )
    df_mean.index.name = groupby
    df_mean.columns.name = groupby
    df_mean.name = "pairwise edistance"

    df_var = pd.DataFrame(edistance_vars.get(), index=groups_list, columns=groups_list)
    df_var.index.name = groupby
    df_var.columns.name = groupby
    df_var.name = "pairwise edistance variance"

    return df_mean, df_var


def compute_pairwise_means_gpu_edistance(
    embedding: cp.ndarray,
    *,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    k: int,
    groups_list: list[str],
    groupby: str,
) -> pd.DataFrame:
    # 3. Compute decomposed components
    # d_itself = compute_d_itself_gpu(embedding, cat_offsets, cell_indices, k)
    pairwise_means = compute_pairwise_means_gpu(embedding, cat_offsets, cell_indices, k)

    # 4. Compute final edistance: df[a,b] = 2*d_other[a,b] - d_itself[a] - d_itself[b]
    edistance_matrix = cp.zeros((k, k), dtype=np.float32)
    for a in range(k):
        for b in range(a + 1, k):
            edistance_matrix[a, b] = (
                2 * pairwise_means[a, b] - pairwise_means[a, a] - pairwise_means[b, b]
            )
            edistance_matrix[b, a] = edistance_matrix[a, b]

    # 5. Create output DataFrame

    df = pd.DataFrame(edistance_matrix.get(), index=groups_list, columns=groups_list)
    df.index.name = groupby
    df.columns.name = groupby
    df.name = "pairwise edistance"

    return df
