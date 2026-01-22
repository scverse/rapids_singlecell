from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd

from rapids_singlecell.preprocessing._harmony._helper import (
    _create_category_index_mapping,
)
from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

from ._base_metric import BaseMetric
from ._kernels._edistance_fallback import (
    get_compute_group_distances_kernel_fallback,
)
from ._kernels._edistance_multiblock import (
    get_compute_group_distances_kernel_multiblock,
)

# Max shared memory to use (conservative, works on all modern GPUs)
_MAX_SHARED_MEM_BYTES = 48_000
_DEFAULT_TILE_SIZE = 32

if TYPE_CHECKING:
    from anndata import AnnData


class EDistanceMetric(BaseMetric):
    """
    GPU-accelerated Energy Distance metric.

    Energy distance is a statistical distance between probability distributions
    that generalizes the Euclidean distance to distributions. It is particularly
    useful for comparing groups of cells in high-dimensional spaces.

    Parameters
    ----------
    layer_key : str | None
        Key in adata.layers for cell data. Mutually exclusive with obsm_key.
    obsm_key : str | None
        Key in adata.obsm for embeddings (default: 'X_pca')
    kernel : str
        Kernel strategy: 'auto' or 'manual'.
        - 'auto': Dynamically choose optimal blocks_per_pair (default)
        - 'manual': Use the specified blocks_per_pair directly
    blocks_per_pair : int
        Number of blocks per pair (default: 32). For 'auto', this is the maximum.
        Higher values increase parallelism but add atomic overhead.

    References
    ----------
    Székely, G. J., & Rizzo, M. L. (2013).
    Energy statistics: A class of statistics based on distances.
    Journal of Statistical Planning and Inference, 143(8), 1249-1272.
    """

    def __init__(
        self,
        layer_key: str | None = None,
        obsm_key: str | None = "X_pca",
        kernel: str = "auto",
        blocks_per_pair: int = 32,
    ):
        """Initialize energy distance metric."""
        super().__init__(obsm_key=obsm_key)
        self.layer_key = layer_key
        self.kernel = kernel
        self.blocks_per_pair = blocks_per_pair

    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        *,
        groups: list[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        inplace: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute pairwise energy distances between all cell groups.

        Returns a DataFrame containing pairwise distances. When bootstrap=True,
        returns a tuple of (distances, distances_var) DataFrames.

        The distances DataFrame contains:
        distances[a,b] = 2*d[a,b] - d[a] - d[b]

        When bootstrap=True, distances_var contains:
        distances_var[a,b] = 4*d_var[a,b] + d_var[a] + d_var[b]

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        groupby : str
            Key in adata.obs for grouping
        groups : list[str] | None
            Specific groups to compute (if None, use all)
        bootstrap : bool
            Whether to compute bootstrap variance estimates
        n_bootstrap : int
            Number of bootstrap iterations (if bootstrap=True)
        random_state : int
            Random seed for reproducibility
        inplace : bool
            Whether to store results in adata.uns

        Returns
        -------
        result : pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
            DataFrame with pairwise distances. If bootstrap=True, returns
            tuple of (distances, distances_var) DataFrames.
        """
        _assert_categorical_obs(adata, key=groupby)

        embedding = self._get_embedding(adata)
        original_groups = adata.obs[groupby]
        group_map = {v: i for i, v in enumerate(original_groups.cat.categories.values)}
        group_labels = cp.array([group_map[c] for c in original_groups], dtype=cp.int32)

        # Use harmony's category mapping
        k = len(group_map)
        cat_offsets, cell_indices = _create_category_index_mapping(group_labels, k)

        all_groups = list(original_groups.cat.categories.values)
        groups_list = all_groups if groups is None else groups

        if not bootstrap:
            df = self._prepare_edistance_df(
                embedding=embedding,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                k=k,
                all_groups=all_groups,
                groups_list=groups_list,
                groupby=groupby,
            )

            if inplace:
                adata.uns[f"{groupby}_pairwise_edistance"] = {"distances": df}

            return df

        else:
            df, df_var = self._prepare_edistance_df_bootstrap(
                embedding=embedding,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                k=k,
                all_groups=all_groups,
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

    def onesided_distances(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: str,
        *,
        groups: list[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> pd.Series:
        """
        Compute energy distances from one selected group to all other groups.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        groupby : str
            Key in adata.obs for grouping cells
        selected_group : str
            Reference group to compute distances from
        groups : list[str] | None
            Specific groups to compute distances to (if None, use all)
        bootstrap : bool
            Whether to compute bootstrap mean (if True, returns bootstrap mean)
        n_bootstrap : int
            Number of bootstrap iterations (if bootstrap=True)
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        distances : pd.Series
            Series containing distances from selected_group to all other groups
        """
        # For bootstrap, fall back to pairwise (computes all pairs)
        if bootstrap:
            df, df_var = self.pairwise(
                adata=adata,
                groupby=groupby,
                groups=groups,
                bootstrap=True,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                inplace=False,
            )
            if selected_group not in df.index:
                raise ValueError(
                    f"Selected group '{selected_group}' not found in groupby '{groupby}'"
                )
            return df.loc[selected_group]

        # Optimized path: only compute k pairs instead of k*(k+1)/2
        _assert_categorical_obs(adata, key=groupby)

        embedding = self._get_embedding(adata)
        original_groups = adata.obs[groupby]
        group_map = {v: i for i, v in enumerate(original_groups.cat.categories.values)}

        if selected_group not in group_map:
            raise ValueError(
                f"Selected group '{selected_group}' not found in groupby '{groupby}'"
            )

        group_labels = cp.array([group_map[c] for c in original_groups], dtype=cp.int32)
        k = len(group_map)
        cat_offsets, cell_indices = _create_category_index_mapping(group_labels, k)

        all_groups = list(original_groups.cat.categories.values)
        groups_list = all_groups if groups is None else groups

        # Compute onesided means (only pairs involving selected_group)
        selected_idx = group_map[selected_group]
        onesided_means = self._onesided_means(
            embedding, cat_offsets, cell_indices, k, selected_idx
        )

        # Compute energy distances: e[s,b] = 2*d[s,b] - d[s,s] - d[b,b]
        diag = cp.diag(onesided_means)
        edistances = 2 * onesided_means[selected_idx, :] - diag[selected_idx] - diag
        edistances[selected_idx] = 0.0  # Self-distance is 0

        # Create Series
        series = pd.Series(edistances.get(), index=all_groups, name=selected_group)
        series.index.name = groupby

        # Filter to requested groups if needed
        if groups_list != all_groups:
            series = series.loc[groups_list]

        return series

    def bootstrap(
        self,
        adata: AnnData,
        groupby: str,
        group_a: str,
        group_b: str,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[float, float]:
        """
        Compute bootstrap mean and variance for energy distance between two groups.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        groupby : str
            Key in adata.obs for grouping cells
        group_a : str
            First group name
        group_b : str
            Second group name
        n_bootstrap : int
            Number of bootstrap iterations
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        mean : float
            Bootstrap mean distance
        variance : float
            Bootstrap variance
        """
        # Compute pairwise with bootstrap
        df, df_var = self.pairwise(
            adata=adata,
            groupby=groupby,
            groups=[group_a, group_b],
            bootstrap=True,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            inplace=False,
        )

        mean = df.loc[group_a, group_b]
        variance = df_var.loc[group_a, group_b]

        return float(mean), float(variance)

    # Helper methods

    def _get_embedding(self, adata: AnnData) -> cp.ndarray:
        """Get embedding from adata using layer_key or obsm_key."""
        if self.layer_key:
            data = adata.layers[self.layer_key]
        else:
            data = adata.obsm[self.obsm_key]

        # Convert to cupy array if needed
        if isinstance(data, cp.ndarray):
            return data.astype(np.float32)
        return cp.array(data).astype(np.float32)

    def compute_distance(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
    ) -> float:
        """
        Compute energy distance between two arrays directly.

        Parameters
        ----------
        X : np.ndarray | cp.ndarray
            First array of shape (n_samples_x, n_features)
        Y : np.ndarray | cp.ndarray
            Second array of shape (n_samples_y, n_features)

        Returns
        -------
        distance : float
            Energy distance between X and Y
        """
        # Convert to cupy arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        Y_gpu = cp.asarray(Y, dtype=cp.float32)

        if len(X_gpu) == 0 or len(Y_gpu) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        # Compute mean pairwise distances
        d_xy = self._mean_pairwise_distance(X_gpu, Y_gpu)
        d_xx = self._mean_pairwise_distance_within(X_gpu)
        d_yy = self._mean_pairwise_distance_within(Y_gpu)

        # Energy distance formula
        return float(2 * d_xy - d_xx - d_yy)

    def bootstrap_arrays(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[float, float]:
        """
        Compute bootstrap mean and variance for energy distance between arrays.

        Parameters
        ----------
        X : np.ndarray | cp.ndarray
            First array of shape (n_samples_x, n_features)
        Y : np.ndarray | cp.ndarray
            Second array of shape (n_samples_y, n_features)
        n_bootstrap : int
            Number of bootstrap iterations
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        mean : float
            Bootstrap mean distance
        variance : float
            Bootstrap variance
        """
        # Convert to cupy arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        Y_gpu = cp.asarray(Y, dtype=cp.float32)

        if len(X_gpu) == 0 or len(Y_gpu) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        rng = np.random.default_rng(random_state)

        distances = []
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            X_idx = rng.choice(len(X_gpu), size=len(X_gpu), replace=True)
            Y_idx = rng.choice(len(Y_gpu), size=len(Y_gpu), replace=True)

            X_boot = X_gpu[X_idx]
            Y_boot = Y_gpu[Y_idx]

            # Compute distance for this bootstrap sample
            d_xy = self._mean_pairwise_distance(X_boot, Y_boot)
            d_xx = self._mean_pairwise_distance_within(X_boot)
            d_yy = self._mean_pairwise_distance_within(Y_boot)

            distance = 2 * d_xy - d_xx - d_yy
            distances.append(float(distance))

        return float(np.mean(distances)), float(np.var(distances))

    def _mean_pairwise_distance(
        self,
        X: cp.ndarray,
        Y: cp.ndarray,
    ) -> float:
        """Compute mean pairwise Euclidean distance between X and Y."""
        # Use cdist-like computation on GPU
        # For moderate sizes, direct computation is fast
        # X: (n, d), Y: (m, d)
        # dist[i,j] = ||X[i] - Y[j]||

        n, d = X.shape
        _ = Y.shape[0]  # m, unused but documents Y shape

        # Compute squared distances using broadcasting
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        X_sq = cp.sum(X * X, axis=1, keepdims=True)  # (n, 1)
        Y_sq = cp.sum(Y * Y, axis=1, keepdims=True)  # (m, 1)
        XY = X @ Y.T  # (n, m)

        dist_sq = X_sq + Y_sq.T - 2 * XY
        dist_sq = cp.maximum(dist_sq, 0)  # Numerical stability
        distances = cp.sqrt(dist_sq)

        return float(cp.mean(distances))

    def _mean_pairwise_distance_within(self, X: cp.ndarray) -> float:
        """Compute mean pairwise Euclidean distance within X (upper triangle only)."""
        n = len(X)
        if n < 2:
            return 0.0

        # Compute all pairwise distances
        X_sq = cp.sum(X * X, axis=1, keepdims=True)
        dist_sq = X_sq + X_sq.T - 2 * (X @ X.T)
        dist_sq = cp.maximum(dist_sq, 0)
        distances = cp.sqrt(dist_sq)

        # Get upper triangle (excluding diagonal)
        triu_indices = cp.triu_indices(n, k=1)
        upper_distances = distances[triu_indices]

        return float(cp.mean(upper_distances))

    # Internal methods from original _edistance.py

    def _pairwise_means(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
    ) -> cp.ndarray:
        """Compute between-group mean distances for all group pairs."""
        n_cells, n_features = embedding.shape

        # Vectorized upper triangular indices (including diagonal)
        triu_indices = cp.triu_indices(k)
        pair_left = triu_indices[0].astype(cp.int32)
        pair_right = triu_indices[1].astype(cp.int32)
        num_pairs = len(pair_left)

        # Calculate optimal blocks_per_pair
        blocks_per_pair = self._calculate_blocks_per_pair(num_pairs)

        # Check if n_features is too large for tiled kernel
        dtype_size = cp.dtype(embedding.dtype).itemsize
        required_shared_mem = _DEFAULT_TILE_SIZE * n_features * dtype_size
        use_tiled = required_shared_mem <= _MAX_SHARED_MEM_BYTES

        pairwise_sums = cp.zeros((k, k), dtype=embedding.dtype)
        grid = (num_pairs, blocks_per_pair)
        block = (1024,)

        if use_tiled:
            kernel, shared_mem = get_compute_group_distances_kernel_multiblock(
                embedding.dtype, n_features, tile_size=_DEFAULT_TILE_SIZE
            )
            kernel(
                grid,
                block,
                (
                    embedding,
                    cat_offsets,
                    cell_indices,
                    pair_left,
                    pair_right,
                    pairwise_sums,
                    k,
                    n_features,
                    blocks_per_pair,
                ),
                shared_mem=shared_mem,
            )
        else:
            kernel = get_compute_group_distances_kernel_fallback(embedding.dtype)
            kernel(
                grid,
                block,
                (
                    embedding,
                    cat_offsets,
                    cell_indices,
                    pair_left,
                    pair_right,
                    pairwise_sums,
                    k,
                    n_features,
                    blocks_per_pair,
                ),
            )

        # Normalize sums to means
        group_sizes = cp.diff(cat_offsets)
        diag_counts = group_sizes * (group_sizes - 1) // 2
        cross_counts = cp.outer(group_sizes, group_sizes)
        norm_matrix = cross_counts.astype(embedding.dtype)
        cp.fill_diagonal(norm_matrix, diag_counts.astype(embedding.dtype))

        return pairwise_sums / norm_matrix

    def _calculate_blocks_per_pair(self, num_pairs: int) -> int:
        """Calculate optimal blocks_per_pair based on workload.

        Based on profiling insights:
        - Need ~300K+ total blocks for good GPU throughput on modern GPUs
        - blocks_per_pair=1 is efficient when num_pairs is already large

        Returns
        -------
        blocks_per_pair : int
            Optimal number of blocks per pair
        """
        if self.kernel == "manual":
            return self.blocks_per_pair

        # auto - calculate optimal blocks_per_pair
        # Target ~300K total blocks for good GPU utilization
        # From profiling: 320K blocks gave 66% throughput vs 22% with 20K blocks
        target_blocks = 300_000

        blocks_per_pair = max(1, (target_blocks + num_pairs - 1) // num_pairs)
        blocks_per_pair = min(blocks_per_pair, self.blocks_per_pair)

        return blocks_per_pair

    def _onesided_means(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        selected_idx: int,
    ) -> cp.ndarray:
        """Compute mean distances from selected group to all groups.

        Computes:
        - d[selected_idx, i] for all i (cross-distances)
        - d[i, i] for all i (self-distances needed for energy distance formula)
        """
        n_cells, n_features = embedding.shape

        # Need pairs for:
        # 1. (selected_idx, i) for all i - cross distances from selected group
        # 2. (i, i) for all i where i != selected_idx - self-distances for other groups
        # Note: (selected_idx, selected_idx) is already in set 1

        # Cross-distance pairs: (selected_idx, 0), ..., (selected_idx, k-1)
        cross_left = cp.full(k, selected_idx, dtype=cp.int32)
        cross_right = cp.arange(k, dtype=cp.int32)

        # Diagonal pairs for other groups (excluding selected_idx which is already computed)
        other_diag = cp.array(
            [i for i in range(k) if i != selected_idx], dtype=cp.int32
        )

        # Combine all pairs
        pair_left = cp.concatenate([cross_left, other_diag])
        pair_right = cp.concatenate([cross_right, other_diag])
        num_pairs = len(pair_left)

        # Calculate optimal blocks_per_pair
        blocks_per_pair = self._calculate_blocks_per_pair(num_pairs)

        # Check if n_features is too large for tiled kernel
        dtype_size = cp.dtype(embedding.dtype).itemsize
        required_shared_mem = _DEFAULT_TILE_SIZE * n_features * dtype_size
        use_tiled = required_shared_mem <= _MAX_SHARED_MEM_BYTES

        onesided_sums = cp.zeros((k, k), dtype=embedding.dtype)
        grid = (num_pairs, blocks_per_pair)
        block = (1024,)

        if use_tiled:
            kernel, shared_mem = get_compute_group_distances_kernel_multiblock(
                embedding.dtype, n_features, tile_size=_DEFAULT_TILE_SIZE
            )
            kernel(
                grid,
                block,
                (
                    embedding,
                    cat_offsets,
                    cell_indices,
                    pair_left,
                    pair_right,
                    onesided_sums,
                    k,
                    n_features,
                    blocks_per_pair,
                ),
                shared_mem=shared_mem,
            )
        else:
            kernel = get_compute_group_distances_kernel_fallback(embedding.dtype)
            kernel(
                grid,
                block,
                (
                    embedding,
                    cat_offsets,
                    cell_indices,
                    pair_left,
                    pair_right,
                    onesided_sums,
                    k,
                    n_features,
                    blocks_per_pair,
                ),
            )

        # Normalize sums to means
        group_sizes = cp.diff(cat_offsets)
        diag_counts = group_sizes * (group_sizes - 1) // 2
        cross_counts = cp.outer(group_sizes, group_sizes)
        norm_matrix = cross_counts.astype(embedding.dtype)
        cp.fill_diagonal(norm_matrix, diag_counts.astype(embedding.dtype))

        return onesided_sums / norm_matrix

    def _generate_bootstrap_indices(
        self,
        cat_offsets: cp.ndarray,
        k: int,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> list[list[cp.ndarray]]:
        """Generate bootstrap indices for all groups and all bootstrap iterations."""
        import numpy as np

        # Use same RNG logic as CPU code
        rng = np.random.default_rng(random_state)

        # Convert to numpy for CPU-based random generation
        cat_offsets_np = cat_offsets.get()

        bootstrap_indices = []

        for _ in range(n_bootstrap):
            group_indices = []

            for group_idx in range(k):
                start_idx = cat_offsets_np[group_idx]
                end_idx = cat_offsets_np[group_idx + 1]
                group_size = end_idx - start_idx

                if group_size > 0:
                    bootstrap_group_indices = rng.choice(
                        group_size, size=group_size, replace=True
                    )
                    group_indices.append(
                        cp.array(bootstrap_group_indices, dtype=cp.int32)
                    )
                else:
                    group_indices.append(cp.array([], dtype=cp.int32))

            bootstrap_indices.append(group_indices)

        return bootstrap_indices

    def _bootstrap_sample_cells_from_indices(
        self,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        bootstrap_group_indices: list[cp.ndarray],
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Bootstrap sample cells using pre-generated indices."""
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

    def _pairwise_means_bootstrap(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute bootstrap statistics for between-group distances."""
        # Generate all bootstrap indices upfront
        bootstrap_indices = self._generate_bootstrap_indices(
            cat_offsets, k, n_bootstrap, random_state
        )

        bootstrap_results = []

        for i in range(n_bootstrap):
            boot_cat_offsets, boot_cell_indices = (
                self._bootstrap_sample_cells_from_indices(
                    cat_offsets=cat_offsets,
                    cell_indices=cell_indices,
                    k=k,
                    bootstrap_group_indices=bootstrap_indices[i],
                )
            )

            pairwise_means = self._pairwise_means(
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

    def _prepare_edistance_df_bootstrap(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        all_groups: list[str],
        groups_list: list[str],
        groupby: str,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare bootstrap edistance DataFrames."""
        # Bootstrap computation
        pairwise_means_boot, pairwise_vars_boot = self._pairwise_means_bootstrap(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        # Vectorized edistance computation for means:
        # edistance[a, b] = 2 * pairwise_means[a, b] - pairwise_means[a, a] - pairwise_means[b, b]
        diag_means = cp.diag(pairwise_means_boot)
        edistance_means = (
            2 * pairwise_means_boot - diag_means[:, None] - diag_means[None, :]
        )
        cp.fill_diagonal(edistance_means, 0)

        # Vectorized variance computation (delta method approximation):
        # var[a, b] = 4 * var[a, b] + var[a, a] + var[b, b]
        diag_vars = cp.diag(pairwise_vars_boot)
        edistance_vars = (
            4 * pairwise_vars_boot + diag_vars[:, None] + diag_vars[None, :]
        )
        cp.fill_diagonal(edistance_vars, 0)

        # Create full DataFrames with all groups
        df_mean = pd.DataFrame(
            edistance_means.get(), index=all_groups, columns=all_groups
        )
        df_mean.index.name = groupby
        df_mean.columns.name = groupby
        df_mean.name = "pairwise edistance"

        df_var = pd.DataFrame(
            edistance_vars.get(), index=all_groups, columns=all_groups
        )
        df_var.index.name = groupby
        df_var.columns.name = groupby
        df_var.name = "pairwise edistance variance"

        # Filter to requested groups if needed
        if groups_list != all_groups:
            df_mean = df_mean.loc[groups_list, groups_list]
            df_var = df_var.loc[groups_list, groups_list]

        return df_mean, df_var

    def _prepare_edistance_df(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        all_groups: list[str],
        groups_list: list[str],
        groupby: str,
    ) -> pd.DataFrame:
        """Prepare edistance DataFrame."""
        # Compute means
        pairwise_means = self._pairwise_means(embedding, cat_offsets, cell_indices, k)

        # Vectorized edistance computation:
        # edistance[a, b] = 2 * pairwise_means[a, b] - pairwise_means[a, a] - pairwise_means[b, b]
        diag = cp.diag(pairwise_means)
        edistance_matrix = 2 * pairwise_means - diag[:, None] - diag[None, :]
        cp.fill_diagonal(edistance_matrix, 0)  # Self-distance is 0

        # Create full DataFrame with all groups
        df = pd.DataFrame(edistance_matrix.get(), index=all_groups, columns=all_groups)
        df.index.name = groupby
        df.columns.name = groupby
        df.name = "pairwise edistance"

        # Filter to requested groups if needed
        if groups_list != all_groups:
            df = df.loc[groups_list, groups_list]

        return df
