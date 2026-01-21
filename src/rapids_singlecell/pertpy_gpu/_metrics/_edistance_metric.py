from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import cupy as cp
import numpy as np
import pandas as pd

from rapids_singlecell.preprocessing._harmony._helper import (
    _create_category_index_mapping,
)
from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

from ._base_metric import BaseMetric
from ._kernels._edistance import (
    get_compute_group_distances_kernel,
)

if TYPE_CHECKING:
    from anndata import AnnData


class EDistanceResult(NamedTuple):
    """Result object for energy distance computation."""

    distances: pd.DataFrame
    distances_var: pd.DataFrame | None


class EDistanceMetric(BaseMetric):
    """
    GPU-accelerated Energy Distance metric.

    Energy distance is a statistical distance between probability distributions
    that generalizes the Euclidean distance to distributions. It is particularly
    useful for comparing groups of cells in high-dimensional spaces.

    Parameters
    ----------
    obsm_key : str
        Key in adata.obsm for embeddings (default: 'X_pca')

    References
    ----------
    Székely, G. J., & Rizzo, M. L. (2013).
    Energy statistics: A class of statistics based on distances.
    Journal of Statistical Planning and Inference, 143(8), 1249-1272.
    """

    def __init__(self, obsm_key: str = "X_pca"):
        """Initialize energy distance metric."""
        super().__init__(obsm_key=obsm_key)

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
    ) -> EDistanceResult:
        """
        Compute pairwise energy distances between all cell groups.

        Returns EDistanceResult containing the distances and distances_var.
        The distances DataFrame is where:
        distances[a,b] = 2*d[a,b] - d[a] - d[b]
        The distances_var DataFrame is where:
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
        result : EDistanceResult
            EDistanceResult containing the distances and if bootstrap is True, the distances_var.
        """
        _assert_categorical_obs(adata, key=groupby)

        embedding = cp.array(adata.obsm[self.obsm_key]).astype(np.float32)
        original_groups = adata.obs[groupby]
        group_map = {v: i for i, v in enumerate(original_groups.cat.categories.values)}
        group_labels = cp.array([group_map[c] for c in original_groups], dtype=cp.int32)

        # Use harmony's category mapping
        k = len(group_map)
        cat_offsets, cell_indices = _create_category_index_mapping(group_labels, k)

        all_groups = list(original_groups.cat.categories.values)
        groups_list = all_groups if groups is None else groups

        result = None
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
            result = EDistanceResult(distances=df, distances_var=None)

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
            result = EDistanceResult(distances=df, distances_var=df_var)

        if inplace:
            adata.uns[f"{groupby}_pairwise_edistance"] = result._asdict()

        return result

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
        # Compute pairwise distances
        result = self.pairwise(
            adata=adata,
            groupby=groupby,
            groups=groups,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            inplace=False,
        )

        # Extract distances for the selected group
        if selected_group not in result.distances.index:
            raise ValueError(
                f"Selected group '{selected_group}' not found in groupby '{groupby}'"
            )

        return result.distances.loc[selected_group]

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
        result = self.pairwise(
            adata=adata,
            groupby=groupby,
            groups=[group_a, group_b],
            bootstrap=True,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            inplace=False,
        )

        mean = result.distances.loc[group_a, group_b]
        variance = result.distances_var.loc[group_a, group_b]

        return float(mean), float(variance)

    # Internal methods from original _edistance.py

    def _calculate_blocks_per_pair(
        self,
        num_pairs: int,
        target_multiplier: int = 4,
        max_blocks_per_pair: int = 32,
    ) -> int:
        """
        Calculate optimal blocks_per_pair for GPU utilization.

        Parameters
        ----------
        num_pairs : int
            Number of group pairs to process
        target_multiplier : int, default=4
            Multiplier for SM count to determine target total blocks.
            Higher values increase GPU utilization but may increase overhead.
        max_blocks_per_pair : int, default=32
            Maximum blocks per pair to avoid excessive atomic contention

        Returns
        -------
        int
            Optimal number of blocks to assign per pair

        Notes
        -----
        Strategy:
        - Target total blocks = target_multiplier * SM_count (e.g., 4x for ~100-200 SMs)
        - Distribute blocks evenly across pairs: target_blocks // num_pairs
        - Ensure at least 1 block per pair
        - Cap at max_blocks_per_pair to limit atomic contention overhead
        """
        device = cp.cuda.Device()
        sm_count = device.attributes["MultiProcessorCount"]
        target_blocks = target_multiplier * sm_count
        # Handle edge case where num_pairs is 0 or negative
        if num_pairs <= 0:
            return 1
        blocks_per_pair = max(1, target_blocks // num_pairs)
        return min(blocks_per_pair, max_blocks_per_pair)

    def _pairwise_means(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
    ) -> cp.ndarray:
        """Compute between-group mean distances for all group pairs."""
        _, n_features = embedding.shape

        # Vectorized upper triangular indices (including diagonal)
        triu_indices = cp.triu_indices(k)
        pair_left = triu_indices[0].astype(cp.int32)
        pair_right = triu_indices[1].astype(cp.int32)

        num_pairs = len(pair_left)

        # Allocate output - will accumulate sums, then normalize
        pairwise_sums = cp.zeros((k, k), dtype=embedding.dtype)

        # Determine blocks_per_pair for good GPU utilization
        blocks_per_pair = self._calculate_blocks_per_pair(num_pairs)

        compute_group_distances_kernel = get_compute_group_distances_kernel(
            embedding.dtype
        )

        # 2D grid: (num_pairs, blocks_per_pair)
        grid = (num_pairs, blocks_per_pair)
        block = (1024,)

        compute_group_distances_kernel(
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

        # Normalize sums to means on GPU
        group_sizes = cp.diff(cat_offsets)  # [k]
        norm_matrix = cp.outer(group_sizes, group_sizes).astype(
            embedding.dtype
        )  # [k, k]
        pairwise_means = pairwise_sums / norm_matrix

        return pairwise_means

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
