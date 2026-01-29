from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import cupy as cp
import numpy as np
import pandas as pd

from rapids_singlecell._utils import (
    _calculate_blocks_per_pair,
    _create_category_index_mapping,
    _split_pairs,
)
from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

from ._base_metric import BaseMetric, parse_device_ids
from ._kernels._edistance_kernel import (
    get_compute_group_distances_kernel,
)

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
    layer_key
        Key in adata.layers for cell data. Mutually exclusive with obsm_key.
    obsm_key
        Key in adata.obsm for embeddings (default: 'X_pca')

    References
    ----------
    Székely, G. J., & Rizzo, M. L. (2013).
    Energy statistics: A class of statistics based on distances.
    Journal of Statistical Planning and Inference, 143(8), 1249-1272.
    """

    supports_multi_gpu: bool = True

    def __init__(
        self,
        layer_key: str | None = None,
        obsm_key: str | None = "X_pca",
    ):
        """Initialize energy distance metric."""
        if layer_key is not None and obsm_key is not None:
            raise ValueError(
                "Cannot use 'layer_key' and 'obsm_key' at the same time. "
                "Please provide only one of the two keys."
            )
        super().__init__(obsm_key=obsm_key)
        self.layer_key = layer_key

    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
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
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping
        groups
            Specific groups to compute (if None, use all)
        bootstrap
            Whether to compute bootstrap variance estimates
        n_bootstrap
            Number of bootstrap iterations (if bootstrap=True)
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only GPU 0
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        result
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
            return self._prepare_edistance_df(
                embedding=embedding,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                k=k,
                all_groups=all_groups,
                groups_list=groups_list,
                groupby=groupby,
                multi_gpu=multi_gpu,
            )

        return self._prepare_edistance_df_bootstrap(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            all_groups=all_groups,
            groups_list=groups_list,
            groupby=groupby,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
        )

    def onesided_distances(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: str,
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """
        Compute energy distances from one selected group to all other groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        selected_group
            Reference group to compute distances from
        groups
            Specific groups to compute distances to (if None, use all)
        bootstrap
            Whether to compute bootstrap variance estimates
        n_bootstrap
            Number of bootstrap iterations (if bootstrap=True)
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only GPU 0
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        distances
            Series containing distances from selected_group to all other groups.
            If bootstrap=True, returns tuple of (distances, distances_var).
        """
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
        groups_list = all_groups if groups is None else list(groups)
        selected_idx = group_map[selected_group]

        device_ids = parse_device_ids(multi_gpu=multi_gpu)

        if bootstrap:
            onesided_means, onesided_vars = self._onesided_means_bootstrap(
                embedding=embedding,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                k=k,
                selected_idx=selected_idx,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                device_ids=device_ids,
            )

            # Compute energy distances: e[s,b] = 2*d[s,b] - d[s,s] - d[b,b]
            diag_means = cp.diag(onesided_means)
            row = onesided_means[selected_idx, :]
            edistances = 2 * row - diag_means[selected_idx] - diag_means
            edistances[selected_idx] = 0.0

            # Variance: var[s,b] = 4*var[s,b] + var[s,s] + var[b,b]
            diag_vars = cp.diag(onesided_vars)
            ed_vars = (
                4 * onesided_vars[selected_idx, :] + diag_vars[selected_idx] + diag_vars
            )
            ed_vars[selected_idx] = 0.0

            series_name = f"edistance to {selected_group}"
            distances = pd.Series(edistances.get(), index=all_groups, name=series_name)
            distances.index.name = groupby
            variances = pd.Series(ed_vars.get(), index=all_groups, name=series_name)
            variances.index.name = groupby

            if groups_list != all_groups:
                distances = distances.loc[groups_list]
                variances = variances.loc[groups_list]

            return distances, variances

        # Non-bootstrap path: compute onesided means directly
        onesided_means = self._onesided_means(
            embedding,
            cat_offsets,
            cell_indices,
            k,
            selected_idx=selected_idx,
            device_ids=device_ids,
        )

        # Compute energy distances: e[s,b] = 2*d[s,b] - d[s,s] - d[b,b]
        diag = cp.diag(onesided_means)
        edistances = 2 * onesided_means[selected_idx, :] - diag[selected_idx] - diag
        edistances[selected_idx] = 0.0  # Self-distance is 0

        # Create Series with pertpy-compatible name
        series = pd.Series(
            edistances.get(), index=all_groups, name=f"edistance to {selected_group}"
        )
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
        multi_gpu: bool | list[int] | str | None = None,
    ) -> tuple[float, float]:
        """
        Compute bootstrap mean and variance for energy distance between two groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        group_a
            First group name
        group_b
            Second group name
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only GPU 0
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        mean
            Bootstrap mean distance
        variance
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
            multi_gpu=multi_gpu,
        )

        mean = df.loc[group_a, group_b]
        variance = df_var.loc[group_a, group_b]

        return float(mean), float(variance)

    # Helper methods

    def _get_embedding(self, adata: AnnData) -> cp.ndarray:
        """Get embedding from adata using layer_key or obsm_key.

        Preserves the input dtype (float32 or float64) for precision control.
        """
        if self.layer_key:
            data = adata.layers[self.layer_key]
        else:
            data = adata.obsm[self.obsm_key]

        # Convert to cupy array if needed, preserving dtype
        if isinstance(data, cp.ndarray):
            return data
        return cp.asarray(data)

    def compute_distance(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
    ) -> float:
        """
        Compute energy distance between two arrays directly.

        Parameters
        ----------
        X
            First array of shape (n_samples_x, n_features)
        Y
            Second array of shape (n_samples_y, n_features)

        Returns
        -------
        float
            Energy distance between X and Y
        """
        # Convert to cupy arrays, preserving dtype
        X_gpu = cp.asarray(X)
        Y_gpu = cp.asarray(Y)

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
        X
            First array of shape (n_samples_x, n_features)
        Y
            Second array of shape (n_samples_y, n_features)
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility

        Returns
        -------
        mean
            Bootstrap mean distance
        variance
            Bootstrap variance
        """
        # Convert to cupy arrays, preserving dtype
        X_gpu = cp.asarray(X)
        Y_gpu = cp.asarray(Y)

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
        # X: (n, d), Y: (m, d) -> dist[i,j] = ||X[i] - Y[j]||

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
        device_ids: list[int],
    ) -> cp.ndarray:
        """Compute between-group mean distances for all group pairs.

        Splits pairs across specified GPUs and aggregates results on GPU 0.

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cp.ndarray
            Matrix of mean pairwise distances (k x k)
        """
        n_devices = len(device_ids)
        _, n_features = embedding.shape

        # Get group sizes to filter out single-cell diagonal pairs
        group_sizes = cp.diff(cat_offsets).astype(cp.int64)

        # Build upper triangular indices, excluding diagonal for single-cell groups
        triu_indices = cp.triu_indices(k)
        pair_left = triu_indices[0].astype(cp.int32)
        pair_right = triu_indices[1].astype(cp.int32)

        # Filter out diagonal pairs where group has < 2 cells
        is_diagonal = pair_left == pair_right
        has_pairs = group_sizes[pair_left] >= 2
        keep_mask = ~is_diagonal | has_pairs

        pair_left = pair_left[keep_mask]
        pair_right = pair_right[keep_mask]
        num_pairs = len(pair_left)

        if num_pairs == 0:
            # No pairs to compute
            norm_matrix = self._compute_norm_matrix(group_sizes, embedding.dtype)
            return cp.zeros((k, k), dtype=embedding.dtype) / norm_matrix

        # Split pairs across devices with load balancing
        pair_chunks = _split_pairs(pair_left, pair_right, n_devices, group_sizes)

        # Phase 1: Create streams and start async data transfer to all devices
        streams = {}
        device_data = []

        for i, device_id in enumerate(device_ids):
            chunk_left, chunk_right = pair_chunks[i]
            if len(chunk_left) == 0:
                device_data.append(None)
                continue

            with cp.cuda.Device(device_id):
                # Create non-blocking stream for this device
                streams[device_id] = cp.cuda.Stream(non_blocking=True)

                with streams[device_id]:
                    # Replicate data to this device (async on stream)
                    if device_id == 0:
                        dev_emb = embedding
                        dev_off = cat_offsets
                        dev_idx = cell_indices
                    else:
                        dev_emb = cp.asarray(embedding)
                        dev_off = cp.asarray(cat_offsets)
                        dev_idx = cp.asarray(cell_indices)

                    # Copy pair indices to this device
                    dev_pair_left = cp.asarray(chunk_left)
                    dev_pair_right = cp.asarray(chunk_right)

                    # Initialize local accumulator
                    dev_sums = cp.zeros((k, k), dtype=embedding.dtype)

                    device_data.append(
                        {
                            "emb": dev_emb,
                            "off": dev_off,
                            "idx": dev_idx,
                            "pair_left": dev_pair_left,
                            "pair_right": dev_pair_right,
                            "sums": dev_sums,
                            "n_pairs": len(dev_pair_left),
                            "device_id": device_id,
                        }
                    )

        # Phase 2: Synchronize data transfers, then launch kernels
        for data in device_data:
            if data is None:
                continue

            device_id = data["device_id"]
            with cp.cuda.Device(device_id):
                # Wait for data transfer to complete on this device
                streams[device_id].synchronize()

                # Launch kernel (on default stream, async)
                kernel, shared_mem, block_size = get_compute_group_distances_kernel(
                    embedding.dtype, n_features
                )
                blocks_per_pair = _calculate_blocks_per_pair(data["n_pairs"])
                grid = (data["n_pairs"], blocks_per_pair)
                block = (block_size,)

                kernel(
                    grid,
                    block,
                    (
                        data["emb"],
                        data["off"],
                        data["idx"],
                        data["pair_left"],
                        data["pair_right"],
                        data["sums"],
                        k,
                        n_features,
                        blocks_per_pair,
                    ),
                    shared_mem=shared_mem,
                )

        # Phase 3: Synchronize all devices (wait for kernels to complete)
        for data in device_data:
            if data is not None:
                with cp.cuda.Device(data["device_id"]):
                    cp.cuda.Stream.null.synchronize()

        # Phase 4: Aggregate on GPU 0
        with cp.cuda.Device(0):
            pairwise_sums = cp.zeros((k, k), dtype=embedding.dtype)
            for data in device_data:
                if data is not None:
                    # cp.asarray handles cross-device copy
                    pairwise_sums += cp.asarray(data["sums"])

            # Normalize sums to means
            norm_matrix = self._compute_norm_matrix(group_sizes, embedding.dtype)
            return pairwise_sums / norm_matrix

    def _compute_norm_matrix(
        self, group_sizes: cp.ndarray, dtype: np.dtype
    ) -> cp.ndarray:
        """Compute normalization matrix for pairwise means.

        Parameters
        ----------
        group_sizes
            Array of group sizes
        dtype
            Data type for output matrix

        Returns
        -------
        cp.ndarray
            Normalization matrix (k x k)
        """
        diag_counts = group_sizes * (group_sizes - 1) // 2
        # Handle single-cell groups: replace 0 with 1 to avoid division by zero
        diag_counts = cp.maximum(diag_counts, 1)
        cross_counts = cp.outer(group_sizes, group_sizes)
        norm_matrix = cross_counts.astype(dtype)
        cp.fill_diagonal(norm_matrix, diag_counts.astype(dtype))
        return norm_matrix

    def _onesided_means(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        *,
        selected_idx: int,
        device_ids: list[int],
    ) -> cp.ndarray:
        """Compute mean distances from selected group to all groups.

        Splits pairs across specified GPUs and aggregates results on GPU 0.

        Computes:
        - d[selected_idx, i] for all i (cross-distances)
        - d[i, i] for all i (self-distances needed for energy distance formula)

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        selected_idx
            Index of the selected group
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cp.ndarray
            Matrix of mean onesided distances (k x k)
        """
        n_devices = len(device_ids)
        _, n_features = embedding.shape

        # Get group sizes
        group_sizes = cp.diff(cat_offsets).astype(cp.int64)

        # Build pairs for onesided computation
        all_indices = cp.arange(k, dtype=cp.int32)
        cross_left = cp.full(k, selected_idx, dtype=cp.int32)
        cross_right = all_indices

        # Diagonal pairs for other groups with >= 2 cells
        mask = (all_indices != selected_idx) & (group_sizes >= 2)
        other_diag = all_indices[mask]

        # Combine all pairs
        pair_left = cp.concatenate([cross_left, other_diag])
        pair_right = cp.concatenate([cross_right, other_diag])
        num_pairs = len(pair_left)

        if num_pairs == 0:
            norm_matrix = self._compute_norm_matrix(group_sizes, embedding.dtype)
            return cp.zeros((k, k), dtype=embedding.dtype) / norm_matrix

        # Split pairs across devices with load balancing
        pair_chunks = _split_pairs(pair_left, pair_right, n_devices, group_sizes)

        # Phase 1: Create streams and start async data transfer to all devices
        streams = {}
        device_data = []

        for i, device_id in enumerate(device_ids):
            chunk_left, chunk_right = pair_chunks[i]
            if len(chunk_left) == 0:
                device_data.append(None)
                continue

            with cp.cuda.Device(device_id):
                # Create non-blocking stream for this device
                streams[device_id] = cp.cuda.Stream(non_blocking=True)

                with streams[device_id]:
                    # Replicate data to this device (async on stream)
                    if device_id == device_ids[0]:
                        dev_emb = embedding
                        dev_off = cat_offsets
                        dev_idx = cell_indices
                    else:
                        dev_emb = cp.asarray(embedding)
                        dev_off = cp.asarray(cat_offsets)
                        dev_idx = cp.asarray(cell_indices)

                    dev_pair_left = cp.asarray(chunk_left)
                    dev_pair_right = cp.asarray(chunk_right)

                    dev_sums = cp.zeros((k, k), dtype=embedding.dtype)

                    device_data.append(
                        {
                            "emb": dev_emb,
                            "off": dev_off,
                            "idx": dev_idx,
                            "pair_left": dev_pair_left,
                            "pair_right": dev_pair_right,
                            "sums": dev_sums,
                            "n_pairs": len(dev_pair_left),
                            "device_id": device_id,
                        }
                    )

        # Phase 2: Synchronize data transfers, then launch kernels
        for data in device_data:
            if data is None:
                continue

            device_id = data["device_id"]
            with cp.cuda.Device(device_id):
                # Wait for data transfer to complete on this device
                streams[device_id].synchronize()

                # Launch kernel (on default stream, async)
                kernel, shared_mem, block_size = get_compute_group_distances_kernel(
                    embedding.dtype, n_features
                )
                blocks_per_pair = _calculate_blocks_per_pair(data["n_pairs"])
                grid = (data["n_pairs"], blocks_per_pair)
                block = (block_size,)

                kernel(
                    grid,
                    block,
                    (
                        data["emb"],
                        data["off"],
                        data["idx"],
                        data["pair_left"],
                        data["pair_right"],
                        data["sums"],
                        k,
                        n_features,
                        blocks_per_pair,
                    ),
                    shared_mem=shared_mem,
                )

        # Phase 3: Synchronize all devices (wait for kernels to complete)
        for data in device_data:
            if data is not None:
                with cp.cuda.Device(data["device_id"]):
                    cp.cuda.Stream.null.synchronize()

        # Phase 4: Aggregate on GPU 0
        with cp.cuda.Device(device_ids[0]):
            onesided_sums = cp.zeros((k, k), dtype=embedding.dtype)
            for data in device_data:
                if data is not None:
                    onesided_sums += cp.asarray(data["sums"])

            norm_matrix = self._compute_norm_matrix(group_sizes, embedding.dtype)
            return onesided_sums / norm_matrix

    def _pairwise_means_bootstrap(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        n_bootstrap: int,
        random_state: int,
        device_ids: list[int],
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute bootstrap statistics for pairwise distances.

        Distributes bootstrap iterations across GPUs - each GPU gets a contiguous
        batch of iterations to run independently.

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        tuple
            (means, variances) matrices (k x k each)
        """
        n_devices = len(device_ids)

        # Generate all bootstrap indices on CPU for reproducibility
        bootstrap_indices = self._generate_bootstrap_indices(
            cat_offsets, k, n_bootstrap, random_state
        )

        # Split iterations across devices - each GPU gets a batch
        iters_per_device = (n_bootstrap + n_devices - 1) // n_devices

        # Phase 1: Transfer data to all devices
        streams: dict[int, cp.cuda.Stream] = {}
        device_data: list[dict | None] = []

        for i, device_id in enumerate(device_ids):
            start_iter = i * iters_per_device
            end_iter = min(start_iter + iters_per_device, n_bootstrap)
            n_iters = end_iter - start_iter

            if n_iters == 0:
                device_data.append(None)
                continue

            with cp.cuda.Device(device_id):
                streams[device_id] = cp.cuda.Stream(non_blocking=True)

                with streams[device_id]:
                    if device_id == device_ids[0]:
                        dev_emb = embedding
                        dev_off = cat_offsets
                        dev_idx = cell_indices
                    else:
                        dev_emb = cp.asarray(embedding)
                        dev_off = cp.asarray(cat_offsets)
                        dev_idx = cp.asarray(cell_indices)

                    device_data.append(
                        {
                            "embedding": dev_emb,
                            "cat_offsets": dev_off,
                            "cell_indices": dev_idx,
                            "start_iter": start_iter,
                            "n_iters": n_iters,
                            "device_id": device_id,
                        }
                    )

        # Phase 2: Run iterations on each device (batched per GPU)
        for data in device_data:
            if data is None:
                continue

            device_id = data["device_id"]
            start_iter = data["start_iter"]
            n_iters = data["n_iters"]

            with cp.cuda.Device(device_id):
                with streams[device_id]:
                    results = []
                    for j in range(n_iters):
                        iter_idx = start_iter + j

                        # Create bootstrap sample
                        boot_cat_offsets, boot_cell_indices = (
                            self._bootstrap_sample_cells_from_indices(
                                cat_offsets=data["cat_offsets"],
                                cell_indices=data["cell_indices"],
                                k=k,
                                bootstrap_group_indices=bootstrap_indices[iter_idx],
                            )
                        )

                        # Compute pairwise means
                        pairwise_means = self._pairwise_means(
                            embedding=data["embedding"],
                            cat_offsets=boot_cat_offsets,
                            cell_indices=boot_cell_indices,
                            k=k,
                            device_ids=[device_id],
                        )
                        results.append(pairwise_means)

                    data["results"] = results

        # Phase 3: Synchronize all devices
        for device_id in streams:
            with cp.cuda.Device(device_id):
                streams[device_id].synchronize()

        # Phase 4: Gather results from all devices
        all_results = []
        for data in device_data:
            if data is None:
                continue
            device_id = data["device_id"]
            with cp.cuda.Device(device_id):
                for result in data["results"]:
                    all_results.append(result.get())

        # Compute statistics on first GPU
        with cp.cuda.Device(device_ids[0]):
            bootstrap_stack = cp.array(all_results)  # [n_bootstrap, k, k]
            means = cp.mean(bootstrap_stack, axis=0)
            variances = cp.var(bootstrap_stack, axis=0)

        return means, variances

    def _onesided_means_bootstrap(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        selected_idx: int,
        n_bootstrap: int,
        random_state: int,
        device_ids: list[int],
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute bootstrap statistics for onesided distances.

        Distributes bootstrap iterations across GPUs - each GPU gets a contiguous
        batch of iterations to run independently.

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        selected_idx
            Index of the selected group
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        tuple
            (means, variances) matrices (k x k each)
        """
        n_devices = len(device_ids)

        # Generate all bootstrap indices on CPU for reproducibility
        bootstrap_indices = self._generate_bootstrap_indices(
            cat_offsets, k, n_bootstrap, random_state
        )

        # Split iterations across devices - each GPU gets a batch
        iters_per_device = (n_bootstrap + n_devices - 1) // n_devices

        # Phase 1: Transfer data to all devices
        streams: dict[int, cp.cuda.Stream] = {}
        device_data: list[dict | None] = []

        for i, device_id in enumerate(device_ids):
            start_iter = i * iters_per_device
            end_iter = min(start_iter + iters_per_device, n_bootstrap)
            n_iters = end_iter - start_iter

            if n_iters == 0:
                device_data.append(None)
                continue

            with cp.cuda.Device(device_id):
                streams[device_id] = cp.cuda.Stream(non_blocking=True)

                with streams[device_id]:
                    if device_id == device_ids[0]:
                        dev_emb = embedding
                        dev_off = cat_offsets
                        dev_idx = cell_indices
                    else:
                        dev_emb = cp.asarray(embedding)
                        dev_off = cp.asarray(cat_offsets)
                        dev_idx = cp.asarray(cell_indices)

                    device_data.append(
                        {
                            "embedding": dev_emb,
                            "cat_offsets": dev_off,
                            "cell_indices": dev_idx,
                            "start_iter": start_iter,
                            "n_iters": n_iters,
                            "device_id": device_id,
                        }
                    )

        # Phase 2: Run iterations on each device (batched per GPU)
        for data in device_data:
            if data is None:
                continue

            device_id = data["device_id"]
            start_iter = data["start_iter"]
            n_iters = data["n_iters"]

            with cp.cuda.Device(device_id):
                with streams[device_id]:
                    results = []
                    for j in range(n_iters):
                        iter_idx = start_iter + j

                        # Create bootstrap sample
                        boot_cat_offsets, boot_cell_indices = (
                            self._bootstrap_sample_cells_from_indices(
                                cat_offsets=data["cat_offsets"],
                                cell_indices=data["cell_indices"],
                                k=k,
                                bootstrap_group_indices=bootstrap_indices[iter_idx],
                            )
                        )

                        # Compute onesided means
                        onesided_means = self._onesided_means(
                            embedding=data["embedding"],
                            cat_offsets=boot_cat_offsets,
                            cell_indices=boot_cell_indices,
                            k=k,
                            selected_idx=selected_idx,
                            device_ids=[device_id],
                        )
                        results.append(onesided_means)

                    data["results"] = results

        # Phase 3: Synchronize all devices
        for device_id in streams:
            with cp.cuda.Device(device_id):
                streams[device_id].synchronize()

        # Phase 4: Gather results from all devices
        all_results = []
        for data in device_data:
            if data is None:
                continue
            device_id = data["device_id"]
            with cp.cuda.Device(device_id):
                for result in data["results"]:
                    all_results.append(result.get())

        # Compute statistics on first GPU
        with cp.cuda.Device(device_ids[0]):
            bootstrap_stack = cp.array(all_results)
            means = cp.mean(bootstrap_stack, axis=0)
            variances = cp.var(bootstrap_stack, axis=0)

        return means, variances

    def _generate_bootstrap_indices(
        self,
        cat_offsets: cp.ndarray,
        k: int,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> list[list[cp.ndarray]]:
        """Generate bootstrap indices for all groups and all bootstrap iterations."""
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
        multi_gpu: bool | list[int] | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare bootstrap edistance DataFrames."""
        device_ids = parse_device_ids(multi_gpu=multi_gpu)
        pairwise_means_boot, pairwise_vars_boot = self._pairwise_means_bootstrap(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            device_ids=device_ids,
        )

        # Vectorized edistance: e[a,b] = 2*d[a,b] - d[a,a] - d[b,b]
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
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame:
        """Prepare edistance DataFrame."""
        device_ids = parse_device_ids(multi_gpu=multi_gpu)
        pairwise_means = self._pairwise_means(
            embedding, cat_offsets, cell_indices, k, device_ids
        )

        # Vectorized edistance: e[a,b] = 2*d[a,b] - d[a,a] - d[b,b]
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
