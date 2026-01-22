from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np
    import pandas as pd
    from anndata import AnnData


class MeanVar(NamedTuple):
    """Result of bootstrap computation containing mean and variance."""

    mean: float
    variance: float


class Distance:
    """
    GPU-accelerated distance computation between groups of cells.

    This class provides an extensible framework for computing various distance metrics
    between cell groups in single-cell data, with GPU acceleration via CuPy.

    The API is designed to be compatible with pertpy's Distance class.

    Parameters
    ----------
    metric : str
        Distance metric to use. Currently supported:
        - 'edistance': Energy distance (GPU-accelerated)
    layer_key : str | None
        Name of the counts layer containing raw counts to calculate distances for.
        Mutually exclusive with 'obsm_key'. If None, the parameter is ignored.
    obsm_key : str | None
        Key in adata.obsm for embeddings. Mutually exclusive with 'layer_key'.
        Defaults to 'X_pca' if neither layer_key nor obsm_key is specified.
    kernel : str
        Kernel strategy: 'auto' or 'manual'.
        - 'auto': Dynamically choose optimal blocks_per_pair (default)
        - 'manual': Use the specified blocks_per_pair directly
    blocks_per_pair : int
        Number of blocks per pair (default: 32). For 'auto', this is the maximum.
        Higher values increase parallelism but add atomic overhead.

    Examples
    --------
    >>> import rapids_singlecell as rsc
    >>> distance = rsc.ptg.Distance(metric='edistance')
    >>> result = distance.pairwise(adata, groupby='perturbation')
    >>> print(result.distances)

    # Direct computation on arrays (pertpy-compatible API)
    >>> X = adata.obsm["X_pca"][adata.obs["group"] == "A"]
    >>> Y = adata.obsm["X_pca"][adata.obs["group"] == "B"]
    >>> d = distance(X, Y)  # Returns energy distance as float
    """

    def __init__(
        self,
        metric: Literal["edistance"] = "edistance",
        layer_key: str | None = None,
        obsm_key: str | None = None,
        kernel: Literal["auto", "manual"] = "auto",
        blocks_per_pair: int = 32,
    ):
        """Initialize Distance calculator with specified metric."""
        if layer_key and obsm_key:
            raise ValueError(
                "Cannot use 'layer_key' and 'obsm_key' at the same time.\n"
                "Please provide only one of the two keys."
            )
        if not layer_key and not obsm_key:
            obsm_key = "X_pca"

        self.metric = metric
        self.layer_key = layer_key
        self.obsm_key = obsm_key
        self.kernel = kernel
        self.blocks_per_pair = blocks_per_pair
        self._metric_impl = None
        self._initialize_metric()

    def _initialize_metric(self):
        """Initialize the metric implementation based on the metric type."""
        if self.metric == "edistance":
            from rapids_singlecell.pertpy_gpu._metrics._edistance_metric import (
                EDistanceMetric,
            )

            self._metric_impl = EDistanceMetric(
                layer_key=self.layer_key,
                obsm_key=self.obsm_key,
                kernel=self.kernel,
                blocks_per_pair=self.blocks_per_pair,
            )
        else:
            raise ValueError(
                f"Unknown metric: {self.metric}. Supported metrics: ['edistance']"
            )

    def __call__(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
    ) -> float:
        """
        Compute distance between two cell groups directly from arrays.

        This provides pertpy-compatible API for direct distance computation.

        Parameters
        ----------
        X : np.ndarray | cp.ndarray
            First array of shape (n_samples_x, n_features)
        Y : np.ndarray | cp.ndarray
            Second array of shape (n_samples_y, n_features)

        Returns
        -------
        distance : float
            Distance between X and Y

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> X = adata.obsm["X_pca"][adata.obs["group"] == "A"]
        >>> Y = adata.obsm["X_pca"][adata.obs["group"] == "B"]
        >>> d = distance(X, Y)
        """
        if not hasattr(self._metric_impl, "compute_distance"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support direct distance computation"
            )
        return self._metric_impl.compute_distance(X, Y)

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
    ):
        """
        Compute pairwise distances between all cell groups.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        groupby : str
            Key in adata.obs for grouping cells
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
        result
            Result object containing distances and optional variance DataFrames.
            The exact type depends on the metric used.

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> result = distance.pairwise(adata, groupby='condition')
        >>> print(result.distances)
        """
        return self._metric_impl.pairwise(
            adata=adata,
            groupby=groupby,
            groups=groups,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            inplace=inplace,
        )

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
        Compute distances from one selected group to all other groups.

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
            Whether to compute bootstrap variance estimates
        n_bootstrap : int
            Number of bootstrap iterations (if bootstrap=True)
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        distances : pd.Series
            Series containing distances from selected_group to all other groups

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> distances = distance.onesided_distances(
        ...     adata, groupby='condition', selected_group='control'
        ... )
        >>> print(distances)
        """
        if not hasattr(self._metric_impl, "onesided_distances"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support onesided_distances"
            )
        return self._metric_impl.onesided_distances(
            adata=adata,
            groupby=groupby,
            selected_group=selected_group,
            groups=groups,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

    def bootstrap(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> MeanVar:
        """
        Compute bootstrap mean and variance for distance between two arrays.

        This provides pertpy-compatible API for bootstrap computation directly
        on arrays without requiring an AnnData object.

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
        result : MeanVar
            Named tuple containing mean and variance of bootstrapped distances

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> X = adata.obsm["X_pca"][adata.obs["group"] == "A"]
        >>> Y = adata.obsm["X_pca"][adata.obs["group"] == "B"]
        >>> result = distance.bootstrap(X, Y, n_bootstrap=100)
        >>> print(f"Distance: {result.mean:.3f} ± {result.variance**0.5:.3f}")
        """
        if not hasattr(self._metric_impl, "bootstrap_arrays"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support bootstrap"
            )
        mean, variance = self._metric_impl.bootstrap_arrays(
            X=X,
            Y=Y,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        return MeanVar(mean=mean, variance=variance)

    def bootstrap_adata(
        self,
        adata: AnnData,
        groupby: str,
        group_a: str,
        group_b: str,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> MeanVar:
        """
        Compute bootstrap mean and variance for distance between two groups in AnnData.

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
        result : MeanVar
            Named tuple containing mean and variance of bootstrapped distances

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> result = distance.bootstrap_adata(
        ...     adata, groupby='condition', group_a='treated', group_b='control'
        ... )
        >>> print(f"Distance: {result.mean:.3f} ± {result.variance**0.5:.3f}")
        """
        if not hasattr(self._metric_impl, "bootstrap"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support bootstrap"
            )
        mean, variance = self._metric_impl.bootstrap(
            adata=adata,
            groupby=groupby,
            group_a=group_a,
            group_b=group_b,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        return MeanVar(mean=mean, variance=variance)

    def __repr__(self) -> str:
        """String representation of Distance object."""
        if self.layer_key:
            return f"Distance(metric='{self.metric}', layer_key='{self.layer_key}')"
        return f"Distance(metric='{self.metric}', obsm_key='{self.obsm_key}')"
