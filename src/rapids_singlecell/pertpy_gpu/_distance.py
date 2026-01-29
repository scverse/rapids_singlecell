from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    API compatible with pertpy's Distance class.

    Currently supported metrics:

    - ``"edistance"``: Energy distance (default).
        Twice the mean pairwise distance between cells of two groups minus
        the mean pairwise distance between cells within each group. See
        `Peidli et al. (2023) <https://doi.org/10.1101/2022.08.20.504663>`__.

    Parameters
    ----------
    metric
        Distance metric to use. Currently only ``"edistance"`` is supported.
    layer_key
        Key in adata.layers for cell data. Mutually exclusive with ``obsm_key``.
    obsm_key
        Key in adata.obsm for embeddings. Mutually exclusive with ``layer_key``.
        Defaults to ``"X_pca"`` if neither is specified.

    Examples
    --------
    >>> import rapids_singlecell as rsc
    >>> distance = rsc.ptg.Distance(metric='edistance')
    >>> result = distance.pairwise(adata, groupby='perturbation')

    >>> # Direct computation on arrays
    >>> d = distance(X, Y)
    """

    def __init__(
        self,
        metric: Literal["edistance"] = "edistance",
        layer_key: str | None = None,
        obsm_key: str | None = None,
    ):
        """Initialize Distance calculator with specified metric."""
        if layer_key is not None and obsm_key is not None:
            raise ValueError(
                "Cannot use 'layer_key' and 'obsm_key' at the same time.\n"
                "Please provide only one of the two keys."
            )
        if layer_key is None and obsm_key is None:
            obsm_key = "X_pca"

        self.metric = metric
        self.layer_key = layer_key
        self.obsm_key = obsm_key
        self._metric_impl = None
        self._initialize_metric()

    def _initialize_metric(self):
        """Initialize the metric implementation based on the metric type."""
        if self.metric == "edistance":
            from rapids_singlecell.pertpy_gpu._metrics._edistance import (
                EDistanceMetric,
            )

            self._metric_impl = EDistanceMetric(
                layer_key=self.layer_key,
                obsm_key=self.obsm_key,
            )
        else:
            raise ValueError(
                f"Unknown metric: {self.metric}. Supported metrics: ['edistance']"
            )

    def _check_multi_gpu_support(
        self, *, multi_gpu: bool | list[int] | str | None
    ) -> bool | list[int] | str:
        """Check if metric supports multi-GPU and resolve None default.

        Parameters
        ----------
        multi_gpu
            The multi_gpu parameter passed by the user. None means use default
            (True if supported, False otherwise).

        Returns
        -------
        multi_gpu
            Returns False if metric doesn't support multi-GPU, otherwise
            returns the resolved value (True if None was passed and supported).
        """
        # If None, default to True if supported, False otherwise
        if multi_gpu is None:
            return self._metric_impl.supports_multi_gpu

        if not self._metric_impl.supports_multi_gpu:
            # Check if user explicitly requested multi-GPU
            uses_multi_gpu = (
                multi_gpu is True
                or (isinstance(multi_gpu, list) and len(multi_gpu) > 1)
                or (isinstance(multi_gpu, str) and "," in multi_gpu)
            )
            if uses_multi_gpu:
                warnings.warn(
                    f"Metric '{self.metric}' does not support multi-GPU. "
                    "Falling back to single GPU (device 0).",
                    UserWarning,
                    stacklevel=3,
                )
            return False
        return multi_gpu

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
        X
            First array of shape (n_samples_x, n_features)
        Y
            Second array of shape (n_samples_y, n_features)

        Returns
        -------
        float
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
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ):
        """
        Compute pairwise distances between all cell groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
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

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> result = distance.pairwise(adata, groupby='condition')
        """
        multi_gpu = self._check_multi_gpu_support(multi_gpu=multi_gpu)
        return self._metric_impl.pairwise(
            adata=adata,
            groupby=groupby,
            groups=groups,
            bootstrap=bootstrap,
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
        Compute distances from one selected group to all other groups.

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

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> distances = distance.onesided_distances(
        ...     adata, groupby='condition', selected_group='control'
        ... )
        """
        if not hasattr(self._metric_impl, "onesided_distances"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support onesided_distances"
            )
        multi_gpu = self._check_multi_gpu_support(multi_gpu=multi_gpu)
        return self._metric_impl.onesided_distances(
            adata=adata,
            groupby=groupby,
            selected_group=selected_group,
            groups=groups,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
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
        result
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

    def __repr__(self) -> str:
        """String representation of Distance object."""
        if self.layer_key:
            return f"Distance(metric='{self.metric}', layer_key='{self.layer_key}')"
        return f"Distance(metric='{self.metric}', obsm_key='{self.obsm_key}')"
