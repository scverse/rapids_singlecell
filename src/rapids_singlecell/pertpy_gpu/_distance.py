from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


class Distance:
    """
    GPU-accelerated distance computation between groups of cells.

    This class provides an extensible framework for computing various distance metrics
    between cell groups in single-cell data, with GPU acceleration via CuPy.

    Parameters
    ----------
    metric : str
        Distance metric to use. Currently supported:
        - 'edistance': Energy distance (GPU-accelerated)
    obsm_key : str
        Key in adata.obsm for embeddings (default: 'X_pca')

    Examples
    --------
    >>> import rapids_singlecell as rsc
    >>> distance = rsc.ptg.Distance(metric='edistance')
    >>> result = distance.pairwise(adata, groupby='perturbation')
    >>> print(result.distances)
    """

    def __init__(
        self,
        metric: Literal["edistance"] = "edistance",
        obsm_key: str = "X_pca",
    ):
        """Initialize Distance calculator with specified metric."""
        self.metric = metric
        self.obsm_key = obsm_key
        self._metric_impl = None
        self._initialize_metric()

    def _initialize_metric(self):
        """Initialize the metric implementation based on the metric type."""
        if self.metric == "edistance":
            from rapids_singlecell.pertpy_gpu._metrics._edistance_metric import (
                EDistanceMetric,
            )

            self._metric_impl = EDistanceMetric(obsm_key=self.obsm_key)
        else:
            raise ValueError(
                f"Unknown metric: {self.metric}. Supported metrics: ['edistance']"
            )

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
        adata: AnnData,
        groupby: str,
        group_a: str,
        group_b: str,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[float, float]:
        """
        Compute bootstrap mean and variance for distance between two specific groups.

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

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> mean, var = distance.bootstrap(
        ...     adata, groupby='condition', group_a='treated', group_b='control'
        ... )
        >>> print(f"Distance: {mean:.3f} ± {var**0.5:.3f}")
        """
        if not hasattr(self._metric_impl, "bootstrap"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support bootstrap"
            )
        return self._metric_impl.bootstrap(
            adata=adata,
            groupby=groupby,
            group_a=group_a,
            group_b=group_b,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

    def __repr__(self) -> str:
        """String representation of Distance object."""
        return f"Distance(metric='{self.metric}', obsm_key='{self.obsm_key}')"
