from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData


class BaseMetric(ABC):
    """
    Abstract base class for distance metrics.

    All distance metric implementations should inherit from this class
    and implement the required methods.

    Parameters
    ----------
    obsm_key
        Key in adata.obsm for embeddings (default: 'X_pca')
    """

    def __init__(self, obsm_key: str = "X_pca"):
        """Initialize base metric with obsm_key."""
        self.obsm_key = obsm_key

    @abstractmethod
    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
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

        Returns
        -------
        result
            Result object containing distances and optional variance information.
        """
        ...

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
    ):
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

        Returns
        -------
        distances
            Distance values from selected_group to other groups
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement onesided_distances"
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
    ):
        """
        Compute bootstrap mean and variance for distance between two specific groups.

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

        Returns
        -------
        mean
            Bootstrap mean distance
        variance
            Bootstrap variance
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement bootstrap"
        )
