from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData


def parse_device_ids(*, multi_gpu: bool | list[int] | str | None) -> list[int]:
    """Parse multi_gpu parameter into a list of device IDs.

    Parameters
    ----------
    multi_gpu
        GPU selection:
        - None or True: Use all available GPUs
        - False: Use only GPU 0
        - list[int]: Use specific GPU IDs (e.g., [0, 2])
        - str: Comma-separated GPU IDs (e.g., "0,2")

    Returns
    -------
    list[int]
        List of device IDs to use

    Raises
    ------
    ValueError
        If any specified device ID is invalid or out of range
    """
    n_available = cp.cuda.runtime.getDeviceCount()

    if multi_gpu is None or multi_gpu is True:
        return list(range(n_available))
    elif multi_gpu is False:
        return [0]
    elif isinstance(multi_gpu, str):
        device_ids = [int(x.strip()) for x in multi_gpu.split(",")]
    elif isinstance(multi_gpu, list):
        device_ids = multi_gpu
    else:
        raise ValueError(
            f"multi_gpu must be bool, list[int], or str, got {type(multi_gpu)}"
        )

    # Validate device IDs
    invalid_ids = [d for d in device_ids if d < 0 or d >= n_available]
    if invalid_ids:
        raise ValueError(
            f"Invalid GPU device ID(s): {invalid_ids}. "
            f"Available devices: {list(range(n_available))}"
        )

    if len(device_ids) == 0:
        raise ValueError("multi_gpu must specify at least one device")

    return device_ids


class BaseMetric(ABC):
    """
    Abstract base class for distance metrics.

    All distance metric implementations should inherit from this class
    and implement the required methods.

    Parameters
    ----------
    obsm_key
        Key in adata.obsm for embeddings (default: 'X_pca')

    Attributes
    ----------
    supports_multi_gpu
        Whether this metric supports multi-GPU computation.
        Subclasses should override this to True if they implement multi-GPU.
    """

    supports_multi_gpu: bool = False

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
        multi_gpu: bool | list[int] | str | None = None,
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
        multi_gpu: bool | list[int] | str | None = None,
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement bootstrap"
        )
