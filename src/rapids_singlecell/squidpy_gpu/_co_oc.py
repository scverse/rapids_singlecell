from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cuml.metrics import pairwise_distances

from ._utils import _assert_categorical_obs, _assert_spatial_basis
from .kernels._co_oc import occur_count_kernel, occur_count_kernel2

if TYPE_CHECKING:
    from anndata import AnnData
    from cupy import NDArray as NDArrayC
    from numpy import NDArray as NDArrayA


def co_occurrence(
    adata: AnnData,
    cluster_key: str,
    spatial_key: str = "spatial",
    interval: int | NDArrayA | NDArrayC = 50,
    copy: bool = False,
) -> tuple[NDArrayA, NDArrayA] | None:
    """
    Compute co-occurrence probability of clusters.

    Parameters
    ----------
    adata
        Annotated data object.
    cluster_key
        Key for the cluster labels.
    spatial_key
        Key for the spatial coordinates.
    interval
        Distances interval at which co-occurrence is computed. If :class:`int`, uniformly spaced interval
        of the given size will be used.
    copy
        If ``True``, return the co-occurrence probability and the distance thresholds intervals.

    Returns
    -------
    If ``copy = True``, returns the co-occurrence probability and the distance thresholds intervals.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['occ']`` - the co-occurrence probabilities
        across interval thresholds.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['interval']`` - the distance thresholds
        computed at ``interval``.
    """

    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)

    spatial = cp.array(adata.obsm[spatial_key]).astype(np.float32)
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    labs = cp.array([clust_map[c] for c in original_clust], dtype=np.int32)

    # create intervals thresholds
    if isinstance(interval, int):
        thresh_min, thresh_max = _find_min_max(spatial)
        interval = cp.linspace(thresh_min, thresh_max, num=interval, dtype=np.float32)
    else:
        interval = cp.array(sorted(interval), dtype=np.float32, copy=True)
    if len(interval) <= 1:
        raise ValueError(
            f"Expected interval to be of length `>= 2`, found `{len(interval)}`."
        )

    out = _co_occurrence_helper(spatial, interval, labs)
    out, interval = out.get(), interval.get()
    if copy:
        return out, interval

    adata.uns[f"{cluster_key}_co_occurrence"] = {"occ": out, "interval": interval}


def _find_min_max(spatial: NDArrayC) -> tuple[float, float]:
    coord_sum = cp.sum(spatial, axis=1)
    min_idx, min_idx2 = cp.argpartition(coord_sum, 2)[:2]
    max_idx = cp.argmax(coord_sum)
    thres_max = (
        pairwise_distances(
            spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1)
        )[0, 0]
        / 2.0
    )
    thres_min = pairwise_distances(
        spatial[min_idx, :].reshape(1, -1), spatial[min_idx2, :].reshape(1, -1)
    )[0, 0]
    return thres_min.astype(np.float32), thres_max.astype(np.float32)


def _co_occurrence_helper(
    spatial: NDArrayC, v_radium: NDArrayC, labs: NDArrayC
) -> NDArrayC:
    """
    Fast co-occurrence probability computation using cuda kernels.

    Parameters
    ----------
    spatial
        Spatial coordinates.
    v_radium
        Distance thresholds (in ascending order).
    labs
        Cluster labels (as integers).

    Returns
    -------
    occ_prob
        A 3D array of shape (k, k, len(v_radium)-1) containing the co-occurrence probabilities.

    """
    n = spatial.shape[0]
    labs_unique = cp.unique(labs)
    k = len(labs_unique)
    l_val = len(v_radium) - 1
    thresholds = (v_radium[1:]) ** 2
    counts = cp.zeros((k, k, l_val * 2), dtype=cp.int32)
    grid = (n,)
    block = (32,)
    occur_count_kernel(grid, block, (spatial, thresholds, labs, counts, n, k, l_val))
    occ_prob = cp.empty((k, k, l_val), dtype=np.float32)
    shared_mem_size = (k * k + k) * cp.dtype("float32").itemsize
    grid = (l_val,)
    occur_count_kernel2(
        grid, block, (counts, occ_prob, k, l_val), shared_mem=shared_mem_size
    )
    return occ_prob
