from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cuml.metrics import pairwise_distances

try:
    from rapids_singlecell._cuda import _cooc_cuda as _co
except ImportError:
    _co = None

from rapids_singlecell.preprocessing._harmony._helper import (
    _create_category_index_mapping,
)

from ._utils import _assert_categorical_obs, _assert_spatial_basis

if TYPE_CHECKING:
    from anndata import AnnData


def co_occurrence(
    adata: AnnData,
    cluster_key: str,
    *,
    spatial_key: str = "spatial",
    interval: int | np.ndarray | cp.ndarray = 50,
    copy: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
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
    out = _co_occurrence_helper(spatial, interval, labs, fast=True)
    out, interval = out.get(), interval.get()
    if copy:
        return out, interval

    adata.uns[f"{cluster_key}_co_occurrence"] = {"occ": out, "interval": interval}


def _find_min_max(spatial: cp.ndarray) -> tuple[float, float]:
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
    spatial: cp.ndarray, v_radium: cp.ndarray, labs: cp.ndarray, *, fast: bool = True
) -> cp.ndarray:
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
    # labels are dense [0, k)
    k = int(cp.asnumpy(labs.max())) + 1
    l_val = len(v_radium) - 1
    thresholds = (v_radium[1:]) ** 2
    use_fast_kernel = False  # Flag to track which kernel path was taken
    if fast:
        # New CSR-based per-category-pair kernel using per-warp shared histograms (size l_val)
        # 1 block per (cat_a, cat_b) with a<=b
        # Prepare category CSR structures
        cat_offsets, cell_indices = _create_category_index_mapping(labs, k)
        # Build pair list (upper triangle including diagonal)
        pair_left = []
        pair_right = []
        for a in range(k):
            for b in range(a, k):
                pair_left.append(a)
                pair_right.append(b)
        pair_left = cp.asarray(pair_left, dtype=cp.int32)
        pair_right = cp.asarray(pair_right, dtype=cp.int32)
        # Let C++ pick tpb; fall back to slow if insufficient shared memory
        counts = cp.zeros((k, k, l_val), dtype=cp.int32)
        reader = 1
        use_fast_kernel = _co.count_csr_catpairs_auto(
            spatial.data.ptr,
            thresholds=thresholds.data.ptr,
            cat_offsets=cat_offsets.data.ptr,
            cell_indices=cell_indices.data.ptr,
            pair_left=pair_left.data.ptr,
            pair_right=pair_right.data.ptr,
            counts_delta=counts.data.ptr,
            num_pairs=pair_left.size,
            k=k,
            l_val=l_val,
            stream=cp.cuda.get_current_stream().ptr,
        )

    # Fallback to the standard kernel if fast=False or shared memory was insufficient
    if not use_fast_kernel:
        counts = cp.zeros((k, k, l_val * 2), dtype=cp.int32)
        _co.count_pairwise(
            spatial.data.ptr,
            thresholds=thresholds.data.ptr,
            labels=labs.data.ptr,
            result=counts.data.ptr,
            n=spatial.shape[0],
            k=k,
            l_val=l_val,
            stream=cp.cuda.get_current_stream().ptr,
        )
        reader = 0

    occ_prob = cp.empty((k, k, l_val), dtype=np.float32)
    ok = False
    if fast:
        ok = _co.reduce_shared(
            counts.data.ptr,
            out=occ_prob.data.ptr,
            k=k,
            l_val=l_val,
            format=reader,
            stream=cp.cuda.get_current_stream().ptr,
        )
    if not ok:
        inter_out = cp.zeros((l_val, k, k), dtype=np.float32)
        _co.reduce_global(
            counts.data.ptr,
            inter_out=inter_out.data.ptr,
            out=occ_prob.data.ptr,
            k=k,
            l_val=l_val,
            format=reader,
            stream=cp.cuda.get_current_stream().ptr,
        )

    return occ_prob
