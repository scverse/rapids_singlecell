from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import scipy.sparse as sp

from rapids_singlecell.preprocessing._utils import _sparse_to_dense

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

EPS = 1e-9
WARP_SIZE = 32
MAX_THREADS_PER_BLOCK = 512


def _round_up_to_warp(n: int) -> int:
    """Round up to nearest multiple of WARP_SIZE, capped at MAX_THREADS_PER_BLOCK."""
    return min(MAX_THREADS_PER_BLOCK, ((n + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE)


def _select_top_n(scores: NDArray, n_top: int) -> NDArray:
    """Select indices of top n scores.

    Uses argpartition + argsort for O(n + k log k) complexity where k = n_top.
    This is faster than full sorting when k << n.
    """
    n_from = scores.shape[0]
    reference_indices = np.arange(n_from, dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices


def _select_groups(
    labels: pd.Series, groups_order_subset: Literal["all"] | list[str] = "all"
) -> tuple[NDArray, NDArray[np.bool_]]:
    """Select groups and create masks for each group."""
    groups_order = labels.cat.categories
    groups_masks = np.zeros(
        (len(labels.cat.categories), len(labels.cat.codes)), dtype=bool
    )
    for iname, name in enumerate(labels.cat.categories):
        if labels.cat.categories[iname] in labels.cat.codes:
            mask = labels.cat.categories[iname] == labels.cat.codes
        else:
            mask = iname == labels.cat.codes
        groups_masks[iname] = mask.values
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != "all":
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(np.where(name == labels.cat.categories)[0])
        if len(groups_ids) == 0:
            groups_ids = np.where(
                np.isin(
                    np.arange(len(labels.cat.categories)).astype(str),
                    np.array(groups_order_subset),
                )
            )[0]
        groups_ids = [groups_id.item() for groups_id in groups_ids]
        if len(groups_ids) > 2:
            groups_ids = np.sort(groups_ids)
        groups_masks = groups_masks[groups_ids]
        groups_order_subset = labels.cat.categories[groups_ids].to_numpy()
    else:
        groups_order_subset = groups_order.to_numpy()
    return groups_order_subset, groups_masks


def _choose_chunk_size(requested: int | None) -> int:
    """Choose chunk size for gene processing."""
    if requested is not None:
        return int(requested)
    return 128


def _csc_columns_to_gpu(X_csc, start: int, stop: int, n_rows: int) -> cp.ndarray:
    """
    Extract columns from a CSC matrix via direct indptr pointer slicing.

    Works for both scipy and CuPy CSC matrices. Much faster than
    ``X[:, start:stop]`` which rebuilds index arrays internally.
    """
    s_ptr = int(X_csc.indptr[start])
    e_ptr = int(X_csc.indptr[stop])
    chunk_data = cp.asarray(X_csc.data[s_ptr:e_ptr])
    chunk_indices = cp.asarray(X_csc.indices[s_ptr:e_ptr])
    chunk_indptr = cp.asarray(X_csc.indptr[start : stop + 1] - s_ptr)
    csc_chunk = cpsp.csc_matrix(
        (chunk_data, chunk_indices, chunk_indptr), shape=(n_rows, stop - start)
    )
    return _sparse_to_dense(csc_chunk, order="F").astype(cp.float64)


def _get_column_block(X, start: int, stop: int) -> cp.ndarray:
    """Extract a column block as a dense F-order float64 CuPy array."""
    match X:
        case sp.csc_matrix() | sp.csc_array():
            return _csc_columns_to_gpu(X, start, stop, X.shape[0])
        case sp.spmatrix() | sp.sparray():
            chunk = cpsp.csc_matrix(X[:, start:stop].tocsc())
            return _sparse_to_dense(chunk, order="F").astype(cp.float64)
        case cpsp.csc_matrix():
            return _csc_columns_to_gpu(X, start, stop, X.shape[0])
        case cpsp.spmatrix():
            return _sparse_to_dense(X[:, start:stop], order="F").astype(cp.float64)
        case np.ndarray() | cp.ndarray():
            return cp.asarray(X[:, start:stop], dtype=cp.float64, order="F")
        case _:
            raise ValueError(f"Unsupported matrix type: {type(X)}")
