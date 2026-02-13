from __future__ import annotations

from typing import TYPE_CHECKING

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


def _select_groups(
    labels: pd.Series,
    selected: list | None,
) -> tuple[NDArray, NDArray[np.int32], NDArray[np.int64]]:
    """Build integer group codes from a categorical Series.

    Parameters
    ----------
    labels
        Categorical Series (from ``adata.obs[groupby]``).
    selected
        Group names to keep, or ``None`` for all groups.
        Must already include the reference group if applicable.

    Returns
    -------
    groups_order
        Selected group names as a numpy array.
    group_codes
        Per-cell int32 codes: ``0..n_groups-1`` for selected cells,
        ``n_groups`` (sentinel) for unselected cells.
    group_sizes
        Number of cells per selected group (int64).
    """
    all_categories = labels.cat.categories

    if selected is None:
        selected = list(all_categories)
    elif len(selected) > 2:
        # Sort to match original category order (scanpy convention)
        cat_order = {str(c): i for i, c in enumerate(all_categories)}
        selected.sort(key=lambda x: cat_order.get(str(x), len(all_categories)))

    n_groups = len(selected)
    groups_order = np.array(selected)

    # Map original category index → selected group index
    str_to_sel = {str(name): idx for idx, name in enumerate(selected)}
    orig_to_sel: dict[int, int] = {}
    for cat_idx, cat_name in enumerate(all_categories):
        sel_idx = str_to_sel.get(str(cat_name))
        if sel_idx is not None:
            orig_to_sel[cat_idx] = sel_idx

    orig_codes = labels.cat.codes.to_numpy()
    group_codes = np.full(len(orig_codes), n_groups, dtype=np.int32)
    for orig_idx, sel_idx in orig_to_sel.items():
        group_codes[orig_codes == orig_idx] = sel_idx

    group_sizes = np.bincount(group_codes, minlength=n_groups + 1)[:n_groups].astype(
        np.int64
    )

    # Validate singlet groups
    invalid_groups = {str(selected[i]) for i in range(n_groups) if group_sizes[i] < 2}
    if invalid_groups:
        msg = (
            f"Could not calculate statistics for groups {', '.join(invalid_groups)} "
            "since they only contain one sample."
        )
        raise ValueError(msg)

    return groups_order, group_codes, group_sizes


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
