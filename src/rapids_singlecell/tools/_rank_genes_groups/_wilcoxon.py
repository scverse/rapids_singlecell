from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc
from rapids_singlecell._utils._csr_to_csc import _fast_csr_to_csc

from ._utils import _choose_chunk_size, _get_column_block

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from ._core import _RankGenes


def _alloc_sort_workspace(n_rows: int, n_cols: int) -> dict[str, cp.ndarray]:
    """
    Pre-allocate CuPy buffers for CUB segmented sort.

    All allocations go through CuPy's memory pool (RMM-compatible).
    Buffers can be reused across chunks with the same or smaller dimensions.
    """
    cub_temp_bytes = _wc.get_sort_temp_bytes(n_rows=n_rows, n_cols=n_cols)
    return {
        "sorted_vals": cp.empty((n_rows, n_cols), dtype=cp.float64, order="F"),
        "sorter": cp.empty((n_rows, n_cols), dtype=cp.int32, order="F"),
        "iota": cp.empty((n_rows, n_cols), dtype=cp.int32, order="F"),
        "offsets": cp.empty(n_cols + 1, dtype=cp.int32),
        "cub_temp": cp.empty(cub_temp_bytes, dtype=cp.uint8),
    }


def _average_ranks(
    matrix: cp.ndarray,
    workspace: dict[str, cp.ndarray] | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute average ranks and tie correction for each column.

    Uses a fused CUB segmented radix sort + CUDA kernel pipeline:

    1. CUB DeviceSegmentedRadixSort sorts each column independently
    2. Binary-search kernel assigns average ranks to original positions
    3. Tie correction kernel computes ``1 - sum(t^3 - t) / (n^3 - n)``

    Parameters
    ----------
    matrix
        Input matrix (n_rows, n_cols), F-order float64.
        Overwritten in-place with average ranks.
    workspace
        Pre-allocated buffers from :func:`_alloc_sort_workspace`.
        If ``None``, a fresh workspace is allocated (convenient for tests).

    Returns
    -------
    ranks, correction
        ranks is the input matrix (now containing average ranks).
        correction is the tie correction factor per column.
    """
    n_rows, n_cols = matrix.shape
    if workspace is None:
        workspace = _alloc_sort_workspace(n_rows, n_cols)
    correction = cp.empty(n_cols, dtype=cp.float64)
    stream = cp.cuda.get_current_stream().ptr
    _wc.compute_ranks(
        matrix,
        correction,
        workspace["sorted_vals"],
        workspace["sorter"],
        workspace["iota"],
        workspace["offsets"],
        workspace["cub_temp"],
        n_rows=n_rows,
        n_cols=n_cols,
        stream=stream,
    )
    return matrix, correction


def wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool = False,
    chunk_size: int | None = None,
) -> Generator[tuple[int, NDArray, NDArray], None, None]:
    """Compute Wilcoxon rank-sum test statistics."""
    # Compute basic stats - uses Aggregate if on GPU, else defers to chunks
    rg._basic_stats()
    X = rg.X
    n_cells, n_total_genes = rg.X.shape
    group_sizes = rg.groups_masks_obs.sum(axis=1).astype(np.int64)

    if rg.ireference is not None:
        # Compare each group against a specific reference group
        yield from _wilcoxon_with_reference(
            rg,
            X,
            n_total_genes,
            group_sizes,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            chunk_size=chunk_size,
        )
    else:
        # Compare each group against "rest" (all other cells)
        yield from _wilcoxon_vs_rest(
            rg,
            X,
            n_cells,
            n_total_genes,
            group_sizes,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            chunk_size=chunk_size,
        )


def _wilcoxon_vs_rest(
    rg: _RankGenes,
    X,
    n_cells: int,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
) -> Generator[tuple[int, NDArray, NDArray], None, None]:
    """Wilcoxon test: each group vs rest of cells."""
    # Warn for small groups
    for name, size in zip(rg.groups_order, group_sizes, strict=False):
        rest = n_cells - size
        if size <= 25 or rest <= 25:
            warnings.warn(
                f"Group {name} has size {size} (rest {rest}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=4,
            )

    group_matrix = cp.asarray(rg.groups_masks_obs.T, dtype=cp.float64)
    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev

    chunk_width = _choose_chunk_size(chunk_size)

    # Accumulate results per group
    all_scores = {i: [] for i in range(len(rg.groups_order))}
    all_pvals = {i: [] for i in range(len(rg.groups_order))}

    # One-time CSR->CSC via fast parallel Numba kernel; _get_column_block
    # then uses direct indptr pointer copy for each chunk.
    if isinstance(X, sp.spmatrix | sp.sparray):
        X = _fast_csr_to_csc(X) if X.format == "csr" else X.tocsc()

    # Pre-allocate sort workspace (reused across all gene chunks)
    workspace = _alloc_sort_workspace(n_cells, chunk_width)

    for start in range(0, n_total_genes, chunk_width):
        stop = min(start + chunk_width, n_total_genes)

        # Slice and convert to dense GPU array (F-order for column ops)
        block = _get_column_block(X, start, stop)

        # Accumulate stats for this chunk
        rg._accumulate_chunk_stats_vs_rest(
            block,
            start,
            stop,
            group_matrix=group_matrix,
            group_sizes_dev=group_sizes_dev,
            n_cells=n_cells,
        )

        ranks, tie_corr = _average_ranks(block, workspace)
        if not tie_correct:
            tie_corr = cp.ones(ranks.shape[1], dtype=cp.float64)

        rank_sums = group_matrix.T @ ranks
        expected = group_sizes_dev[:, None] * (n_cells + 1) / 2.0
        variance = tie_corr[None, :] * group_sizes_dev[:, None] * rest_sizes[:, None]
        variance *= (n_cells + 1) / 12.0
        std = cp.sqrt(variance)
        diff = rank_sums - expected
        if use_continuity:
            diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
        z = diff / std
        cp.nan_to_num(z, copy=False)
        p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))

        z_host = z.get()
        p_host = p_values.get()

        for idx in range(len(rg.groups_order)):
            all_scores[idx].append(z_host[idx])
            all_pvals[idx].append(p_host[idx])

    # Yield results per group
    for group_index in range(len(rg.groups_order)):
        scores = np.concatenate(all_scores[group_index])
        pvals = np.concatenate(all_pvals[group_index])
        yield group_index, scores, pvals


def _wilcoxon_with_reference(
    rg: _RankGenes,
    X,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
) -> Generator[tuple[int, NDArray, NDArray], None, None]:
    """Wilcoxon test: each group vs a specific reference group."""
    mask_ref = rg.groups_masks_obs[rg.ireference]
    n_ref = int(group_sizes[rg.ireference])

    for group_index, mask_obs in enumerate(rg.groups_masks_obs):
        if group_index == rg.ireference:
            continue

        n_group = int(group_sizes[group_index])
        n_combined = n_group + n_ref

        # Warn for small groups
        if n_group <= 25 or n_ref <= 25:
            warnings.warn(
                f"Group {rg.groups_order[group_index]} has size {n_group} "
                f"(reference {n_ref}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=4,
            )

        # Combined mask: group + reference
        mask_combined = mask_obs | mask_ref

        # Subset matrix ONCE before chunking (10x faster than filtering each chunk)
        X_subset = X[mask_combined, :]

        # One-time CSR->CSC via fast parallel Numba kernel
        if isinstance(X_subset, sp.spmatrix | sp.sparray):
            X_subset = (
                _fast_csr_to_csc(X_subset)
                if X_subset.format == "csr"
                else X_subset.tocsc()
            )

        # Create mask for group within the combined array (constant across chunks)
        combined_indices = np.where(mask_combined)[0]
        group_indices_in_combined = np.isin(combined_indices, np.where(mask_obs)[0])
        group_mask_gpu = cp.asarray(group_indices_in_combined)

        chunk_width = _choose_chunk_size(chunk_size)

        # Pre-allocate output arrays and sort workspace for this group pair
        scores = np.empty(n_total_genes, dtype=np.float64)
        pvals = np.empty(n_total_genes, dtype=np.float64)
        workspace = _alloc_sort_workspace(n_combined, chunk_width)

        for start in range(0, n_total_genes, chunk_width):
            stop = min(start + chunk_width, n_total_genes)

            # Get block for combined cells only
            block = _get_column_block(X_subset, start, stop)

            # Accumulate stats for this chunk
            rg._accumulate_chunk_stats_with_ref(
                block,
                start,
                stop,
                group_index=group_index,
                group_mask_gpu=group_mask_gpu,
                n_group=n_group,
                n_ref=n_ref,
            )

            # Ranks for combined group+reference cells
            ranks, tie_corr = _average_ranks(block, workspace)
            if not tie_correct:
                tie_corr = cp.ones(ranks.shape[1], dtype=cp.float64)

            # Rank sum for the group
            rank_sums = (ranks * group_mask_gpu[:, None]).sum(axis=0)

            # Wilcoxon z-score formula for two groups
            expected = n_group * (n_combined + 1) / 2.0
            variance = tie_corr * n_group * n_ref * (n_combined + 1) / 12.0
            std = cp.sqrt(variance)
            diff = rank_sums - expected
            if use_continuity:
                diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
            z = diff / std
            cp.nan_to_num(z, copy=False)
            p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))

            # Fill pre-allocated arrays
            scores[start:stop] = z.get()
            pvals[start:stop] = p_values.get()

        yield group_index, scores, pvals
