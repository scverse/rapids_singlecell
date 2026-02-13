from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._utils._csr_to_csc import _fast_csr_to_csc

from ._kernels._wilcoxon import _rank_kernel, _tie_correction_kernel
from ._utils import _choose_chunk_size, _get_column_block, _round_up_to_warp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes


def _average_ranks(
    matrix: cp.ndarray, *, return_sorted: bool = False
) -> cp.ndarray | tuple[cp.ndarray, cp.ndarray]:
    """
    Compute average ranks for each column using GPU kernel.

    Uses scipy.stats.rankdata 'average' method: ties get the average
    of the ranks they would span.

    Parameters
    ----------
    matrix
        Input matrix (n_rows, n_cols)
    return_sorted
        If True, also return sorted values (useful for tie correction)

    Returns
    -------
    ranks or (ranks, sorted_vals)
    """
    n_rows, n_cols = matrix.shape

    # Sort each column
    sorter = cp.argsort(matrix, axis=0)
    sorted_vals = cp.take_along_axis(matrix, sorter, axis=0)

    # Ensure F-order for kernel (columns contiguous in memory)
    sorted_vals = cp.asfortranarray(sorted_vals)
    sorter = cp.asfortranarray(sorter.astype(cp.int32))

    # Launch kernel: one block per column, threads must be multiple of WARP_SIZE
    threads_per_block = _round_up_to_warp(n_rows)
    blocks = n_cols
    _rank_kernel(
        (blocks,),
        (threads_per_block,),
        (sorted_vals, sorter, matrix, n_rows, n_cols),
    )

    if return_sorted:
        return matrix, sorted_vals
    return matrix


def _tie_correction(sorted_vals: cp.ndarray) -> cp.ndarray:
    """
    Compute tie correction factor for Wilcoxon test.

    Takes pre-sorted values (column-wise) to avoid re-sorting.
    Formula: tc = 1 - sum(t^3 - t) / (n^3 - n)
    where t is the count of tied values.
    """
    n_rows, n_cols = sorted_vals.shape
    correction = cp.ones(n_cols, dtype=cp.float64)

    if n_rows < 2:
        return correction

    # Ensure F-order
    sorted_vals = cp.asfortranarray(sorted_vals)

    # Threads must be multiple of WARP_SIZE for correct warp reduction
    threads_per_block = _round_up_to_warp(n_rows)
    _tie_correction_kernel(
        (n_cols,),
        (threads_per_block,),
        (sorted_vals, correction, n_rows, n_cols),
    )

    return correction


def wilcoxon(
    rg: _RankGenes, *, tie_correct: bool, chunk_size: int | None = None
) -> list[tuple[int, NDArray, NDArray]]:
    """Compute Wilcoxon rank-sum test statistics."""
    # Compute basic stats - uses Aggregate if on GPU, else defers to chunks
    rg._basic_stats()
    X = rg.X
    n_cells, n_total_genes = rg.X.shape
    group_sizes = rg.group_sizes

    if rg.ireference is not None:
        # Compare each group against a specific reference group
        return _wilcoxon_with_reference(
            rg,
            X,
            n_total_genes,
            group_sizes,
            tie_correct=tie_correct,
            chunk_size=chunk_size,
        )
    # Compare each group against "rest" (all other cells)
    return _wilcoxon_vs_rest(
        rg,
        X,
        n_cells,
        n_total_genes,
        group_sizes,
        tie_correct=tie_correct,
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
    chunk_size: int | None,
) -> list[tuple[int, NDArray, NDArray]]:
    """Wilcoxon test: each group vs rest of cells."""
    n_groups = len(rg.groups_order)

    # Warn for small groups
    for name, size in zip(rg.groups_order, group_sizes, strict=True):
        rest = n_cells - size
        if size <= 25 or rest <= 25:
            warnings.warn(
                f"Group {name} has size {size} (rest {rest}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=4,
            )

    # Build one-hot indicator matrix from group codes
    codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int64)
    group_matrix = cp.zeros((n_cells, n_groups), dtype=cp.float64)
    valid_idx = cp.where(codes_gpu < n_groups)[0]
    group_matrix[valid_idx, codes_gpu[valid_idx]] = 1.0

    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev

    chunk_width = _choose_chunk_size(chunk_size)

    # Accumulate results per group
    all_scores: dict[int, list] = {i: [] for i in range(n_groups)}
    all_pvals: dict[int, list] = {i: [] for i in range(n_groups)}

    # One-time CSR->CSC via fast parallel Numba kernel; _get_column_block
    # then uses direct indptr pointer copy for each chunk.
    if isinstance(X, sp.spmatrix | sp.sparray):
        X = _fast_csr_to_csc(X) if X.format == "csr" else X.tocsc()

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

        if tie_correct:
            ranks, sorted_vals = _average_ranks(block, return_sorted=True)
            tie_corr = _tie_correction(sorted_vals)
        else:
            ranks = _average_ranks(block)
            tie_corr = cp.ones(ranks.shape[1], dtype=cp.float64)

        rank_sums = group_matrix.T @ ranks
        expected = group_sizes_dev[:, None] * (n_cells + 1) / 2.0
        variance = tie_corr[None, :] * group_sizes_dev[:, None] * rest_sizes[:, None]
        variance *= (n_cells + 1) / 12.0
        std = cp.sqrt(variance)
        z = (rank_sums - expected) / std
        cp.nan_to_num(z, copy=False)
        p_values = 2.0 * (1.0 - cupyx_special.ndtr(cp.abs(z)))

        z_host = z.get()
        p_host = p_values.get()

        for idx in range(n_groups):
            all_scores[idx].append(z_host[idx])
            all_pvals[idx].append(p_host[idx])

    # Collect results per group
    return [
        (gi, np.concatenate(all_scores[gi]), np.concatenate(all_pvals[gi]))
        for gi in range(n_groups)
    ]


def _wilcoxon_with_reference(
    rg: _RankGenes,
    X,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    chunk_size: int | None,
) -> list[tuple[int, NDArray, NDArray]]:
    """Wilcoxon test: each group vs a specific reference group."""
    codes = rg.group_codes
    n_ref = int(group_sizes[rg.ireference])
    mask_ref = codes == rg.ireference

    results: list[tuple[int, NDArray, NDArray]] = []

    for group_index in range(len(rg.groups_order)):
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
        mask_obs = codes == group_index
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

        # Within the combined array, True = group cell, False = reference cell
        group_mask_gpu = cp.asarray(mask_obs[mask_combined])

        chunk_width = _choose_chunk_size(chunk_size)

        # Pre-allocate output arrays
        scores = np.empty(n_total_genes, dtype=np.float64)
        pvals = np.empty(n_total_genes, dtype=np.float64)

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
            if tie_correct:
                ranks, sorted_vals = _average_ranks(block, return_sorted=True)
                tie_corr = _tie_correction(sorted_vals)
            else:
                ranks = _average_ranks(block)
                tie_corr = cp.ones(ranks.shape[1], dtype=cp.float64)

            # Rank sum for the group
            rank_sums = (ranks * group_mask_gpu[:, None]).sum(axis=0)

            # Wilcoxon z-score formula for two groups
            expected = n_group * (n_combined + 1) / 2.0
            variance = tie_corr * n_group * n_ref * (n_combined + 1) / 12.0
            std = cp.sqrt(variance)
            z = (rank_sums - expected) / std
            cp.nan_to_num(z, copy=False)
            p_values = 2.0 * (1.0 - cupyx_special.ndtr(cp.abs(z)))

            # Fill pre-allocated arrays
            scores[start:stop] = z.get()
            pvals[start:stop] = p_values.get()

        results.append((group_index, scores, pvals))

    return results
