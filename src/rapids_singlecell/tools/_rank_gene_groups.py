"""Rank genes according to differential expression."""

from __future__ import annotations

import sys
import warnings
from functools import partial
from typing import TYPE_CHECKING, Literal, assert_never

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.stats.multitest import multipletests

from rapids_singlecell._compat import DaskArray, _meta_dense
from rapids_singlecell._utils._csr_to_csc import _fast_csr_to_csc
from rapids_singlecell.get import X_to_GPU
from rapids_singlecell.get._aggregated import Aggregate
from rapids_singlecell.preprocessing._utils import _check_gpu_X, _sparse_to_dense
from rapids_singlecell.tools._kernels._wilcoxon import (
    _rank_kernel,
    _tie_correction_kernel,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from anndata import AnnData
    from numpy.typing import NDArray

type _CorrMethod = Literal["benjamini-hochberg", "bonferroni"]
type _Method = Literal["logreg", "t-test", "t-test_overestim_var", "wilcoxon"]

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


def _csc_columns_to_gpu(
    X_csc: sp.csc_matrix, start: int, stop: int, n_rows: int
) -> cp.ndarray:
    """
    Extract columns from a scipy CSC matrix via direct pointer copy to GPU.

    Instead of using scipy's ``X[:, start:stop]`` (which rebuilds indices),
    this directly slices the underlying data/indices/indptr arrays and
    constructs a CuPy CSC matrix from them.
    """
    s_ptr = X_csc.indptr[start]
    e_ptr = X_csc.indptr[stop]
    chunk_data = cp.asarray(X_csc.data[s_ptr:e_ptr])
    chunk_indices = cp.asarray(X_csc.indices[s_ptr:e_ptr])
    chunk_indptr = cp.asarray(X_csc.indptr[start : stop + 1] - s_ptr)
    csc_chunk = cpsp.csc_matrix(
        (chunk_data, chunk_indices, chunk_indptr), shape=(n_rows, stop - start)
    )
    return _sparse_to_dense(csc_chunk, order="F").astype(cp.float64)


def _get_column_block(
    X, start: int, stop: int, *, X_csc: sp.csc_matrix | None = None
) -> cp.ndarray:
    """
    Extract a column block from matrix X and convert to dense CuPy array.

    Handles scipy sparse, cupy sparse, numpy arrays, and cupy arrays.
    Returns F-order array for efficient column operations.

    If *X_csc* is provided (pre-computed scipy CSC), columns are extracted
    via direct indptr pointer copy — much faster than scipy's column slicing.
    """
    if X_csc is not None:
        return _csc_columns_to_gpu(X_csc, start, stop, X_csc.shape[0])

    X_chunk = X[:, start:stop]

    match X_chunk:
        case sp.spmatrix() | sp.sparray():
            # SciPy sparse -> CuPy sparse CSC -> dense
            X_chunk = cpsp.csc_matrix(X_chunk.tocsc())
            return _sparse_to_dense(X_chunk, order="F").astype(cp.float64)
        case cpsp.spmatrix():
            # CuPy sparse -> dense
            return _sparse_to_dense(X_chunk, order="F").astype(cp.float64)
        case np.ndarray() | cp.ndarray():
            return cp.asarray(X_chunk, dtype=cp.float64, order="F")
        case _:
            raise ValueError(f"Unsupported matrix type: {type(X_chunk)}")


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


class _RankGenes:
    """Class for computing differential expression statistics on GPU."""

    def __init__(
        self,
        adata: AnnData,
        groups: Iterable[str] | Literal["all"],
        groupby: str,
        *,
        mask_var: NDArray[np.bool_] | None = None,
        reference: Literal["rest"] | str = "rest",
        use_raw: bool | None = None,
        layer: str | None = None,
        comp_pts: bool = False,
        pre_load: bool = False,
    ) -> None:
        # Handle groups parameter
        if groups == "all" or groups is None:
            groups_order: Literal["all"] | list[str] = "all"
        elif isinstance(groups, str | int):
            msg = "Specify a sequence of groups"
            raise ValueError(msg)
        else:
            groups_order = list(groups)
            if isinstance(groups_order[0], int):
                groups_order = [str(n) for n in groups_order]
            if reference != "rest" and reference not in set(groups_order):
                groups_order += [reference]

        self.labels = pd.Series(adata.obs[groupby]).reset_index(drop=True)

        if reference != "rest" and reference not in set(self.labels.cat.categories):
            cats = self.labels.cat.categories.tolist()
            msg = f"reference = {reference} needs to be one of groupby = {cats}."
            raise ValueError(msg)

        self.groups_order, self.groups_masks_obs = _select_groups(
            self.labels, groups_order
        )

        # Validate singlet groups
        invalid_groups = set()
        for name, mask in zip(self.groups_order, self.groups_masks_obs, strict=False):
            if np.count_nonzero(mask) < 2:
                invalid_groups.add(str(name))
        if invalid_groups:
            msg = (
                f"Could not calculate statistics for groups {', '.join(invalid_groups)} "
                "since they only contain one sample."
            )
            raise ValueError(msg)

        # Get data matrix
        if layer is not None:
            if use_raw is True:
                msg = "Cannot specify `layer` and have `use_raw=True`."
                raise ValueError(msg)
            self.X = adata.layers[layer]
            self.var_names = adata.var_names
        elif use_raw is None and adata.raw is not None:
            self.X = adata.raw.X
            self.var_names = adata.raw.var_names
        elif use_raw is True:
            if adata.raw is None:
                msg = "Received `use_raw=True`, but `adata.raw` is empty."
                raise ValueError(msg)
            self.X = adata.raw.X
            self.var_names = adata.raw.var_names
        else:
            self.X = adata.X
            self.var_names = adata.var_names

        # Apply mask_var to select subset of genes
        if mask_var is not None:
            self.X = self.X[:, mask_var]
            self.var_names = self.var_names[mask_var]

        self.pre_load = pre_load

        self.ireference = None
        if reference != "rest":
            self.ireference = np.where(self.groups_order == reference)[0][0]

        # Set up expm1 function based on log base
        base = adata.uns.get("log1p", {}).get("base")
        if base is not None:
            self.expm1_func = lambda x: np.expm1(x * np.log(base))
        else:
            self.expm1_func = np.expm1

        # For logreg
        self.grouping_mask = self.labels.isin(pd.Series(self.groups_order))
        self.grouping = self.labels.loc[self.grouping_mask]

        # For basic stats
        self.comp_pts = comp_pts
        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.pts: np.ndarray | None = None
        self.means_rest: np.ndarray | None = None
        self.vars_rest: np.ndarray | None = None
        self.pts_rest: np.ndarray | None = None

        self.stats: pd.DataFrame | None = None
        self._compute_stats_in_chunks: bool = False

    def _init_stats_arrays(self, n_genes: int) -> None:
        """Pre-allocate stats arrays before chunk loop."""
        n_groups = len(self.groups_order)

        self.means = np.zeros((n_groups, n_genes), dtype=np.float64)
        self.vars = np.zeros((n_groups, n_genes), dtype=np.float64)
        self.pts = (
            np.zeros((n_groups, n_genes), dtype=np.float64) if self.comp_pts else None
        )

        if self.ireference is None:
            self.means_rest = np.zeros((n_groups, n_genes), dtype=np.float64)
            self.vars_rest = np.zeros((n_groups, n_genes), dtype=np.float64)
            self.pts_rest = (
                np.zeros((n_groups, n_genes), dtype=np.float64)
                if self.comp_pts
                else None
            )
        else:
            self.means_rest = None
            self.vars_rest = None
            self.pts_rest = None

    def _basic_stats(self) -> None:
        """Compute means, vars, and pts for each group.

        If data is already on GPU, uses Aggregate for fast single-pass computation.
        Otherwise, sets flag for chunk-based computation during wilcoxon loop,
        unless force_compute is True (needed for t-test).

        Parameters
        ----------
        force_compute
            If True, compute stats immediately even if data is not on GPU.
            Required for t-test methods that don't use chunked computation.
        """
        n_genes = self.X.shape[1]
        n_groups = len(self.groups_order)

        # Check if data is already on GPU
        try:
            _check_gpu_X(self.X, allow_dask=True)
        except TypeError:
            is_on_gpu = False
        else:
            is_on_gpu = True

        if not is_on_gpu:
            # Data not on GPU - defer to chunk-based computation
            self._compute_stats_in_chunks = True
            self._init_stats_arrays(n_genes)
            return

        # Data is on GPU - use Aggregate for fast computation
        self._compute_stats_in_chunks = False

        agg = Aggregate(groupby=self.labels.cat, data=self.X)

        if isinstance(self.X, DaskArray):
            result = agg.count_mean_var_dask(dof=1)
        elif cpsp.issparse(self.X):
            result = agg.count_mean_var_sparse(dof=1)
        else:
            result = agg.count_mean_var_dense(dof=1)

        # Map results to selected groups order
        cat_names = list(self.labels.cat.categories)

        means = np.zeros((n_groups, n_genes), dtype=np.float64)
        vars_ = np.zeros((n_groups, n_genes), dtype=np.float64)
        pts = np.zeros((n_groups, n_genes), dtype=np.float64) if self.comp_pts else None

        for idx, group_name in enumerate(self.groups_order):
            group_name_str = str(group_name)
            if group_name_str in cat_names:
                cat_idx = cat_names.index(group_name_str)
                means[idx] = cp.asnumpy(result["mean"][cat_idx])
                vars_[idx] = cp.asnumpy(result["var"][cat_idx])
                if self.comp_pts:
                    n_cells = self.groups_masks_obs[idx].sum()
                    pts[idx] = cp.asnumpy(result["count_nonzero"][cat_idx]) / n_cells

        self.means = means
        # Clip tiny negative variances to 0 (floating-point precision artifacts)
        self.vars = np.maximum(vars_, 0)
        self.pts = pts

        # Compute rest statistics if reference='rest'
        if self.ireference is None:
            self.means_rest = np.zeros((n_groups, n_genes), dtype=np.float64)
            self.vars_rest = np.zeros((n_groups, n_genes), dtype=np.float64)
            self.pts_rest = (
                np.zeros((n_groups, n_genes), dtype=np.float64)
                if self.comp_pts
                else None
            )

            n_cells_per_group = np.array([mask.sum() for mask in self.groups_masks_obs])
            total_sum = self.means * n_cells_per_group[:, None]
            total_sum_all = total_sum.sum(axis=0)

            # Compute total sum of squares for variance calculation
            total_sq_sum = (
                self.vars * (n_cells_per_group[:, None] - 1)
                + self.means**2 * n_cells_per_group[:, None]
            ).sum(axis=0)

            if self.comp_pts:
                total_count = (self.pts * n_cells_per_group[:, None]).sum(axis=0)

            total_n = n_cells_per_group.sum()

            for idx in range(n_groups):
                n_group = n_cells_per_group[idx]
                n_rest = total_n - n_group

                if n_rest > 0:
                    rest_sum = total_sum_all - total_sum[idx]
                    self.means_rest[idx] = rest_sum / n_rest

                    group_sq_sum = (
                        self.vars[idx] * (n_group - 1) + self.means[idx] ** 2 * n_group
                    )
                    rest_sq_sum = total_sq_sum - group_sq_sum

                    rest_mean_sq = self.means_rest[idx] ** 2
                    if n_rest > 1:
                        self.vars_rest[idx] = np.maximum(
                            (rest_sq_sum / n_rest - rest_mean_sq)
                            * n_rest
                            / (n_rest - 1),
                            0,
                        )

                    if self.comp_pts:
                        rest_count = total_count - self.pts[idx] * n_group
                        self.pts_rest[idx] = rest_count / n_rest
        else:
            self.means_rest = None
            self.vars_rest = None
            self.pts_rest = None

    def _accumulate_chunk_stats_vs_rest(
        self,
        block: cp.ndarray,
        start: int,
        stop: int,
        *,
        group_matrix: cp.ndarray,
        group_sizes_dev: cp.ndarray,
        n_cells: int,
    ) -> None:
        """Compute and store stats for one gene chunk (vs rest mode)."""
        if not self._compute_stats_in_chunks:
            return  # Stats already computed via Aggregate

        rest_sizes = n_cells - group_sizes_dev

        # Group sums and sum of squares
        group_sums = group_matrix.T @ block
        group_sum_sq = group_matrix.T @ (block**2)

        # Means
        chunk_means = group_sums / group_sizes_dev[:, None]
        self.means[:, start:stop] = cp.asnumpy(chunk_means)

        # Variances (with Bessel correction)
        chunk_vars = group_sum_sq / group_sizes_dev[:, None] - chunk_means**2
        chunk_vars *= group_sizes_dev[:, None] / (group_sizes_dev[:, None] - 1)
        self.vars[:, start:stop] = cp.asnumpy(chunk_vars)

        # Pts (fraction expressing)
        if self.comp_pts:
            group_nnz = group_matrix.T @ (block != 0).astype(cp.float64)
            self.pts[:, start:stop] = cp.asnumpy(group_nnz / group_sizes_dev[:, None])

        # Rest statistics
        if self.ireference is None:
            total_sum = block.sum(axis=0)
            total_sum_sq = (block**2).sum(axis=0)

            rest_sums = total_sum[None, :] - group_sums
            rest_means = rest_sums / rest_sizes[:, None]
            self.means_rest[:, start:stop] = cp.asnumpy(rest_means)

            rest_sum_sq = total_sum_sq[None, :] - group_sum_sq
            rest_vars = rest_sum_sq / rest_sizes[:, None] - rest_means**2
            rest_vars *= rest_sizes[:, None] / (rest_sizes[:, None] - 1)
            self.vars_rest[:, start:stop] = cp.asnumpy(rest_vars)

            if self.comp_pts:
                total_nnz = (block != 0).sum(axis=0)
                rest_nnz = total_nnz[None, :] - group_nnz
                self.pts_rest[:, start:stop] = cp.asnumpy(
                    rest_nnz / rest_sizes[:, None]
                )

    def _accumulate_chunk_stats_with_ref(
        self,
        block: cp.ndarray,
        start: int,
        stop: int,
        *,
        group_index: int,
        group_mask_gpu: cp.ndarray,
        n_group: int,
        n_ref: int,
    ) -> None:
        """Compute and store stats for one gene chunk (with reference mode)."""
        if not self._compute_stats_in_chunks:
            return  # Stats already computed via Aggregate

        # Group stats
        group_data = block[group_mask_gpu]
        group_mean = group_data.mean(axis=0)
        self.means[group_index, start:stop] = cp.asnumpy(group_mean)

        if n_group > 1:
            group_var = group_data.var(axis=0, ddof=1)
            self.vars[group_index, start:stop] = cp.asnumpy(group_var)

        if self.comp_pts:
            group_nnz = (group_data != 0).sum(axis=0)
            self.pts[group_index, start:stop] = cp.asnumpy(group_nnz / n_group)

        # Reference stats (only compute once, on first non-reference group)
        if self.means[self.ireference, start] == 0:  # Not yet computed
            ref_data = block[~group_mask_gpu]
            ref_mean = ref_data.mean(axis=0)
            self.means[self.ireference, start:stop] = cp.asnumpy(ref_mean)

            if n_ref > 1:
                ref_var = ref_data.var(axis=0, ddof=1)
                self.vars[self.ireference, start:stop] = cp.asnumpy(ref_var)

            if self.comp_pts:
                ref_nnz = (ref_data != 0).sum(axis=0)
                self.pts[self.ireference, start:stop] = cp.asnumpy(ref_nnz / n_ref)

    def t_test(
        self, method: Literal["t-test", "t-test_overestim_var"]
    ) -> Generator[tuple[int, NDArray, NDArray], None, None]:
        """Compute t-test statistics using Welch's t-test."""
        from scipy import stats

        self._basic_stats()

        for group_index, (mask_obs, mean_group, var_group) in enumerate(
            zip(self.groups_masks_obs, self.means, self.vars, strict=True)
        ):
            if self.ireference is not None and group_index == self.ireference:
                continue

            ns_group = np.count_nonzero(mask_obs)

            if self.ireference is not None:
                mean_rest = self.means[self.ireference]
                var_rest = self.vars[self.ireference]
                ns_other = np.count_nonzero(self.groups_masks_obs[self.ireference])
            else:
                mean_rest = self.means_rest[group_index]
                var_rest = self.vars_rest[group_index]
                ns_other = self.X.shape[0] - ns_group

            if method == "t-test":
                ns_rest = ns_other
            elif method == "t-test_overestim_var":
                # Hack for overestimating the variance for small groups
                ns_rest = ns_group
            else:
                msg = "Method does not exist."
                raise ValueError(msg)

            # Welch's t-test using pre-computed stats
            with np.errstate(invalid="ignore"):
                scores, pvals = stats.ttest_ind_from_stats(
                    mean1=mean_group,
                    std1=np.sqrt(var_group),
                    nobs1=ns_group,
                    mean2=mean_rest,
                    std2=np.sqrt(var_rest),
                    nobs2=ns_rest,
                    equal_var=False,  # Welch's
                )

            # Handle NaN values (when means are the same and vars are 0)
            scores[np.isnan(scores)] = 0
            pvals[np.isnan(pvals)] = 1

            yield group_index, scores, pvals

    def wilcoxon(
        self, *, tie_correct: bool, chunk_size: int | None = None
    ) -> Generator[tuple[int, NDArray, NDArray, NDArray], None, None]:
        """Compute Wilcoxon rank-sum test statistics."""
        # Compute basic stats - uses Aggregate if on GPU, else defers to chunks
        self._basic_stats()
        X = self.X
        n_cells, n_total_genes = self.X.shape
        group_sizes = self.groups_masks_obs.sum(axis=1).astype(np.int64)

        if self.ireference is not None:
            # Compare each group against a specific reference group
            yield from self._wilcoxon_with_reference(
                X,
                n_total_genes,
                group_sizes,
                tie_correct=tie_correct,
                chunk_size=chunk_size,
            )
        else:
            # Compare each group against "rest" (all other cells)
            yield from self._wilcoxon_vs_rest(
                X,
                n_cells,
                n_total_genes,
                group_sizes,
                tie_correct=tie_correct,
                chunk_size=chunk_size,
            )

    def _wilcoxon_vs_rest(
        self,
        X,
        n_cells: int,
        n_total_genes: int,
        group_sizes: NDArray,
        *,
        tie_correct: bool,
        chunk_size: int | None,
    ) -> Generator[tuple[int, NDArray, NDArray, NDArray], None, None]:
        """Wilcoxon test: each group vs rest of cells."""
        # Warn for small groups
        for name, size in zip(self.groups_order, group_sizes, strict=False):
            rest = n_cells - size
            if size <= 25 or rest <= 25:
                warnings.warn(
                    f"Group {name} has size {size} (rest {rest}); normal approximation "
                    "of the Wilcoxon statistic may be inaccurate.",
                    RuntimeWarning,
                    stacklevel=4,
                )

        group_matrix = cp.asarray(self.groups_masks_obs.T, dtype=cp.float64)
        group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
        rest_sizes = n_cells - group_sizes_dev

        chunk_width = _choose_chunk_size(chunk_size)

        # Accumulate results per group
        all_scores = {i: [] for i in range(len(self.groups_order))}
        all_pvals = {i: [] for i in range(len(self.groups_order))}

        # One-time CSR→CSC for scipy sparse (fast Numba kernel + direct
        # pointer copy is much faster than scipy's X[:, start:stop]).
        X_csc = None
        if isinstance(X, sp.spmatrix | sp.sparray):
            X_csc = _fast_csr_to_csc(X) if X.format == "csr" else X.tocsc()

        for start in range(0, n_total_genes, chunk_width):
            stop = min(start + chunk_width, n_total_genes)

            # Slice and convert to dense GPU array (F-order for column ops)
            block = _get_column_block(X, start, stop, X_csc=X_csc)

            # Accumulate stats for this chunk
            self._accumulate_chunk_stats_vs_rest(
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
            variance = (
                tie_corr[None, :] * group_sizes_dev[:, None] * rest_sizes[:, None]
            )
            variance *= (n_cells + 1) / 12.0
            std = cp.sqrt(variance)
            z = (rank_sums - expected) / std
            cp.nan_to_num(z, copy=False)
            p_values = 2.0 * (1.0 - cupyx_special.ndtr(cp.abs(z)))

            z_host = z.get()
            p_host = p_values.get()

            for idx in range(len(self.groups_order)):
                all_scores[idx].append(z_host[idx])
                all_pvals[idx].append(p_host[idx])

        # Yield results per group
        for group_index in range(len(self.groups_order)):
            scores = np.concatenate(all_scores[group_index])
            pvals = np.concatenate(all_pvals[group_index])
            yield group_index, scores, pvals

    def _wilcoxon_with_reference(
        self,
        X,
        n_total_genes: int,
        group_sizes: NDArray,
        *,
        tie_correct: bool,
        chunk_size: int | None,
    ) -> Generator[tuple[int, NDArray, NDArray], None, None]:
        """Wilcoxon test: each group vs a specific reference group."""
        mask_ref = self.groups_masks_obs[self.ireference]
        n_ref = int(group_sizes[self.ireference])

        for group_index, mask_obs in enumerate(self.groups_masks_obs):
            if group_index == self.ireference:
                continue

            n_group = int(group_sizes[group_index])
            n_combined = n_group + n_ref

            # Warn for small groups
            if n_group <= 25 or n_ref <= 25:
                warnings.warn(
                    f"Group {self.groups_order[group_index]} has size {n_group} "
                    f"(reference {n_ref}); normal approximation "
                    "of the Wilcoxon statistic may be inaccurate.",
                    RuntimeWarning,
                    stacklevel=4,
                )

            # Combined mask: group + reference
            mask_combined = mask_obs | mask_ref

            # Subset matrix ONCE before chunking (10x faster than filtering each chunk)
            X_subset = X[mask_combined, :]

            # One-time CSR→CSC for scipy sparse subsets
            X_subset_csc = None
            if isinstance(X_subset, sp.spmatrix | sp.sparray):
                X_subset_csc = (
                    _fast_csr_to_csc(X_subset)
                    if X_subset.format == "csr"
                    else X_subset.tocsc()
                )

            # Create mask for group within the combined array (constant across chunks)
            combined_indices = np.where(mask_combined)[0]
            group_indices_in_combined = np.isin(combined_indices, np.where(mask_obs)[0])
            group_mask_gpu = cp.asarray(group_indices_in_combined)

            chunk_width = _choose_chunk_size(chunk_size)

            # Pre-allocate output arrays
            scores = np.empty(n_total_genes, dtype=np.float64)
            pvals = np.empty(n_total_genes, dtype=np.float64)

            for start in range(0, n_total_genes, chunk_width):
                stop = min(start + chunk_width, n_total_genes)

                # Get block for combined cells only
                block = _get_column_block(X_subset, start, stop, X_csc=X_subset_csc)

                # Accumulate stats for this chunk
                self._accumulate_chunk_stats_with_ref(
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

            yield group_index, scores, pvals

    def logreg(self, **kwds) -> Generator[tuple[int, NDArray, None], None, None]:
        """Compute logistic regression scores."""
        if len(self.groups_order) == 1:
            msg = "Cannot perform logistic regression on a single cluster."
            raise ValueError(msg)

        X = self.X[self.grouping_mask.values, :]

        grouping_logreg = self.grouping.cat.codes.to_numpy().astype(X.dtype)
        uniques = np.unique(grouping_logreg)
        for idx, cat in enumerate(uniques):
            grouping_logreg[np.where(grouping_logreg == cat)] = idx

        if isinstance(X, DaskArray):
            import dask.array as da
            from cuml.dask.linear_model import LogisticRegression

            grouping_logreg = da.from_array(
                grouping_logreg,
                chunks=(X.chunks[0]),
                meta=_meta_dense(grouping_logreg.dtype),
            )
        else:
            from cuml.linear_model import LogisticRegression

        clf = LogisticRegression(**kwds)
        clf.fit(X, grouping_logreg)
        scores_all = cp.array(clf.coef_)
        if len(self.groups_order) == scores_all.shape[1]:
            scores_all = scores_all.T

        for igroup, _group in enumerate(self.groups_order):
            if len(self.groups_order) <= 2:
                scores = scores_all[0].get()
            else:
                scores = scores_all[igroup].get()

            yield igroup, scores, None

            if len(self.groups_order) <= 2:
                break

    def compute_statistics(
        self,
        method: _Method,
        *,
        corr_method: _CorrMethod = "benjamini-hochberg",
        n_genes_user: int | None = None,
        rankby_abs: bool = False,
        tie_correct: bool = False,
        chunk_size: int | None = None,
        **kwds,
    ) -> None:
        """Compute statistics for all groups."""
        if self.pre_load or method in {"t-test", "t-test_overestim_var"}:
            self.X = X_to_GPU(self.X)
        if method in {"t-test", "t-test_overestim_var"}:
            generate_test_results = self.t_test(method)
        elif method == "wilcoxon":
            if isinstance(self.X, DaskArray):
                msg = "Wilcoxon test is not supported for Dask arrays. Please convert your data to CuPy arrays."
                raise ValueError(msg)
            generate_test_results = self.wilcoxon(
                tie_correct=tie_correct, chunk_size=chunk_size
            )
        elif method == "logreg":
            generate_test_results = self.logreg(**kwds)
        else:
            assert_never(method)

        n_genes = self.X.shape[1]

        # Collect all stats data first to avoid DataFrame fragmentation
        stats_data: dict[tuple[str, str], np.ndarray] = {}

        for group_index, scores, pvals in generate_test_results:
            group_name = str(self.groups_order[group_index])

            if n_genes_user is not None:
                scores_sort = np.abs(scores) if rankby_abs else scores
                global_indices = _select_top_n(scores_sort, n_genes_user)
            else:
                global_indices = slice(None)

            if n_genes_user is not None:
                stats_data[group_name, "names"] = np.asarray(self.var_names)[
                    global_indices
                ]

            stats_data[group_name, "scores"] = scores[global_indices]

            if pvals is not None:
                stats_data[group_name, "pvals"] = pvals[global_indices]
                if corr_method == "benjamini-hochberg":
                    pvals_clean = np.array(pvals, copy=True)
                    pvals_clean[np.isnan(pvals_clean)] = 1.0
                    _, pvals_adj, _, _ = multipletests(
                        pvals_clean, alpha=0.05, method="fdr_bh"
                    )
                elif corr_method == "bonferroni":
                    pvals_adj = np.minimum(pvals * n_genes, 1.0)
                stats_data[group_name, "pvals_adj"] = pvals_adj[global_indices]

            # Compute logfoldchanges from accumulated means (like scanpy)
            if self.means is not None:
                mean_group = self.means[group_index]
                if self.ireference is None:
                    mean_rest = self.means_rest[group_index]
                else:
                    mean_rest = self.means[self.ireference]
                foldchanges = (self.expm1_func(mean_group) + EPS) / (
                    self.expm1_func(mean_rest) + EPS
                )
                stats_data[group_name, "logfoldchanges"] = np.log2(
                    foldchanges[global_indices]
                )

        # Create DataFrame all at once to avoid fragmentation
        if stats_data:
            self.stats = pd.DataFrame(stats_data)
            self.stats.columns = pd.MultiIndex.from_tuples(self.stats.columns)
            if n_genes_user is None:
                self.stats.index = self.var_names
        else:
            self.stats = None


def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    mask_var: NDArray[np.bool_] | str | None = None,
    use_raw: bool | None = None,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: str | None = None,
    method: _Method | None = None,
    corr_method: _CorrMethod = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: str | None = None,
    chunk_size: int | None = None,
    pre_load: bool = False,
    **kwds,
) -> None:
    """
    Rank genes for characterizing groups using GPU acceleration.

    Expects logarithmized data.

    .. note::
        **Dask support:** Only `'t-test'` and `'t-test_overestim_var'` methods
        support Dask arrays. The `'wilcoxon'` and `'logreg'` methods do not
        support Dask arrays and will raise an error if used with Dask input.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    mask_var
        Select subset of genes to use in statistical tests.
        Can be a boolean array of shape `(n_vars,)` or a key in `adata.var`.
    use_raw
        Use `raw` attribute of `adata` if present.
    groups
        Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `'all'` (default), for all groups.
    reference
        If `'rest'`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    n_genes
        The number of genes that appear in the returned tables.
        Defaults to all genes.
    rankby_abs
        Rank genes by the absolute value of the score, not by the
        score. The returned scores are never the absolute values.
    pts
        Compute the fraction of cells expressing the genes.
    key_added
        The key in `adata.uns` information is saved to.
    method
        `'t-test'` uses Welch's t-test (default),
        `'t-test_overestim_var'` overestimates variance of each group,
        `'wilcoxon'` uses Wilcoxon rank-sum,
        `'logreg'` uses logistic regression.
    corr_method
        p-value correction method. Used only for `'t-test'`, `'t-test_overestim_var'`, and `'wilcoxon'`.
    tie_correct
        Use tie correction for `'wilcoxon'` scores.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    chunk_size
        Number of genes to process at once. If None, automatically determined
        based on available GPU memory. Used only for `'wilcoxon'`.
    pre_load
        Pre-load the data into GPU memory. Used only for `'wilcoxon'`.
    **kwds
        Additional arguments passed to the method. For `'logreg'`, these are
        passed to :class:`cuml.linear_model.LogisticRegression`.

    Returns
    -------
    Updates `adata` with the following fields:

    `adata.uns['rank_genes_groups' | key_added]['names']`
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['scores']`
        Structured array to be indexed by group id storing the z-score
        underlying the computation of a p-value for each gene for each
        group. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['logfoldchanges']`
        Structured array to be indexed by group id storing the log2
        fold change for each gene for each group.
    `adata.uns['rank_genes_groups' | key_added]['pvals']`
        p-values. Only for `'t-test'`, `'t-test_overestim_var'`, and `'wilcoxon'`.
    `adata.uns['rank_genes_groups' | key_added]['pvals_adj']`
        Corrected p-values. Only for `'t-test'`, `'t-test_overestim_var'`, and `'wilcoxon'`.
    `adata.uns['rank_genes_groups' | key_added]['pts']`
        Fraction of cells expressing genes per group. Only if `pts=True`.
    `adata.uns['rank_genes_groups' | key_added]['pts_rest']`
        Fraction of cells expressing genes in rest. Only if `pts=True` and `reference='rest'`.
    """
    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        msg = "corr_method must be either 'benjamini-hochberg' or 'bonferroni'."
        raise ValueError(msg)

    if method is None:
        method = "t-test"

    if method not in {"logreg", "t-test", "t-test_overestim_var", "wilcoxon"}:
        msg = f"method must be one of 'logreg', 't-test', 't-test_overestim_var', 'wilcoxon'. Got {method!r}."
        raise ValueError(msg)

    if key_added is None:
        key_added = "rank_genes_groups"

    # Process mask_var: convert string to boolean array
    mask_var_array: NDArray[np.bool_] | None = None
    if mask_var is not None:
        if isinstance(mask_var, str):
            if mask_var not in adata.var.columns:
                msg = f"mask_var key {mask_var!r} not found in adata.var."
                raise KeyError(msg)
            mask_var_array = adata.var[mask_var].values.astype(bool)
        else:
            mask_var_array = np.asarray(mask_var, dtype=bool)
            if mask_var_array.shape[0] != adata.n_vars:
                msg = f"mask_var has wrong shape: {mask_var_array.shape[0]} != {adata.n_vars}"
                raise ValueError(msg)

    test_obj = _RankGenes(
        adata,
        groups,
        groupby,
        mask_var=mask_var_array,
        reference=reference,
        use_raw=use_raw,
        layer=layer,
        comp_pts=pts,
        pre_load=pre_load,
    )

    # Determine n_genes_user
    n_genes_user = n_genes
    if n_genes_user is None or n_genes_user > test_obj.X.shape[1]:
        n_genes_user = test_obj.X.shape[1]

    test_obj.compute_statistics(
        method,
        corr_method=corr_method,
        n_genes_user=n_genes_user,
        rankby_abs=rankby_abs,
        tie_correct=tie_correct,
        chunk_size=chunk_size,
        **kwds,
    )

    # Build output
    test_obj.stats.columns = test_obj.stats.columns.swaplevel()

    dtypes = {
        "names": "U50",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }

    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "groupby": groupby,
        "reference": reference,
        "method": method,
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method,
    }

    # Store pts results if computed
    if test_obj.pts is not None:
        groups_names = [str(name) for name in test_obj.groups_order]
        adata.uns[key_added]["pts"] = pd.DataFrame(
            test_obj.pts.T, index=test_obj.var_names, columns=groups_names
        )
    if test_obj.pts_rest is not None:
        adata.uns[key_added]["pts_rest"] = pd.DataFrame(
            test_obj.pts_rest.T, index=test_obj.var_names, columns=groups_names
        )

    if method == "wilcoxon":
        adata.uns[key_added]["params"]["tie_correct"] = tie_correct

    for col in test_obj.stats.columns.levels[0]:
        if col in dtypes:
            adata.uns[key_added][col] = test_obj.stats[col].to_records(
                index=False, column_dtypes=dtypes[col]
            )


if TYPE_CHECKING:
    from warnings import deprecated
else:
    if sys.version_info >= (3, 13):
        from warnings import deprecated as _deprecated
    else:
        from typing_extensions import deprecated as _deprecated
    deprecated = partial(_deprecated, category=FutureWarning)


@deprecated(
    "rank_genes_groups_logreg is deprecated. "
    "Use rank_genes_groups(method='logreg') instead."
)
def rank_genes_groups_logreg(
    adata: AnnData,
    groupby: str,
    *,
    groups: Literal["all"] | Iterable[str] = "all",
    use_raw: bool | None = None,
    reference: str = "rest",
    n_genes: int | None = None,
    key_added: str | None = None,
    layer: str | None = None,
    **kwds,
) -> None:
    rank_genes_groups(
        adata,
        groupby,
        groups=groups,
        use_raw=use_raw,
        reference=reference,
        n_genes=n_genes,
        key_added=key_added,
        method="logreg",
        layer=layer,
        **kwds,
    )
