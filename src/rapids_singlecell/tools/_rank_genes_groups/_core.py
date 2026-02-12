"""Core _RankGenes class for GPU differential expression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.get import X_to_GPU
from rapids_singlecell.get._aggregated import Aggregate
from rapids_singlecell.preprocessing._utils import _check_gpu_X

from ._utils import EPS, _select_groups, _select_top_n

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from anndata import AnnData
    from numpy.typing import NDArray

    from . import _CorrMethod, _Method


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
        from ._ttest import t_test

        return t_test(self, method)

    def wilcoxon(
        self, *, tie_correct: bool, chunk_size: int | None = None
    ) -> Generator[tuple[int, NDArray, NDArray], None, None]:
        """Compute Wilcoxon rank-sum test statistics."""
        from ._wilcoxon import wilcoxon

        return wilcoxon(self, tie_correct=tie_correct, chunk_size=chunk_size)

    def wilcoxon_binned(
        self,
        *,
        n_bins: int = 1000,
        chunk_size: int | None = None,
    ) -> Generator[tuple[int, NDArray, NDArray], None, None]:
        """Histogram-based approximate Wilcoxon rank-sum test (one-vs-rest)."""
        from ._wilcoxon_binned import wilcoxon_binned

        return wilcoxon_binned(self, n_bins=n_bins, chunk_size=chunk_size)

    def logreg(self, **kwds) -> Generator[tuple[int, NDArray, None], None, None]:
        """Compute logistic regression scores."""
        from ._logreg import logreg

        return logreg(self, **kwds)

    def compute_statistics(
        self,
        method: _Method,
        *,
        corr_method: _CorrMethod = "benjamini-hochberg",
        n_genes_user: int | None = None,
        rankby_abs: bool = False,
        tie_correct: bool = False,
        chunk_size: int | None = None,
        n_bins: int = 1000,
        **kwds,
    ) -> None:
        """Compute statistics for all groups."""
        if self.pre_load or method in {
            "t-test",
            "t-test_overestim_var",
            "wilcoxon_binned",
        }:
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
        elif method == "wilcoxon_binned":
            generate_test_results = self.wilcoxon_binned(
                n_bins=n_bins, chunk_size=chunk_size
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

            # Compute logfoldchanges
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
