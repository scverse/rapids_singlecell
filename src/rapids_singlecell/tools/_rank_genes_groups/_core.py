from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from typing import TYPE_CHECKING, Literal, assert_never

import cupy as cp
import numpy as np
import pandas as pd

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.get import X_to_GPU
from rapids_singlecell.get._aggregated import Aggregate
from rapids_singlecell.preprocessing._utils import _check_gpu_X

from ._utils import (
    EPS,
    _benjamini_hochberg,
    _select_groups,
    _select_top_n,
)

POSTPROCESS_PARALLEL_GROUPS = 256
POSTPROCESS_PARALLEL_GENES = 1024
POSTPROCESS_MAX_WORKERS = 8

if TYPE_CHECKING:
    from collections.abc import Iterable

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
        skip_empty_groups: bool = False,
    ) -> None:
        # Handle groups parameter
        if groups == "all" or groups is None:
            selected: list | None = None
        elif isinstance(groups, str | int):
            msg = "Specify a sequence of groups"
            raise ValueError(msg)
        else:
            selected = list(groups)
            if len(selected) > 0 and isinstance(selected[0], int):
                selected = [str(n) for n in selected]
            if reference != "rest" and reference not in set(selected):
                selected.append(reference)

        self.labels = pd.Series(adata.obs[groupby]).reset_index(drop=True)
        all_categories = self.labels.cat.categories

        if reference != "rest" and str(reference) not in {
            str(c) for c in all_categories
        }:
            cats = all_categories.tolist()
            msg = f"reference = {reference} needs to be one of groupby = {cats}."
            raise ValueError(msg)

        self.groups_order, self.group_codes, self.group_sizes = _select_groups(
            self.labels,
            selected,
            skip_empty_groups=skip_empty_groups,
            reference=reference,
        )

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
            self.ireference = int(np.where(self.groups_order == str(reference))[0][0])

        # Set up expm1 function based on log base
        self.is_log1p = "log1p" in adata.uns
        base = adata.uns.get("log1p", {}).get("base")
        if base is not None:
            self.expm1_func = lambda x: np.expm1(x * np.log(base))
        else:
            self.expm1_func = np.expm1

        # For basic stats
        self.comp_pts = comp_pts
        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.pts: np.ndarray | None = None
        self.means_rest: np.ndarray | None = None
        self.vars_rest: np.ndarray | None = None
        self.pts_rest: np.ndarray | None = None

        # Per-stat × per-group arrays.  results[stat] is either None (stat not
        # computed) or a list of 1-D numpy arrays in group_names order.  This
        # replaces the old DataFrame-based pipeline, which burned ~4 s per call
        # on wide workloads (1000+ groups) in pandas DataFrame + to_records.
        self.results: dict[str, list[np.ndarray] | None] = {
            "names": None,
            "scores": None,
            "pvals": None,
            "pvals_adj": None,
            "logfoldchanges": None,
        }
        self.group_names: list[str] = []
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
        Otherwise, sets flag for chunk-based computation during the wilcoxon loop.
        """
        n_genes = self.X.shape[1]

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
        result = agg.count_mean_var(dof=1)

        # Map Aggregate category order → selected groups order
        cat_names = list(self.labels.cat.categories)
        cat_to_idx = {str(name): i for i, name in enumerate(cat_names)}
        order = [cat_to_idx[str(name)] for name in self.groups_order]

        n = cp.asarray(self.group_sizes, dtype=cp.float64)[:, None]
        sums = result["sum"][order]
        sq_sums = result["sq_sum"][order]

        # Compute means and variances from raw sums (all on GPU)
        means = sums / n
        group_ss = sq_sums - n * means**2
        vars_ = cp.maximum(group_ss / cp.maximum(n - 1, 1), 0)

        if self.comp_pts:
            pts = result["count_nonzero"][order].astype(cp.float64) / n
        else:
            pts = None

        # Compute rest statistics if reference='rest'
        if self.ireference is None:
            n_rest = n.sum() - n
            means_rest = (sums.sum(axis=0) - sums) / n_rest
            rest_ss = (sq_sums.sum(axis=0) - sq_sums) - n_rest * means_rest**2
            vars_rest = cp.maximum(rest_ss / cp.maximum(n_rest - 1, 1), 0)

            self.means_rest = cp.asnumpy(means_rest)
            self.vars_rest = cp.asnumpy(vars_rest)

            if self.comp_pts:
                total_count = (pts * n).sum(axis=0)
                self.pts_rest = cp.asnumpy((total_count - pts * n) / n_rest)
            else:
                self.pts_rest = None
        else:
            self.means_rest = None
            self.vars_rest = None
            self.pts_rest = None

        # Transfer to CPU
        self.means = cp.asnumpy(means)
        self.vars = cp.asnumpy(vars_)
        self.pts = cp.asnumpy(pts) if pts is not None else None

    def t_test(
        self, method: Literal["t-test", "t-test_overestim_var"]
    ) -> list[tuple[int, NDArray, NDArray]]:
        """Compute t-test statistics using Welch's t-test."""
        from ._ttest import t_test

        return t_test(self, method)

    def wilcoxon(
        self,
        *,
        tie_correct: bool,
        use_continuity: bool = False,
    ) -> list[tuple[int, NDArray, NDArray]]:
        """Compute Wilcoxon rank-sum test statistics."""
        from ._wilcoxon import wilcoxon

        return wilcoxon(
            self,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
        )

    def wilcoxon_binned(
        self,
        *,
        tie_correct: bool = False,
        use_continuity: bool = False,
        n_bins: int | None = None,
        chunk_size: int | None = None,
        bin_range: Literal["log1p", "auto"] | None = None,
    ) -> list[tuple[int, NDArray, NDArray]]:
        """Histogram-based approximate Wilcoxon rank-sum test."""
        from ._wilcoxon_binned import wilcoxon_binned

        return wilcoxon_binned(
            self,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            n_bins=n_bins,
            chunk_size=chunk_size,
            bin_range=bin_range,
        )

    def logreg(self, **kwds) -> list[tuple[int, NDArray, None]]:
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
        use_continuity: bool = False,
        chunk_size: int | None = None,
        n_bins: int | None = None,
        bin_range: Literal["log1p", "auto"] | None = None,
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
            test_results = self.t_test(method)
        elif method == "wilcoxon":
            if isinstance(self.X, DaskArray):
                msg = "Wilcoxon test is not supported for Dask arrays. Please convert your data to CuPy arrays."
                raise ValueError(msg)
            test_results = self.wilcoxon(
                tie_correct=tie_correct,
                use_continuity=use_continuity,
            )
        elif method == "wilcoxon_binned":
            test_results = self.wilcoxon_binned(
                tie_correct=tie_correct,
                use_continuity=use_continuity,
                n_bins=n_bins,
                chunk_size=chunk_size,
                bin_range=bin_range,
            )
        elif method == "logreg":
            test_results = self.logreg(**kwds)
        else:
            assert_never(method)

        n_genes = self.X.shape[1]

        if not test_results:
            self.group_names = []
            return

        group_indices = np.array([gi for gi, _, _ in test_results], dtype=np.int64)
        self.group_names = [str(self.groups_order[gi]) for gi in group_indices]
        has_lfc = self.means is not None

        # Vectorised log-fold-change across all test groups — avoids 1948
        # individual numpy ops in the hot path (was ~1.5 s of Python
        # overhead on wide workloads).  Cast to the output dtype here so
        # _build_structured never needs a full-array astype pass.
        lfc_all: np.ndarray | None = None
        if has_lfc:
            mean_groups = self.means[group_indices]
            if self.ireference is None:
                mean_rests = self.means_rest[group_indices]
            else:
                mean_rests = self.means[self.ireference][None, :]
            foldchanges = (self.expm1_func(mean_groups) + EPS) / (
                self.expm1_func(mean_rests) + EPS
            )
            lfc_all = np.log2(foldchanges).astype(np.float32, copy=False)

        names_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        pvals_list: list[np.ndarray] = []
        pvals_adj_list: list[np.ndarray] = []
        lfc_list: list[np.ndarray] = []
        has_pvals = False
        # Pre-convert var_names to fixed-width unicode ONCE.  Without this,
        # per-group indexing returns object arrays, and np.stack(..., dtype='U')
        # ends up doing ~35 M object→string conversions inside the hot loop —
        # that was ~1.5 s on the 1948-group / 18k-gene workload.  Using the
        # target width directly turns the final stack into a pure memcpy.
        _vn = np.asarray(self.var_names)
        if _vn.dtype.kind == "U":
            var_names_arr = _vn
        else:
            max_len = max((len(str(n)) for n in _vn), default=1)
            var_names_arr = _vn.astype(f"U{max_len}")

        def _process_result(
            ti: int,
        ) -> tuple[
            int,
            np.ndarray | None,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]:
            _, scores, pvals = test_results[ti]
            if n_genes_user is not None:
                scores_sort = np.abs(scores) if rankby_abs else scores
                global_indices = _select_top_n(scores_sort, n_genes_user)
                names = var_names_arr[global_indices]
            else:
                global_indices = slice(None)
                names = None

            scores_out = scores[global_indices]
            pvals_out = None
            pvals_adj_out = None

            if pvals is not None:
                pvals_out = pvals[global_indices]
                if corr_method == "benjamini-hochberg":
                    pvals_adj = _benjamini_hochberg(pvals)
                elif corr_method == "bonferroni":
                    pvals_adj = np.minimum(pvals * n_genes, 1.0)
                pvals_adj_out = pvals_adj[global_indices]

            lfc_out = None
            if lfc_all is not None:
                lfc_out = lfc_all[ti][global_indices]

            return ti, names, scores_out, pvals_out, pvals_adj_out, lfc_out

        def _process_range(
            start: int, stop: int
        ) -> list[
            tuple[
                int,
                np.ndarray | None,
                np.ndarray,
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
            ]
        ]:
            return [_process_result(ti) for ti in range(start, stop)]

        n_results = len(test_results)
        use_parallel_post = (
            n_results >= POSTPROCESS_PARALLEL_GROUPS
            and n_genes >= POSTPROCESS_PARALLEL_GENES
        )
        if use_parallel_post:
            workers = min(POSTPROCESS_MAX_WORKERS, cpu_count() or 1, n_results)
            chunk = (n_results + workers - 1) // workers
            ranges = [
                (start, min(start + chunk, n_results))
                for start in range(0, n_results, chunk)
            ]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                processed_chunks = executor.map(lambda r: _process_range(*r), ranges)
                processed = [
                    item for chunk_out in processed_chunks for item in chunk_out
                ]
        else:
            processed = _process_range(0, n_results)

        for _, names, scores_out, pvals_out, pvals_adj_out, lfc_out in processed:
            if names is not None:
                names_list.append(names)
            scores_list.append(scores_out)
            if pvals_out is not None and pvals_adj_out is not None:
                has_pvals = True
                pvals_list.append(pvals_out)
                pvals_adj_list.append(pvals_adj_out)
            if lfc_out is not None:
                lfc_list.append(lfc_out)

        if self.group_names:
            self.results["scores"] = scores_list
            if n_genes_user is not None:
                self.results["names"] = names_list
            if has_pvals:
                self.results["pvals"] = pvals_list
                self.results["pvals_adj"] = pvals_adj_list
            if lfc_all is not None:
                self.results["logfoldchanges"] = lfc_list
