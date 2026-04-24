from __future__ import annotations

import sys
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._core import _RankGenes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from numpy.typing import NDArray

type _CorrMethod = Literal["benjamini-hochberg", "bonferroni"]
type _Method = Literal[
    "logreg", "t-test", "t-test_overestim_var", "wilcoxon", "wilcoxon_binned"
]


class _LazyRankGenesColumn:
    def __init__(
        self,
        values: np.ndarray | None = None,
        *,
        var_names: np.ndarray | None = None,
        gene_indices: np.ndarray | None = None,
        dtype: str | np.dtype,
    ) -> None:
        self._values = values
        self._var_names = var_names
        self._gene_indices = gene_indices
        self._dtype = np.dtype(dtype)

    def __len__(self) -> int:
        if self._values is not None:
            return int(self._values.shape[0])
        return int(self._gene_indices.shape[0])

    def __getitem__(self, key):
        if self._values is not None:
            return self._values[key]
        return self._var_names[self._gene_indices[key]]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if self._values is not None:
            arr = np.asarray(self._values, dtype=self._dtype)
        else:
            arr = np.asarray(self._var_names[self._gene_indices], dtype=self._dtype)
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        if copy:
            arr = arr.copy()
        return arr


class _LazyRankGenesRecords(dict):
    def __init__(
        self, group_names: np.ndarray, columns: dict[str, object], dtype: str | np.dtype
    ) -> None:
        super().__init__(columns)
        self._group_names = tuple(str(name) for name in group_names)
        self._dtype = np.dtype([(name, np.dtype(dtype)) for name in self._group_names])

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        return np.asarray(self)[key]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = np.empty(len(next(iter(self.values()))) if self else 0, dtype=self._dtype)
        for name in self._group_names:
            out[name] = np.asarray(super().__getitem__(name))
        if dtype is not None:
            out = np.asarray(out, dtype=dtype)
        if copy:
            out = out.copy()
        return out

    def copy(self) -> np.ndarray:
        return np.asarray(self).copy()


def _array_result_to_lazy_records(
    arrays: dict[str, object], field: str, dtype: str | np.dtype
) -> _LazyRankGenesRecords:
    group_names = arrays["group_names"]
    values = arrays[field]
    columns = {
        str(group_name): _LazyRankGenesColumn(values[row], dtype=dtype)
        for row, group_name in enumerate(group_names)
    }
    return _LazyRankGenesRecords(group_names, columns, dtype)


def _array_result_to_lazy_names(arrays: dict[str, object]) -> _LazyRankGenesRecords:
    group_names = arrays["group_names"]
    var_names = arrays["var_names"]
    gene_indices = arrays["gene_indices"]
    columns = {
        str(group_name): _LazyRankGenesColumn(
            var_names=var_names, gene_indices=gene_indices[row], dtype=object
        )
        for row, group_name in enumerate(group_names)
    }
    return _LazyRankGenesRecords(group_names, columns, object)


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
    use_continuity: bool = False,
    return_u_values: bool = False,
    layer: str | None = None,
    chunk_size: int | None = None,
    pre_load: bool = False,
    n_bins: int | None = None,
    bin_range: Literal["log1p", "auto"] | None = None,
    skip_empty_groups: bool = False,
    **kwds,
) -> None:
    """
    Rank genes for characterizing groups using GPU acceleration.

    Expects nonnegative expression data. Log1p/log-normalized data is expected
    for biologically meaningful log fold changes; sparse inputs with explicit
    negative values are rejected.

    .. note::
        **Dask support:** `'t-test'`, `'t-test_overestim_var'`, and
        `'wilcoxon_binned'` support Dask arrays. The `'wilcoxon'` and
        `'logreg'` methods do not support Dask arrays.

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
        `'wilcoxon_binned'` uses histogram-based approximate Wilcoxon rank-sum
        (faster for large datasets, supports Dask arrays),
        `'logreg'` uses logistic regression.
    corr_method
        p-value correction method. Used only for `'t-test'`,
        `'t-test_overestim_var'`, `'wilcoxon'`, and `'wilcoxon_binned'`.
    tie_correct
        Use tie correction for `'wilcoxon'` and `'wilcoxon_binned'` scores.
        Adjusts the variance of the rank-sum statistic for tied values.
        For `'wilcoxon_binned'`, each histogram bin acts as a tie group
        and the correction is derived from the bin counts.
    use_continuity
        Apply continuity correction to `'wilcoxon'` and `'wilcoxon_binned'`
        z-scores. Subtracts 0.5 from ``|R - E[R]|`` before dividing by the
        standard deviation, matching :func:`scipy.stats.mannwhitneyu`
        default behavior.
    return_u_values
        For `'wilcoxon'`, store Mann-Whitney U statistics in `scores` instead
        of z-scores. P-values are still computed from the z-score normal
        approximation using the selected tie and continuity settings.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    chunk_size
        Number of genes to process at once for `'wilcoxon'` and
        `'wilcoxon_binned'`. Default is 128 for `'wilcoxon'`. For
        `'wilcoxon_binned'` the default is sized dynamically based on
        ``n_groups`` and ``n_bins`` to keep histogram memory stable.
    pre_load
        Pre-load the data into GPU memory. Used only for `'wilcoxon'`.
    n_bins
        Number of histogram bins for `'wilcoxon_binned'`. Higher values give
        a better approximation at slightly increased cost. Default is 1000
        for in-memory arrays and 200 for Dask arrays.
    bin_range
        How to determine the histogram bin range for `'wilcoxon_binned'`.
        ``None`` (default) uses ``'auto'`` for in-memory arrays and
        ``'log1p'`` for Dask arrays (to avoid a costly data scan).
        ``'log1p'`` uses a fixed [0, 15] range suitable for most log1p-normalized data.
        ``'auto'`` computes the actual data range. Use this for nonnegative
        expression data outside the fixed log1p range.
    skip_empty_groups
        Skip selected groups with fewer than two observations after filtering.
        This is useful for perturbation workflows where a per-cell-type slice
        keeps categories that are empty or singleton in that slice.
    **kwds
        Additional arguments passed to the method. For `'logreg'`, these are
        passed to :class:`cuml.linear_model.LogisticRegression`.

    Returns
    -------
    Updates `adata` with the following fields. Rank result fields are lazy
    Scanpy-compatible record objects: group fields can be indexed like
    structured arrays, while full structured arrays are materialized only when
    requested through NumPy conversion or `.copy()`.

    `adata.uns['rank_genes_groups' | key_added]['names']`
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['scores']`
        Structured array to be indexed by group id storing the z-score
        underlying the computation of a p-value for each gene for each
        group, or the Mann-Whitney U statistic when
        `return_u_values=True`. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['logfoldchanges']`
        Structured array to be indexed by group id storing the log2
        fold change for each gene for each group.
    `adata.uns['rank_genes_groups' | key_added]['pvals']`
        p-values. Only for `'t-test'`, `'t-test_overestim_var'`,
        `'wilcoxon'`, and `'wilcoxon_binned'`.
    `adata.uns['rank_genes_groups' | key_added]['pvals_adj']`
        Corrected p-values. Only for `'t-test'`, `'t-test_overestim_var'`,
        `'wilcoxon'`, and `'wilcoxon_binned'`.
    `adata.uns['rank_genes_groups' | key_added]['pts']`
        Fraction of cells expressing genes per group. Only if `pts=True`.
    `adata.uns['rank_genes_groups' | key_added]['pts_rest']`
        Fraction of cells expressing genes in rest. Only if `pts=True` and `reference='rest'`.
    """
    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        msg = "corr_method must be either 'benjamini-hochberg' or 'bonferroni'."
        raise ValueError(msg)

    if "return_format" in kwds:
        msg = (
            "return_format has been removed; rank_genes_groups always writes "
            "lazy Scanpy-compatible results to adata.uns."
        )
        raise TypeError(msg)

    if method is None:
        method = "t-test"

    if method not in {
        "logreg",
        "t-test",
        "t-test_overestim_var",
        "wilcoxon",
        "wilcoxon_binned",
    }:
        msg = (
            "method must be one of 'logreg', 't-test', 't-test_overestim_var', "
            f"'wilcoxon', 'wilcoxon_binned'. Got {method!r}."
        )
        raise ValueError(msg)

    if return_u_values and method != "wilcoxon":
        msg = "return_u_values is only supported for method='wilcoxon'."
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
        skip_empty_groups=skip_empty_groups,
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
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        chunk_size=chunk_size,
        n_bins=n_bins,
        bin_range=bin_range,
        **kwds,
    )

    params = {
        "groupby": groupby,
        "reference": reference,
        "method": method,
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method,
    }
    if method == "wilcoxon":
        params["tie_correct"] = tie_correct
        params["return_u_values"] = return_u_values

    arrays = test_obj.stats_arrays or {}
    adata.uns[key_added] = {"params": params}
    if arrays and len(arrays.get("group_names", ())) > 0:
        adata.uns[key_added]["names"] = _array_result_to_lazy_names(arrays)
        for col, dtype in {
            "scores": "float32",
            "logfoldchanges": "float32",
            "pvals": "float64",
            "pvals_adj": "float64",
        }.items():
            if col in arrays:
                values = arrays[col]
                if hasattr(values, "dtype"):
                    dtype = values.dtype
                adata.uns[key_added][col] = _array_result_to_lazy_records(
                    arrays, col, dtype
                )

    if test_obj.pts is not None:
        groups_names = [str(name) for name in test_obj.groups_order]
        adata.uns[key_added]["pts"] = pd.DataFrame(
            test_obj.pts.T, index=test_obj.var_names, columns=groups_names
        )
    if test_obj.pts_rest is not None:
        adata.uns[key_added]["pts_rest"] = pd.DataFrame(
            test_obj.pts_rest.T, index=test_obj.var_names, columns=groups_names
        )

    return None


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
    return rank_genes_groups(
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
