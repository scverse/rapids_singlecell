from __future__ import annotations

import sys
import warnings
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._core import _RankGenes
from ._utils import NoTestGroupsError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from numpy.typing import NDArray

type _CorrMethod = Literal["benjamini-hochberg", "bonferroni"]
type _Method = Literal[
    "logreg", "t-test", "t-test_overestim_var", "wilcoxon", "wilcoxon_binned"
]


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

    Expects logarithmized data.

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
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    chunk_size
        Number of genes to process at once for `'wilcoxon_binned'`.
        The default is sized dynamically based on ``n_groups`` and
        ``n_bins`` to keep histogram memory stable.
        Ignored for other methods.
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
        ``'auto'`` computes the actual data range. Use this for z-scored
        or unnormalized data.
    skip_empty_groups
        If ``True``, silently drop groups with fewer than 2 cells (issuing
        a ``RuntimeWarning``) instead of raising a ``ValueError``.  Useful
        when iterating over data subsets (e.g. per cell type) where some
        categorical levels may be empty.  If no test groups remain after
        filtering (only the reference has >=2 cells), the call returns
        without updating ``adata.uns``.  The reference group is never
        dropped — if it has <2 cells the call still fails.
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

    try:
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
    except NoTestGroupsError as e:
        # skip_empty_groups=True contract: no test groups left → no-op.
        # Do not write to adata.uns so downstream loops can detect the
        # missing key and skip this subset.
        warnings.warn(
            f"rank_genes_groups: skipping — {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

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
        chunk_size=chunk_size,
        n_bins=n_bins,
        bin_range=bin_range,
        **kwds,
    )

    # Use a U-width tight to the actual gene names rather than the scanpy
    # default of U50.  For a 1948-group × 18k-gene workload this cuts the
    # names structured array from ~7 GB → ~3 GB.  Must match the width that
    # compute_statistics used when converting var_names (see _core.py), so
    # the final stack → structured-array view is a pure memcpy.
    _vn = np.asarray(test_obj.var_names)
    if _vn.dtype.kind == "U":
        max_name_len = _vn.dtype.itemsize // 4
    elif len(_vn):
        max_name_len = max(len(str(n)) for n in _vn)
    else:
        max_name_len = 50
    names_dtype = f"U{max(max_name_len, 1)}"
    dtypes = {
        "names": names_dtype,
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

    # Assemble scanpy-compatible structured arrays directly from per-group
    # arrays, without going through a wide pandas DataFrame + to_records —
    # that pipeline was ~4 s of pure Python overhead on workloads with
    # thousands of groups.
    group_names = test_obj.group_names
    if group_names:
        for stat, per_group_arrays in test_obj.results.items():
            if per_group_arrays is None or stat not in dtypes:
                continue
            adata.uns[key_added][stat] = _build_structured(
                group_names, per_group_arrays, dtypes[stat]
            )


def _build_structured(
    group_names: list[str],
    per_group_arrays: list[np.ndarray],
    field_dtype: str,
) -> np.ndarray:
    """Build a scanpy-style structured array with one field per group.

    Equivalent to assigning ``sa[name] = arr`` in a loop, but ~20× faster
    on wide workloads (1000+ groups): fills a contiguous 2-D (n, n_groups)
    buffer row-by-row and reinterprets it as a structured view, so the
    bulk write pattern matches the structured-array memory layout.
    """
    n = per_group_arrays[0].shape[0]
    # Stack directly into the target dtype — np.stack with `dtype=` avoids
    # the separate astype pass.  axis=1 produces (n, n_groups) C-contig
    # exactly matching the structured memory layout below, so .view() is
    # zero-copy.  Use 'unsafe' casting for string targets (object→U50).
    if np.dtype(field_dtype).kind == "U":
        stacked = np.stack(
            per_group_arrays, axis=1, casting="unsafe", dtype=field_dtype
        )
    else:
        stacked = np.stack(per_group_arrays, axis=1, dtype=field_dtype)
    dtype = np.dtype([(name, field_dtype) for name in group_names])
    return stacked.view(dtype).reshape(n)


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
