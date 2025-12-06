from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import cupyx.scipy.special as cupyx_special
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from rapids_singlecell._compat import DaskArray, _meta_dense

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData


def _select_groups(labels, groups_order_subset="all"):
    groups_order = labels.cat.categories
    groups_masks = np.zeros(
        (len(labels.cat.categories), len(labels.cat.codes)), dtype=bool
    )
    for iname, name in enumerate(labels.cat.categories):
        # if the name is not found, fallback to index retrieval
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
            # fallback to index retrieval
            groups_ids = np.where(
                np.in1d(
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


def rank_genes_groups_logreg(
    adata: AnnData,
    groupby: str,
    *,
    groups: Literal["all"] | Iterable[str] = "all",
    use_raw: bool | None = None,
    reference: str = "rest",
    n_genes: int = None,
    layer: str = None,
    **kwds,
) -> None:
    """
    Rank genes for characterizing groups.

    Parameters
    ----------
        adata
            Annotated data matrix.
        groupby
            The key of the observations grouping to consider.
        groups
            Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
            shall be restricted, or `'all'` (default), for all groups.
        use_raw
            Use `raw` attribute of `adata` if present.
        reference
            If `'rest'`, compare each group to the union of the rest of the group.
            If a group identifier, compare with respect to this group.
        n_genes
            The number of genes that appear in the returned tables.
            Defaults to all genes.
        layer
            Key from `adata.layers` whose value will be used to perform tests on.

    Returns
    -------
        Updates `adata` with the following fields.

            **names** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
                Structured array to be indexed by group id storing the gene
                names. Ordered according to scores.

            **scores** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
                Structured array to be indexed by group id storing the z-score
                underlying the computation of a p-value for each gene for each
                group. Ordered according to scores.
    """
    #### Wherever we see "adata.obs[groupby], we should just replace w/ the groups"

    # for clarity, rename variable
    if groups == "all" or groups is None:
        groups_order = "all"
    elif isinstance(groups, str | int):
        raise ValueError("Specify a sequence of groups")
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != "rest" and reference not in set(groups_order):
            groups_order += [reference]
    labels = pd.Series(adata.obs[groupby]).reset_index(drop="True")
    if reference != "rest" and reference not in set(labels.cat.categories):
        cats = labels.cat.categories.tolist()
        raise ValueError(
            f"reference = {reference} needs to be one of groupby = {cats}."
        )

    groups_order, groups_masks = _select_groups(labels, groups_order)

    if layer and use_raw is True:
        raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
    elif layer:
        X = adata.layers[layer]
        var_names = adata.var_names
    elif use_raw is None and adata.raw:
        print("defaulting to using `.raw`")
        X = adata.raw.X
        var_names = adata.raw.var_names
    elif use_raw is True:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    # for clarity, rename variable
    n_genes_user = n_genes
    # make sure indices are not OoB in case there are less genes than n_genes
    if n_genes is None or n_genes_user > X.shape[1]:
        n_genes_user = X.shape[1]
    # in the following, n_genes is simply another name for the total number of genes

    n_groups = groups_masks.shape[0]
    ns = np.zeros(n_groups, dtype=int)
    for imask, mask in enumerate(groups_masks):
        ns[imask] = np.where(mask)[0].size
    if reference != "rest":
        reference = np.where(groups_order == reference)[0][0]
    reference_indices = cp.arange(X.shape[1], dtype=int)

    rankings_gene_scores = []
    rankings_gene_names = []

    # Perform LogReg

    # if reference is not set, then the groups listed will be compared to the rest
    # if reference is set, then the groups listed will be compared only to the other groups listed
    refname = reference

    reference = groups_order[0]
    if len(groups) == 1:
        raise Exception("Cannot perform logistic regression on a single cluster.")

    grouping_mask = labels.isin(pd.Series(groups_order))
    grouping = labels.loc[grouping_mask]

    X = X[grouping_mask.values, :]
    # Indexing with a series causes issues, possibly segfault

    grouping_logreg = grouping.cat.codes.to_numpy().astype(X.dtype)
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

    clf = LogisticRegression(**kwds)
    clf.fit(X, grouping_logreg)
    scores_all = cp.array(clf.coef_)
    if len(groups_order) == scores_all.shape[1]:
        scores_all = scores_all.T
    for igroup, _group in enumerate(groups_order):
        if len(groups_order) <= 2:  # binary logistic regression
            scores = scores_all[0]
        else:
            scores = scores_all[igroup]

        partition = cp.argpartition(scores, -n_genes_user)[-n_genes_user:]
        partial_indices = cp.argsort(scores[partition])[::-1]
        global_indices = reference_indices[partition][partial_indices]
        rankings_gene_scores.append(scores[global_indices].get())
        rankings_gene_names.append(var_names[global_indices.get()])
        if len(groups_order) <= 2:
            break

    groups_order_save = [str(g) for g in groups_order]
    if len(groups) == 2:
        groups_order_save = [groups_order_save[0]]

    scores = np.rec.fromarrays(
        list(rankings_gene_scores),
        dtype=[(rn, "float32") for rn in groups_order_save],
    )

    names = np.rec.fromarrays(
        list(rankings_gene_names),
        dtype=[(rn, "U50") for rn in groups_order_save],
    )
    adata.uns["rank_genes_groups"] = {}
    adata.uns["rank_genes_groups"]["params"] = {
        "groupby": groupby,
        "method": "logreg",
        "reference": refname,
        "use_raw": use_raw,
    }
    adata.uns["rank_genes_groups"]["scores"] = scores
    adata.uns["rank_genes_groups"]["names"] = names


EPS = 1e-9


def _choose_chunk_size(requested: int | None, n_obs: int, dtype_size: int = 8) -> int:
    if requested is not None:
        return int(requested)
    try:
        free_mem, _ = cp.cuda.runtime.memGetInfo()
    except cp.cuda.runtime.CUDARuntimeError:
        return 500
    bytes_per_gene = n_obs * dtype_size * 4
    if bytes_per_gene == 0:
        return 500
    max_genes = int(0.6 * free_mem / bytes_per_gene)
    return max(min(max_genes, 1000), 100)


def _average_ranks(matrix: cp.ndarray) -> cp.ndarray:
    ranks = cp.empty_like(matrix, dtype=cp.float64)
    for idx in range(matrix.shape[1]):
        column = matrix[:, idx]
        sorter = cp.argsort(column)
        sorted_column = column[sorter]
        unique = cp.concatenate(
            (cp.array([True]), sorted_column[1:] != sorted_column[:-1])
        )
        dense = cp.empty(column.size, dtype=cp.int64)
        dense[sorter] = cp.cumsum(unique)
        boundaries = cp.concatenate((cp.flatnonzero(unique), cp.array([unique.size])))
        ranks[:, idx] = 0.5 * (boundaries[dense] + boundaries[dense - 1] + 1.0)
    return ranks


def _tie_correction(ranks: cp.ndarray) -> cp.ndarray:
    correction = cp.ones(ranks.shape[1], dtype=cp.float64)
    for idx in range(ranks.shape[1]):
        column = cp.sort(ranks[:, idx])
        boundaries = cp.concatenate(
            (
                cp.array([0]),
                cp.flatnonzero(column[1:] != column[:-1]) + 1,
                cp.array([column.size]),
            )
        )
        differences = cp.diff(boundaries).astype(cp.float64)
        size = cp.float64(column.size)
        if size >= 2:
            correction[idx] = 1.0 - (differences**3 - differences).sum() / (
                size**3 - size
            )
    return correction


def rank_genes_groups_wilcoxon(
    adata: AnnData,
    groupby: str,
    *,
    groups: Literal["all"] | Iterable[str] = "all",
    use_raw: bool | None = None,
    reference: str = "rest",
    n_genes: int | None = None,
    tie_correct: bool = False,
    layer: str | None = None,
    chunk_size: int | None = None,
    corr_method: str = "benjamini-hochberg",
) -> None:
    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        msg = "corr_method must be either 'benjamini-hochberg' or 'bonferroni'."
        raise ValueError(msg)
    if reference != "rest":
        msg = "Only reference='rest' is currently supported for the GPU Wilcoxon test."
        raise NotImplementedError(msg)

    if groups == "all" or groups is None:
        groups_order = "all"
    elif isinstance(groups, str | int):
        raise ValueError("Specify a sequence of groups")
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]

    labels = pd.Series(adata.obs[groupby]).reset_index(drop="True")
    groups_order, groups_masks = _select_groups(labels, groups_order)

    group_sizes = groups_masks.sum(axis=1).astype(np.int64)
    n_cells = labels.shape[0]
    for name, size in zip(groups_order, group_sizes, strict=False):
        rest = n_cells - size
        if size <= 25 or rest <= 25:
            warnings.warn(
                f"Group {name} has size {size} (rest {rest}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
            )

    if layer and use_raw is True:
        raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
    elif layer:
        X = adata.layers[layer]
        var_names = adata.var_names
    elif use_raw is None and adata.raw:
        print("defaulting to using `.raw`")
        X = adata.raw.X
        var_names = adata.raw.var_names
    elif use_raw is True:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    if hasattr(X, "toarray"):
        X = X.toarray()

    n_cells, n_total_genes = X.shape
    n_top = n_total_genes if n_genes is None else min(n_genes, n_total_genes)

    group_matrix = cp.asarray(groups_masks.T, dtype=cp.float64)
    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev

    base = adata.uns.get("log1p", {}).get("base")
    if base is not None:
        log_expm1 = lambda arr: cp.expm1(arr * cp.log(base))
    else:
        log_expm1 = cp.expm1

    chunk_width = _choose_chunk_size(chunk_size, n_cells)
    group_keys = [str(key) for key in groups_order]

    scores: dict[str, list[np.ndarray]] = {key: [] for key in group_keys}
    logfc: dict[str, list[np.ndarray]] = {key: [] for key in group_keys}
    pvals: dict[str, list[np.ndarray]] = {key: [] for key in group_keys}
    gene_indices: dict[str, list[np.ndarray]] = {key: [] for key in group_keys}

    for start in range(0, n_total_genes, chunk_width):
        stop = min(start + chunk_width, n_total_genes)
        block = cp.asarray(X[:, start:stop], dtype=cp.float64)
        ranks = _average_ranks(block)
        if tie_correct:
            tie_corr = _tie_correction(ranks)
        else:
            tie_corr = cp.ones(ranks.shape[1], dtype=cp.float64)

        rank_sums = group_matrix.T @ ranks
        expected = group_sizes_dev[:, None] * (n_cells + 1) / 2.0
        variance = tie_corr[None, :] * group_sizes_dev[:, None] * rest_sizes[:, None]
        variance *= (n_cells + 1) / 12.0
        std = cp.sqrt(variance)
        z = (rank_sums - expected) / std
        cp.nan_to_num(z, copy=False)
        p_values = 2.0 * (1.0 - cupyx_special.ndtr(cp.abs(z)))

        group_sums = group_matrix.T @ block
        group_means = group_sums / group_sizes_dev[:, None]
        total_mean = cp.mean(block, axis=0)
        rest_sum = total_mean * n_cells - group_means * group_sizes_dev[:, None]
        mean_rest = rest_sum / rest_sizes[:, None]
        numerator = log_expm1(group_means) + EPS
        denominator = log_expm1(mean_rest) + EPS
        log_fold = cp.log2(numerator / denominator)

        indices = np.arange(start, stop, dtype=int)
        z_host = z.get()
        p_host = p_values.get()
        logfc_host = log_fold.get()

        for idx, key in enumerate(group_keys):
            scores[key].append(z_host[idx])
            logfc[key].append(logfc_host[idx])
            pvals[key].append(p_host[idx])
            gene_indices[key].append(indices)

    var_array = np.asarray(var_names)
    structured = {}
    for key in group_keys:
        all_scores = (
            np.concatenate(scores[key]) if scores[key] else np.empty(0, dtype=float)
        )
        all_logfc = (
            np.concatenate(logfc[key]) if logfc[key] else np.empty(0, dtype=float)
        )
        all_pvals = (
            np.concatenate(pvals[key]) if pvals[key] else np.empty(0, dtype=float)
        )
        all_genes = (
            np.concatenate(gene_indices[key])
            if gene_indices[key]
            else np.empty(0, dtype=int)
        )

        clean = np.array(all_pvals, copy=True)
        clean[np.isnan(clean)] = 1.0
        if clean.size and corr_method == "benjamini-hochberg":
            _, adjusted, _, _ = multipletests(clean, alpha=0.05, method="fdr_bh")
        elif clean.size:
            adjusted = np.minimum(clean * n_total_genes, 1.0)
        else:
            adjusted = np.array([], dtype=float)

        if all_scores.size:
            order = np.argsort(all_scores)[::-1]
        else:
            order = np.empty(0, dtype=int)
        keep = order[: min(n_top, order.size)]

        structured[key] = {
            "scores": all_scores[keep].astype(np.float32, copy=False),
            "logfc": all_logfc[keep].astype(np.float32, copy=False),
            "pvals": clean[keep].astype(np.float64, copy=False),
            "pvals_adj": adjusted[keep].astype(np.float64, copy=False),
            "names": var_array[all_genes[keep]].astype("U50", copy=False),
        }

    dtype_scores = [(key, "float32") for key in group_keys]
    dtype_names = [(key, "U50") for key in group_keys]
    dtype_logfc = [(key, "float32") for key in group_keys]
    dtype_pvals = [(key, "float64") for key in group_keys]

    adata.uns["rank_genes_groups"] = {
        "params": {
            "groupby": groupby,
            "method": "wilcoxon",
            "reference": reference,
            "use_raw": use_raw,
            "tie_correct": tie_correct,
            "layer": layer,
            "corr_method": corr_method,
        },
        "scores": np.rec.fromarrays(
            [structured[key]["scores"] for key in group_keys],
            dtype=dtype_scores,
        ),
        "names": np.rec.fromarrays(
            [structured[key]["names"] for key in group_keys],
            dtype=dtype_names,
        ),
        "logfoldchanges": np.rec.fromarrays(
            [structured[key]["logfc"] for key in group_keys],
            dtype=dtype_logfc,
        ),
        "pvals": np.rec.fromarrays(
            [structured[key]["pvals"] for key in group_keys],
            dtype=dtype_pvals,
        ),
        "pvals_adj": np.rec.fromarrays(
            [structured[key]["pvals_adj"] for key in group_keys],
            dtype=dtype_pvals,
        ),
    }
