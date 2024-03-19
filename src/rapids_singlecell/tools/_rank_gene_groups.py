from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal

import cupy as cp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
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
    use_raw: bool = None,
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

            **pvals** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
                p-values.

            **pvals_adj** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
                Corrected p-values.
    """
    #### Wherever we see "adata.obs[groupby], we should just replace w/ the groups"

    # for clarity, rename variable
    if groups == "all" or groups is None:
        groups_order = "all"
    elif isinstance(groups, (str, int)):
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
    from cuml.linear_model import LogisticRegression

    reference = groups_order[0]
    if len(groups) == 1:
        raise Exception("Cannot perform logistic regression on a single cluster.")

    grouping_mask = labels.isin(pd.Series(groups_order))
    grouping = labels.loc[grouping_mask]

    X = X[grouping_mask.values, :]
    # Indexing with a series causes issues, possibly segfault

    grouping_logreg = grouping.cat.codes.to_numpy().astype("float32")
    uniques = np.unique(grouping_logreg)
    for idx, cat in enumerate(uniques):
        grouping_logreg[np.where(grouping_logreg == cat)] = idx

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
