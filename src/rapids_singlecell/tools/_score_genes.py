from __future__ import annotations

import warnings as Warning
from typing import TYPE_CHECKING, Sequence

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import issparse

from rapids_singlecell.get import _get_obs_rep
from rapids_singlecell.preprocessing._utils import _check_gpu_X, _check_use_raw

if TYPE_CHECKING:
    from anndata import AnnData


def score_genes(
    adata: AnnData,
    gene_list: Sequence[str] | pd.Index[str],
    *,
    ctrl_as_ref: bool = True,
    ctrl_size: int = 50,
    gene_pool: Sequence[str] | pd.Index[str] | None = None,
    n_bins: int = 25,
    score_name: str = "score",
    random_state: int | None = 0,
    copy: bool = False,
    use_raw: bool | None = None,
    layer: str | None = None,
) -> AnnData | None:
    """\
    Score a set of genes :cite:p:`Satija2015`.

    The score is the average expression of a set of genes subtracted with the
    average expression of a reference set of genes. The reference set is
    randomly sampled from the `gene_pool` for each binned expression value.

    This reproduces the approach in Seurat :cite:p:`Satija2015` and has been implemented
    for Scanpy by Davide Cittaro.

    Parameters
    ----------
    adata
        The annotated data matrix.
    gene_list
        The list of gene names used for score calculation.
    ctrl_as_ref
        Allow the algorithm to use the control genes as reference.
        Will be changed to `False` in scanpy 2.0.
    ctrl_size
        Number of reference genes to be sampled from each bin. If `len(gene_list)` is not too
        low, you can set `ctrl_size=len(gene_list)`.
    gene_pool
        Genes for sampling the reference set. Default is all genes.
    n_bins
        Number of expression level bins for sampling.
    score_name
        Name of the field to be added in `.obs`.
    random_state
        The random seed for sampling.
    copy
        Copy `adata` or modify it inplace.
    use_raw
        Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following field:

    `adata.obs[score_name]` : :class:`numpy.ndarray` (dtype `float`)
        Scores of each cell.
    """
    adata = adata.copy() if copy else adata
    use_raw = _check_use_raw(adata, use_raw, layer=layer)
    X = _get_obs_rep(adata, layer=layer, use_raw=use_raw)
    if use_raw:
        if issparse(X):
            if X.format == "csc":
                X = csc_matrix(X)
            elif X.format == "csr":
                X = csr_matrix(X)
            else:
                raise ValueError("X is not in a valid sparse format")
        else:
            X = cp.array(X)
    X = _check_gpu_X(X)

    if random_state is not None:
        np.random.seed(random_state)

    var_names = adata.raw.var_names if use_raw else adata.var_names
    gene_list = pd.Index([gene_list] if isinstance(gene_list, str) else gene_list)
    genes_to_ignore = gene_list.difference(var_names, sort=False)  # first get missing
    gene_list = gene_list.intersection(var_names)  # then restrict to present
    if len(genes_to_ignore) > 0:
        Warning.warning(f"genes are not in var_names and ignored: {genes_to_ignore}")
    if len(gene_list) == 0:
        raise ValueError("No valid genes were passed for scoring.")

    if gene_pool is None:
        gene_pool = var_names.astype("string")
    else:
        gene_pool = pd.Index(gene_pool, dtype="string").intersection(var_names)
    if len(gene_pool) == 0:
        raise ValueError("No valid genes were passed for reference set.")

    # average expression of genes
    obs_avg = pd.Series(_nan_means((gene_pool), axis=0), index=gene_pool)
    # Sometimes (and I don’t know how) missing data may be there, with NaNs for missing entries
    obs_avg = obs_avg[np.isfinite(obs_avg)]

    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    keep_ctrl_in_obs_cut = False if ctrl_as_ref else obs_cut.index.isin(gene_list)

    # now pick `ctrl_size` genes from every cut
    control_genes = pd.Index([], dtype="string")
    for cut in np.unique(obs_cut.loc[gene_list]):
        r_genes: pd.Index[str] = obs_cut[(obs_cut == cut) & ~keep_ctrl_in_obs_cut].index
        """
        if len(r_genes) == 0:
            msg = (
                f"No control genes for {cut=}. You might want to increase "
                f"gene_pool size (current size: {len(gene_pool)})"
            )
            logg.warning(msg)
        """
        if ctrl_size < len(r_genes):
            r_genes = r_genes.to_series().sample(ctrl_size).index
        if ctrl_as_ref:  # otherwise `r_genes` is already filtered
            r_genes = r_genes.difference(gene_list)
        control_genes = control_genes.union(r_genes)

    if len(control_genes) == 0:
        msg = "No control genes found in any cut."
        if ctrl_as_ref:
            msg += " Try setting `ctrl_as_ref=False`."
        raise RuntimeError(msg)

    means_list, means_control = (
        _nan_means((genes), axis=1, dtype="float64")
        for genes in (gene_list, control_genes)
    )
    score = means_list - means_control

    adata.obs[score_name] = pd.Series(
        np.array(score).ravel(), index=adata.obs_names, dtype="float64"
    )

    return adata if copy else None


def _nan_means(X, axis, dtype="float32"):
    pass
