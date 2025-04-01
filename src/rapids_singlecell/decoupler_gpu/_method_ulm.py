from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from rapids_singlecell.preprocessing._utils import _sparse_to_dense

from ._pre import __stdtr, extract, filt_min_n, get_net_mat, match, rename_net


def mat_cov(A, b):
    return cp.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0] - 1)


def mat_cor(A, b):
    cov = mat_cov(A, b)
    ssd = cp.std(A, axis=0, ddof=1) * cp.std(b, axis=0, ddof=1).reshape(-1, 1)
    return cov / ssd


def t_val(r, df):
    return r * cp.sqrt(df / ((1.0 - r + 1.0e-16) * (1.0 + r + 1.0e-16)))


def ulm(mat, net, batch_size=10000, verbose=False):
    # Get dims
    n_samples = mat.shape[0]
    n_features, n_fsets = net.shape
    df = n_features - 2
    net = cp.array(net)
    if isinstance(mat, csr_matrix) or isinstance(mat, cp_csr_matrix):
        n_batches = int(np.ceil(n_samples / batch_size))
        es = cp.zeros((n_samples, n_fsets), dtype=np.float32)
        for i in tqdm(range(n_batches), disable=not verbose):
            # Subset batch
            srt, end = i * batch_size, i * batch_size + batch_size
            if isinstance(mat, cp_csr_matrix):
                batch = _sparse_to_dense(mat[srt:end], order="F").T
            else:
                batch = _sparse_to_dense(cp_csr_matrix(mat[srt:end]), order="F").T

            # Compute R for batch
            r = mat_cor(net, batch)

            # Compute t-value
            es[srt:end] = t_val(r, df)
    else:
        # Compute R value for all
        r = mat_cor(net, mat.T)

        # Compute t-value
        es = t_val(r, df)

    # Compute p-value
    pvals = (2 * (1 - __stdtr(df, cp.abs(es)))).get()
    es = es.get()

    return es, pvals


def run_ulm(
    mat: AnnData | pd.DataFrame | list,
    net: pd.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    weight: str = "weight",
    batch_size: int = 10000,
    min_n: int = 5,
    verbose: bool = False,
    use_raw: bool | None = None,
    layer: str | None = None,
    pre_load: bool | None = None,
):
    """
    Univariate Linear Model (ULM).

    ULM fits a linear model for each sample and regulator, where the observed molecular readouts in `mat` are the response
    variable and the regulator weights in `net` are the explanatory one. Target features with no associated weight are set to
    zero. The obtained t-value from the fitted model is the activity (`ulm_estimate`) of a given regulator.

    Parameters
    ----------
    mat
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net
        Network in long format.
    source
        Column name in net with source nodes.
    target
        Column name in net with target nodes.
    weight
        Column name in net with weights.
    batch_size
        Size of the samples to use for each batch. Increasing this will consume more memory but it will run faster.
    min_n
        Minimum of targets per source. If less, sources are removed.
    verbose
        Whether to show progress.
    use_raw
        Use raw attribute of mat.
    layer
        Layer to use in AnnData object.
    pre_load
        Whether to pre-load the data into memory. This can be faster for small datasets.

    Returns
    -------
    estimate : DataFrame
        ULM scores. Stored in `.obsm['ulm_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['ulm_pvals']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of genes
    m, r, c = extract(
        mat, use_raw=use_raw, layer=layer, verbose=verbose, pre_load=pre_load
    )

    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)

    # Match arrays
    net = match(c, targets, net)

    if verbose:
        print(
            f"Running ulm on mat with {m.shape[0]} samples and {len(c)} targets for {net.shape[1]} sources."
        )

    # Run ULM
    estimate, pvals = ulm(m, net, batch_size=batch_size, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = "ulm_estimate"
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = "ulm_pvals"

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
