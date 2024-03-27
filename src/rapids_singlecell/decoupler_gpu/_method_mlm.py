from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from decoupler.pre import extract, filt_min_n, get_net_mat, match, rename_net
from scipy import stats
from scipy.sparse import csr_matrix
from tqdm import tqdm


def fit_mlm(X, y, inv, df):
    X = cp.ascontiguousarray(X)
    y.shape[1]
    X.shape[1]
    coef, sse, _, _ = cp.linalg.lstsq(X, y, rcond=-1)
    if len(sse) == 0:
        raise ValueError(
            """Couldn't fit a multivariate linear model. This can happen because there are more sources
        (covariates) than unique targets (samples), or because the network\'s matrix rank is smaller than the number of
        sources."""
        )
    sse = sse / df
    inv = cp.diag(inv)
    sse = cp.reshape(sse, (sse.shape[0], 1))
    inv = cp.reshape(inv, (1, inv.shape[0]))
    se = cp.sqrt(sse * inv)
    t = coef.T / se
    return t.astype(np.float32).get()


def mlm(mat, net, batch_size=10000, verbose=False):
    # Get number of batches
    n_samples = mat.shape[0]
    n_features, n_fsets = net.shape
    n_batches = int(np.ceil(n_samples / batch_size))

    # Add intercept to network
    net = cp.column_stack((cp.ones((n_features,), dtype=np.float32), cp.array(net)))

    # Compute inv and df for lm
    inv = cp.linalg.inv(cp.dot(net.T, net))
    df = n_features - n_fsets - 1

    if isinstance(mat, csr_matrix):
        # Init empty acts
        n_batches = int(np.ceil(n_samples / batch_size))
        es = np.zeros((n_samples, n_fsets), dtype=np.float32)
        for i in tqdm(range(n_batches), disable=not verbose):
            # Subset batch
            srt, end = i * batch_size, i * batch_size + batch_size
            y = mat[srt:end].A.T

            # Compute MLM for batch
            es[srt:end] = fit_mlm(net, cp.array(y), inv, df)[:, 1:]
    else:
        # Compute MLM for all
        es = fit_mlm(net, cp.array(mat.T), inv, df)[:, 1:]

    # Get p-values
    pvals = 2 * (1 - stats.t.cdf(np.abs(es), df))

    return es, pvals


def run_mlm(
    mat: AnnData | pd.DataFrame | list,
    net: pd.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    weight: str = "weight",
    batch_size: int = 10000,
    min_n: int = 5,
    verbose: bool = False,
    use_raw: bool = True,
) -> tuple | None:
    """
    Multivariate Linear Model (MLM).
    MLM fits a multivariate linear model for each sample, where the observed molecular readouts in `mat` are the response
    variable and the regulator weights in `net` are the covariates. Target features with no associated weight are set to
    zero. The obtained t-values from the fitted model are the activities (`mlm_estimate`) of the regulators in `net`.

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
            Use raw attribute of mat if present.

    Returns
    -------
        Updates `adata` with the following fields.

            **estimate** : DataFrame
                MLM scores. Stored in `.obsm['mlm_estimate']` if `mat` is AnnData.
            **pvals** : DataFrame
                Obtained p-values. Stored in `.obsm['mlm_pvals']` if `mat` is AnnData.
    """
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)

    # Match arrays
    net = match(c, targets, net)

    if verbose:
        print(
            f"Running mlm on mat with {m.shape[0]} samples and {len(c)} targets for {net.shape[1]} sources."
        )

    # Run MLM
    estimate, pvals = mlm(m, net, batch_size=batch_size, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = "mlm_estimate"
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = "mlm_pvals"

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
