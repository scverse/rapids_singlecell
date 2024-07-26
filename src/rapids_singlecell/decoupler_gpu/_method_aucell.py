from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from tqdm.auto import tqdm

from rapids_singlecell.decoupler_gpu._pre import (
    extract,
    filt_min_n,
    rename_net,
)


def nb_aucell(row, net, *, starts, offsets, n_up, n_fsets):
    row = cp.argsort(cp.argsort(-row)) + 1
    es = cp.zeros(n_fsets, dtype=cp.float32)

    # For each feature set
    for j in range(n_fsets):
        # Extract feature set
        srt = starts[j]
        off = offsets[j] + srt
        fset = net[srt:off]

        # Compute max AUC for fset
        k = min(fset.shape[0], n_up - 1)
        max_auc = k * (k - 1) / 2 + (n_up - k) * k

        # Compute AUC
        x = row[fset]
        x = cp.sort(x[x < n_up])
        y = cp.arange(x.shape[0]) + 1
        x = cp.append(x, n_up)
        # Update acts matrix
        es[j] = cp.sum(cp.diff(x) * y) / max_auc

    return es.get()


def aucell(mat, net, n_up, verbose):
    # Get dims
    n_samples = mat.shape[0]
    n_fsets = net.shape[0]

    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    net = np.concatenate(net.values)
    # Define starts to subset offsets
    starts = np.zeros(n_fsets, dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]

    es = np.zeros((n_samples, n_fsets), dtype=np.float32)
    for i in tqdm(range(mat.shape[0]), disable=not verbose):
        if isinstance(mat, cp_csr_matrix):
            row = mat[i].toarray().flatten()
        else:
            row = cp.array(mat[i])

        # Compute AUC per row
        es[i] = nb_aucell(
            row, net, starts=starts, offsets=offsets, n_up=n_up, n_fsets=n_fsets
        )

    return es


def run_aucell(
    mat,
    net,
    *,
    source="source",
    target="target",
    n_up=None,
    min_n=5,
    seed=42,
    verbose=False,
    use_raw=True,
    layer: str | None = None,
    pre_load: bool | None = None,
):
    """
    AUCell.

    AUCell uses the Area Under the Curve (AUC) to calculate whether a set of targets is enriched within
    the molecular readouts of each sample. To do so, AUCell first ranks the molecular features of each sample from highest to
    lowest value, resolving ties randomly. Then, an AUC can be calculated using by default the top 5% molecular features in the
    ranking. Therefore, this metric, `aucell_estimate`, represents the proportion of abundant molecular features in the target
    set, and their relative abundance value compared to the other features within the sample.


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
    n_up
        Number of top ranked features to select as observed features. If not specified it will be equal to the 5% of the number of features.
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

    # Set n_up
    if n_up is None:
        n_up = int(np.ceil(0.05 * len(c)))
    else:
        n_up = int(np.ceil(n_up))
        n_up = np.min([n_up, c.size])  # Limit n_up to max features
    if not 0 < n_up:
        raise ValueError("n_up needs to be a value higher than 0.")

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)
    net = filt_min_n(c, net, min_n=min_n)

    rng = np.random.default_rng(seed=seed)
    idx = np.arange(m.shape[1])
    rng.shuffle(idx)
    m, c = m[:, idx], c[idx]

    # Transform targets to indxs
    table = {name: i for i, name in enumerate(c)}
    net["target"] = [table[target] for target in net["target"]]
    net = net.groupby("source", observed=True)["target"].apply(
        lambda x: np.array(x, dtype=np.int64)
    )

    if verbose:
        print(
            f"Running aucell on mat with {m.shape[0]} samples and {len(c)} targets for {len(net)} sources."
        )

    # Run AUCell
    estimate = aucell(m, net, n_up, verbose)
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = "aucell_estimate"

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate

    else:
        return estimate
