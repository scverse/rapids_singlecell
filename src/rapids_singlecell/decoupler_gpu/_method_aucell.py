from __future__ import annotations

import math

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from rapids_singlecell.decoupler_gpu._pre import (
    extract,
    filt_min_n,
    rename_net,
)
from rapids_singlecell.preprocessing._utils import _sparse_to_dense

# Kernel definition
reduce_sum_2D_kernel = cp.RawKernel(
    r"""
    extern "C" __global__
    void aucell_reduce(const long long* input, long long* output, long long R, long long C, long long max) {
        long long row = blockIdx.x *blockDim.x+ threadIdx.x ;
        if (row>=R){
            return;
        }
        long long sum = 0;
        long long prev = input[row*C+0];
        int switcher = 1;
        if (prev < max){
            switcher = 0;
            for(int i  = 1; i<C; i++){
                long long data = input[row*C+i];
                if(data<max) {
                    sum+= i *(data-prev);
                    prev = data;
                }
                else{
                    sum += i *(max-prev);
                    switcher = 1;
                    break;
                }
            }
        }
        if (switcher == 0){
            sum += C *(max-prev);

        }


        output[row] = sum;
    }
    """,
    "aucell_reduce",
)


def nb_aucell(row, net, *, starts=None, offsets=None, n_up=None, n_fsets=None):
    # Rank row
    row = cp.argsort(cp.argsort(-row), axis=1) + 1

    # Empty acts
    es = cp.zeros((row.shape[0], n_fsets), dtype=cp.float32)
    for j in range(n_fsets):
        es_inter = cp.zeros(row.shape[0], dtype=cp.int64)
        # Extract feature set
        srt = starts[j]
        off = offsets[j] + srt
        fset = net[srt:off]

        # Compute max AUC for fset
        k = min(fset.shape[0], n_up - 1)
        max_auc = int(k * (k - 1) / 2 + (n_up - k) * k)
        # Compute AUC
        x = row[:, fset]
        x = cp.sort(x, axis=1)

        threads_per_block = 32
        blocks_per_grid = math.ceil(x.shape[0] / threads_per_block)
        reduce_sum_2D_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (x, es_inter, x.shape[0], x.shape[1], n_up),
        )

        # Update acts matrix
        out = es_inter / max_auc
        es[:, j] = out.astype(cp.float32)
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
    offsets = cp.array(offsets)
    starts = cp.array(starts)
    net = cp.array(net)
    n_samples = mat.shape[0]
    batch_size = 5000
    n_batches = int(np.ceil(n_samples / batch_size))
    for i in tqdm(range(n_batches), disable=not verbose):
        # Subset batch
        srt, end = i * batch_size, i * batch_size + batch_size
        if isinstance(mat, csr_matrix):
            row = cp.array(mat[srt:end].toarray())
        elif isinstance(mat, cp_csr_matrix):
            row = _sparse_to_dense(mat[srt:end])
        else:
            row = cp.array(mat[srt:end])
        es[srt:end] = nb_aucell(
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
    msk = np.argsort(c)
    c = c[msk].astype("U")
    m = m[:, msk]
    # Set n_up
    if n_up is None:
        n_up = int(np.ceil(0.05 * len(c)))
    else:
        n_up = int(np.ceil(n_up))
        n_up = np.min([n_up, c.size])  # Limit n_up to max features
    if not 0 < n_up:
        raise ValueError("n_up needs to be a value higher than 0.")
    print(n_up)
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
