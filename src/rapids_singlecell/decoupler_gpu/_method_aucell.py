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
    void aucell_reduce(const int* input, int* output, int R, int C, int max) {
        int row = blockIdx.x *blockDim.x+ threadIdx.x ;
        if (row>=R){
            return;
        }
        int sum = 0;
        int prev = input[row*C+0];
        int switcher = 1;
        if (prev < max){
            switcher = 0;
            for(int i  = 1; i<C; i++){
                int data = input[row*C+i];
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


def nb_aucell(
    row, net, *, starts=None, offsets=None, n_up=None, n_fsets=None, max_aucs=None
):
    # Rank row
    row = cp.argsort(cp.argsort(-row), axis=1) + 1
    row = row.astype(cp.int32)
    # Empty acts
    es = cp.zeros((row.shape[0], n_fsets), dtype=cp.float32)
    es_inter = cp.zeros(row.shape[0], dtype=cp.int32)

    threads_per_block = 32
    blocks_per_grid = math.ceil(row.shape[0] / threads_per_block)

    for j in range(n_fsets):
        # Extract feature set
        srt = starts[j]
        off = offsets[j] + srt
        fset = net[srt:off]
        # Compute AUC
        x = row[:, fset]
        x.sort(axis=1)

        reduce_sum_2D_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (x, es_inter, x.shape[0], x.shape[1], n_up),
        )
        # Update acts matrix
        es[:, j] = (es_inter / max_aucs[j]).astype(cp.float32)
        es_inter[:] = 0
    return es.get()


def aucell(mat, net, n_up, verbose, batch_size=5000):
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
    offsets = cp.array(offsets, dtype=cp.int32)
    starts = cp.array(starts, dtype=cp.int32)
    net = cp.array(net, dtype=cp.int32)
    k = cp.minimum(offsets, n_up - 1)
    max_aucs = (k * (k - 1) / 2 + (n_up - k) * k).astype(cp.int32)
    n_samples = mat.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    for i in tqdm(range(n_batches), disable=not verbose):
        # Subset batch
        srt, end = i * batch_size, i * batch_size + batch_size
        if isinstance(mat, csr_matrix):
            row = _sparse_to_dense(cp_csr_matrix(mat[srt:end]))
        elif isinstance(mat, cp_csr_matrix):
            row = _sparse_to_dense(mat[srt:end])
        elif isinstance(mat, cp.ndarray):
            row = mat[srt:end]
        else:
            row = cp.array(mat[srt:end])
        es[srt:end] = nb_aucell(
            row,
            net,
            starts=starts,
            offsets=offsets,
            n_up=n_up,
            n_fsets=n_fsets,
            max_aucs=max_aucs,
        )
    return es


def run_aucell(
    mat: AnnData | pd.DataFrame | list,
    net: pd.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    batch_size: int = 5000,
    n_up: int | None = None,
    min_n: int = 5,
    seed: int = 42,
    verbose: bool = False,
    use_raw: bool | None = None,
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
    batch_size
        Size of the samples to use for each batch. Increasing this will consume more memory but it will run faster.
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
    estimate = aucell(m, net, n_up, verbose, batch_size)
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = "aucell_estimate"

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate

    else:
        return estimate
