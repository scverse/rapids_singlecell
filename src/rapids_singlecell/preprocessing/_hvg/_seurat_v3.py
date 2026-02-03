from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import issparse, isspmatrix_csc
from scanpy.get import _get_obs_rep

from rapids_singlecell.preprocessing._utils import (
    _check_gpu_X,
    _check_nonnegative_integers,
    _get_mean_var,
)

if TYPE_CHECKING:
    from anndata import AnnData


def _highly_variable_genes_seurat_v3(
    adata: AnnData,
    *,
    layer: str | None = None,
    n_top_genes: int = None,
    batch_key: str | None = None,
    span: float = 0.3,
    check_values=True,
    flavor: str = "seurat_v3",
):
    """\
    See `highly_variable_genes`.
    For further implementation details see https://www.overleaf.com/read/ckptrbgzzzpg

    Returns
    -------
    updates `.var` with the following fields:
    highly_variable : bool
        boolean indicator of highly-variable genes.
    **means**
        means per gene.
    **variances**
        variance per gene.
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches.
    highly_variable_rank : float
        Rank of the gene according to normalized variance, median rank in the case of multiple batches.
    highly_variable_nbatches : int
        If batch_key is given, this denotes in how many batches genes are detected as HVG.
    """
    if n_top_genes is None:
        n_top_genes = 2000
        warnings.warn(
            "`flavor='seurat_v3'` expects `n_top_genes`  to be defined, defaulting to 2000 HVGs",
            UserWarning,
        )
    try:
        from skmisc.loess import loess
    except ImportError:
        raise ImportError(
            "Please install skmisc package via `pip install --user scikit-misc"
        )

    df = pd.DataFrame(index=adata.var.index)
    X = _get_obs_rep(adata, layer=layer)
    _check_gpu_X(X)
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='seurat_v3'` expects raw count data, but non-integers were found.",
            UserWarning,
        )

    mean, var = _get_mean_var(X, axis=0)
    df["means"], df["variances"] = mean.get(), var.get()
    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    else:
        batch_info = adata.obs[batch_key].values
        if isspmatrix_csc(X):
            X = X.tocsr()

    norm_gene_vars = []
    for b in np.unique(batch_info):
        X_batch = X[batch_info == b]
        mean, var = _get_mean_var(X_batch, axis=0)
        not_const = var > 0
        estimat_var = cp.zeros(X_batch.shape[1], dtype=np.float64)

        y = cp.log10(var[not_const])
        x = cp.log10(mean[not_const])
        model = loess(x.get(), y.get(), span=span, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = cp.sqrt(10**estimat_var)
        batch_counts = X_batch
        N = X_batch.shape[0]
        vmax = cp.sqrt(N)
        clip_val = reg_std * vmax + mean
        if issparse(batch_counts):
            seurat_v3_elementwise_kernel = cp.ElementwiseKernel(
                "T data, S idx, raw D clip_val",
                "raw D sq_sum, raw D sum",
                """
                D element = min((double)data, clip_val[idx]);
                atomicAdd(&sq_sum[idx], element * element);
                atomicAdd(&sum[idx], element);
                """,
                "seurat_v3_elementwise_kernel",
                no_return=True,
            )
            if isspmatrix_csc(batch_counts):
                batch_counts = batch_counts.tocsr()
            squared_batch_counts_sum = cp.zeros(clip_val.shape, dtype=cp.float64)
            batch_counts_sum = cp.zeros(clip_val.shape, dtype=cp.float64)
            seurat_v3_elementwise_kernel(
                batch_counts.data,
                batch_counts.indices,
                clip_val,
                squared_batch_counts_sum,
                batch_counts_sum,
            )
        else:
            batch_counts = batch_counts.astype(cp.float64)
            clip_val_broad = cp.repeat(clip_val, batch_counts.shape[0]).reshape(
                batch_counts.shape
            )

            cp.putmask(
                batch_counts,
                batch_counts > clip_val_broad,
                clip_val_broad,
            )
            # Calculate the sum of squared values for each column
            squared_batch_counts_sum = cp.sum(batch_counts**2, axis=0)

            # Calculate the sum for each column
            batch_counts_sum = cp.sum(batch_counts, axis=0)

        norm_gene_var = (1 / ((N - 1) * cp.square(reg_std))) * (
            (N * cp.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )

        norm_gene_vars.append(norm_gene_var.reshape(1, -1))
    norm_gene_vars = cp.concatenate(norm_gene_vars, axis=0)
    ranked_norm_gene_vars = cp.argsort(cp.argsort(-norm_gene_vars, axis=1), axis=1)

    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = cp.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ranked_norm_gene_vars = ranked_norm_gene_vars.get()
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)

    df["highly_variable_nbatches"] = num_batches_high_var.get()
    df["highly_variable_rank"] = median_ranked
    df["variances_norm"] = cp.mean(norm_gene_vars, axis=0).get()
    if flavor == "seurat_v3":
        sort_cols = ["highly_variable_rank", "highly_variable_nbatches"]
        sort_ascending = [True, False]
    elif flavor == "seurat_v3_paper":
        sort_cols = ["highly_variable_nbatches", "highly_variable_rank"]
        sort_ascending = [False, True]
    else:
        raise ValueError(f"Did not recognize flavor {flavor}")
    sorted_index = (
        df[sort_cols]
        .sort_values(sort_cols, ascending=sort_ascending, na_position="last")
        .index
    )
    df["highly_variable"] = False
    df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True
    adata.var["highly_variable"] = df["highly_variable"].values
    adata.var["highly_variable_rank"] = df["highly_variable_rank"].values
    adata.var["means"] = df["means"].values
    adata.var["variances"] = df["variances"].values
    adata.var["variances_norm"] = df["variances_norm"].values.astype(
        "float64", copy=False
    )
    if batch_key:
        adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].values
    adata.uns["hvg"] = {"flavor": flavor}
