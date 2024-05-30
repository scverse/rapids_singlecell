from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import issparse, isspmatrix_csc
from scanpy.get import _get_obs_rep

from ._simple import calculate_qc_metrics
from ._utils import _check_gpu_X, _check_nonnegative_integers, _get_mean_var

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

flavors = Literal[
    "seurat",
    "cell_ranger",
    "seurat_v3",
    "seurat_v3_paper",
    "pearson_residuals",
    "poisson_gene_selection",
]


def highly_variable_genes(
    adata: AnnData,
    *,
    layer: str = None,
    min_mean: float = 0.0125,
    max_mean: float = 3,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    n_top_genes: int = None,
    flavor: flavors = "seurat",
    n_bins: int = 20,
    span: float = 0.3,
    check_values: bool = True,
    theta: int = 100,
    clip: bool = None,
    chunksize: int = 1000,
    n_samples: int = 10000,
    batch_key: str = None,
) -> None:
    """\
    Annotate highly variable genes.
    Expects logarithmized data, except when `flavor='seurat_v3','seurat_v3_paper','pearson_residuals','poisson_gene_selection'`, in which count data is expected.

    Reimplementation of scanpy's function.
    Depending on flavor, this reproduces the R-implementations of Seurat, Cell Ranger, Seurat v3 and Pearson Residuals.
    Flavor `poisson_gene_selection` is an implementation of scvi, which is based on M3Drop. It requires gpu accelerated pytorch to be installed.

    For these dispersion-based methods, the normalized dispersion is obtained by scaling
    with the mean and standard deviation of the dispersions for genes falling into a given
    bin for mean expression of genes. This means that for each bin of mean expression,
    highly variable genes are selected.

    For `flavor='seurat_v3'`/`'seurat_v3_paper'`, a normalized variance for each gene
    is computed. First, the data are standardized (i.e., z-score normalization
    per feature) with a regularized standard deviation. Next, the normalized variance
    is computed as the variance of each gene after the transformation. Genes are ranked
    by the normalized variance.
    Only if `batch_key` is not `None`, the two flavors differ: For `flavor='seurat_v3'`, genes are first sorted by the median (across batches) rank, with ties broken by the number of batches a gene is a HVG.
    For `flavor='seurat_v3_paper'`, genes are first sorted by the number of batches a gene is a HVG, with ties broken by the median (across batches) rank.

    The following may help when comparing to Seurat's naming:
    If `batch_key=None` and `flavor='seurat'`, this mimics Seurat's `FindVariableFeatures(…, method='mean.var.plot')`.
    If `batch_key=None` and `flavor='seurat_v3'`/`flavor='seurat_v3_paper'`, this mimics Seurat's `FindVariableFeatures(..., method='vst')`.
    If `batch_key` is not `None` and `flavor='seurat_v3_paper'`, this mimics Seurat's `SelectIntegrationFeatures`.

    Parameters
    ----------
        adata
            AnnData object
        layer
            If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
        min_mean
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        max_mean
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        min_disp
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        max_disp
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        n_top_genes
            Number of highly-variable genes to keep.
        flavor :
            Choose the flavors for identifying highly variable genes. For the dispersion based methods
            in their default workflows, Seurat passes the cutoffs whereas Cell Ranger passes n_top_genes.
        n_bins
            Number of bins for binning the mean gene expression. Normalization is done with respect to each bin. If just a single gene falls into a bin, the normalized dispersion is artificially set to 1.
        span
            The fraction of the data (cells) used when estimating the variance in the loess
            model fit if `flavor='seurat_v3'`.
        check_values
            Check if counts in selected layer are integers. A Warning is returned if set to True.
            Only used if `flavor='seurat_v3'` or `'pearson_residuals'`.
        theta
            The negative binomial overdispersion parameter `theta` for Pearson residuals.
            Higher values correspond to less overdispersion (`var = mean + mean^2/theta`), and `theta=np.Inf` corresponds to a Poisson model.
        clip
            Only used if `flavor='pearson_residuals'`. Determines if and how residuals are clipped:
                * If `None`, residuals are clipped to the interval `[-sqrt(n_obs), sqrt(n_obs)]`, where `n_obs` is the number of cells in the dataset (default behavior).
                * If any scalar `c`, residuals are clipped to the interval `[-c, c]`. Set `clip=np.Inf` for no clipping.
        chunksize
            If `'poisson_gene_selection'`, this dertermines how many genes are processed at
            once. Choosing a smaller value will reduce the required memory.
        n_samples
            The number of Binomial samples to use to estimate posterior probability
            of enrichment of zeros for each gene (only for `flavor='poisson_gene_selection'`).
        batch_key
            If specified, highly-variable genes are selected within each batch separately and merged.

    Returns
    -------
        updates `adata.var` with the following fields:

            `highly_variable` : bool
                boolean indicator of highly-variable genes
            `means`: float
                means per gene
            `dispersions`: float
                For dispersion-based flavors, dispersions per gene
            `dispersions_norm`: float
                For dispersion-based flavors, normalized dispersions per gene
            `variances`: float
                For `flavor='seurat_v3','pearson_residuals'`, variance per gene
            `variances_norm`: float
                For `flavor='seurat_v3'`, normalized variance per gene, averaged in
                the case of multiple batches
            `residual_variances` : float
                For `flavor='pearson_residuals'`, residual variance per gene. Averaged in the
                case of multiple batches.
            `highly_variable_rank` : float
                For `flavor='seurat_v3','pearson_residuals'`, rank of the gene according to normalized
                variance, median rank in the case of multiple batches
            `highly_variable_nbatches` : int
                If batch_key is given, this denotes in how many batches genes are detected as HVG
            `highly_variable_intersection` : bool
                If batch_key is given, this denotes the genes that are highly variable in all batches
    """
    adata._sanitize()
    if flavor == "seurat_v3" or flavor == "seurat_v3_paper":
        _highly_variable_genes_seurat_v3(
            adata=adata,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            span=span,
            check_values=check_values,
            flavor=flavor,
        )
    elif flavor == "pearson_residuals":
        _highly_variable_pearson_residuals(
            adata=adata,
            theta=theta,
            clip=clip,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            layer=layer,
        )
    elif flavor == "poisson_gene_selection":
        _poisson_gene_selection(
            adata=adata,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            layer=layer,
            n_samples=n_samples,
            minibatch_size=chunksize,
        )
    else:
        cutoff = _Cutoffs.validate(
            n_top_genes=n_top_genes,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
        )
        del min_disp, max_disp, min_mean, max_mean, n_top_genes

        if batch_key is None:
            df = _highly_variable_genes_single_batch(
                adata, layer=layer, cutoff=cutoff, n_bins=n_bins, flavor=flavor
            )
        else:
            df = _highly_variable_genes_batched(
                adata,
                batch_key,
                layer=layer,
                cutoff=cutoff,
                n_bins=n_bins,
                flavor=flavor,
            )

        adata.uns["hvg"] = {"flavor": flavor}

        adata.var["highly_variable"] = df["highly_variable"]
        adata.var["means"] = df["means"]
        adata.var["dispersions"] = df["dispersions"]
        adata.var["dispersions_norm"] = df["dispersions_norm"].astype(
            np.float32, copy=False
        )

        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"]
            adata.var["highly_variable_intersection"] = df[
                "highly_variable_intersection"
            ]


@dataclass
class _Cutoffs:
    min_disp: float
    max_disp: float
    min_mean: float
    max_mean: float

    @classmethod
    def validate(
        cls,
        *,
        n_top_genes: int | None,
        min_disp: float,
        max_disp: float,
        min_mean: float,
        max_mean: float,
    ) -> _Cutoffs | int:
        if n_top_genes is None:
            return cls(min_disp, max_disp, min_mean, max_mean)

        cutoffs = {"min_disp", "max_disp", "min_mean", "max_mean"}
        defaults = {
            p.name: p.default
            for p in signature(highly_variable_genes).parameters.values()
            if p.name in cutoffs
        }
        if {k: v for k, v in locals().items() if k in cutoffs} != defaults:
            msg = "If you pass `n_top_genes`, all cutoffs are ignored."
            warnings.warn(msg, UserWarning)
        return n_top_genes

    def in_bounds(
        self,
        mean: NDArray[np.floating],
        dispersion_norm: NDArray[np.floating],
    ) -> NDArray[np.bool_]:
        return (
            (mean > self.min_mean)
            & (mean < self.max_mean)
            & (dispersion_norm > self.min_disp)
            & (dispersion_norm < self.max_disp)
        )


def _highly_variable_genes_single_batch(
    adata: AnnData,
    *,
    layer: str | None = None,
    cutoff: _Cutoffs | int,
    n_bins: int = 20,
    flavor: Literal["seurat", "cell_ranger"] = "seurat",
) -> pd.DataFrame:
    """\
    See `highly_variable_genes`.

    Returns
    -------
    A DataFrame that contains the columns
    `highly_variable`, `means`, `dispersions`, and `dispersions_norm`.
    """
    X = _get_obs_rep(adata, layer=layer)

    if hasattr(X, "_view_args"):  # AnnData array view
        # For compatibility with anndata<0.9
        X = X.copy()  # Doesn't actually copy memory, just removes View class wrapper

    if flavor == "seurat":
        X = X.copy()
        if issparse(X):
            X = X.expm1()
        else:
            X = cp.expm1(X)
    mean, var = _get_mean_var(X, axis=0)
    mean[mean == 0] = 1e-12
    disp = var / mean
    if flavor == "seurat":  # logarithmized mean as in Seurat
        disp[disp == 0] = np.nan
        disp = cp.log(disp)
        mean = cp.log1p(mean)
    df = pd.DataFrame()
    mean = mean.get()
    disp = disp.get()
    df["means"] = mean
    df["dispersions"] = disp
    df["mean_bin"] = _get_mean_bins(df["means"], flavor, n_bins)
    disp_stats = _get_disp_stats(df, flavor)

    # actually do the normalization
    df["dispersions_norm"] = (df["dispersions"] - disp_stats["avg"]) / disp_stats["dev"]
    df["highly_variable"] = _subset_genes(
        adata,
        mean=mean,
        dispersion_norm=df["dispersions_norm"].to_numpy(),
        cutoff=cutoff,
    )

    df.index = adata.var_names
    return df


def _get_mean_bins(
    means: pd.Series, flavor: Literal["seurat", "cell_ranger"], n_bins: int
) -> pd.Series:
    if flavor == "seurat":
        bins = n_bins
    elif flavor == "cell_ranger":
        bins = np.r_[-np.inf, np.percentile(means, np.arange(10, 105, 5)), np.inf]
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')

    return pd.cut(means, bins=bins)


def _get_disp_stats(
    df: pd.DataFrame, flavor: Literal["seurat", "cell_ranger"]
) -> pd.DataFrame:
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    if flavor == "seurat":
        disp_bin_stats = disp_grouped.agg(avg="mean", dev="std")
        _postprocess_dispersions_seurat(disp_bin_stats, df["mean_bin"])
    elif flavor == "cell_ranger":
        disp_bin_stats = disp_grouped.agg(avg="median", dev=_mad)
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')
    return disp_bin_stats.loc[df["mean_bin"]].set_index(df.index)


def _postprocess_dispersions_seurat(
    disp_bin_stats: pd.DataFrame, mean_bin: pd.Series
) -> None:
    # retrieve those genes that have nan std, these are the ones where
    # only a single gene fell in the bin and implicitly set them to have
    # a normalized disperion of 1
    one_gene_per_bin = disp_bin_stats["dev"].isnull()
    gen_indices = np.flatnonzero(one_gene_per_bin.loc[mean_bin])
    if len(gen_indices) == 0:
        return
    disp_bin_stats.loc[one_gene_per_bin, "dev"] = disp_bin_stats.loc[
        one_gene_per_bin, "avg"
    ]
    disp_bin_stats.loc[one_gene_per_bin, "avg"] = 0


def _mad(a):
    from statsmodels.robust import mad

    with warnings.catch_warnings():
        # MAD calculation raises the warning: "Mean of empty slice"
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return mad(a)


def _subset_genes(
    adata: AnnData,
    *,
    mean: NDArray[np.float64],
    dispersion_norm: NDArray[np.float64],
    cutoff: _Cutoffs | int,
) -> NDArray[np.bool_]:
    """Get boolean mask of genes with normalized dispersion in bounds."""
    if isinstance(cutoff, _Cutoffs):
        dispersion_norm = np.nan_to_num(dispersion_norm)  # similar to Seurat
        return cutoff.in_bounds(mean, dispersion_norm)
    n_top_genes = cutoff
    del cutoff

    if n_top_genes > adata.n_vars:
        n_top_genes = adata.n_vars
    disp_cut_off = _nth_highest(dispersion_norm, n_top_genes)
    return np.nan_to_num(dispersion_norm) >= disp_cut_off


def _nth_highest(x: NDArray[np.float64], n: int) -> float:
    x = x[~np.isnan(x)]
    if n > x.size:
        msg = "`n_top_genes` > number of normalized dispersions, returning all genes with normalized dispersions."
        warnings.warn(msg, UserWarning)
        n = x.size
    # interestingly, np.argpartition is slightly slower
    x[::-1].sort()
    return x[n - 1]


def _highly_variable_genes_batched(
    adata: AnnData,
    batch_key: str,
    *,
    layer: str | None,
    n_bins: int,
    flavor: Literal["seurat", "cell_ranger"],
    cutoff: _Cutoffs | int,
) -> pd.DataFrame:
    adata._sanitize()
    batches = adata.obs[batch_key].cat.categories
    dfs = []
    gene_list = adata.var_names
    for batch in batches:
        adata_subset = adata[adata.obs[batch_key] == batch]

        calculate_qc_metrics(adata_subset, layer=layer)
        filt = adata_subset.var["n_cells_by_counts"].to_numpy() > 0
        adata_subset = adata_subset[:, filt]

        hvg = _highly_variable_genes_single_batch(
            adata_subset, layer=layer, cutoff=cutoff, n_bins=n_bins, flavor=flavor
        )
        hvg.reset_index(drop=False, inplace=True, names=["gene"])

        if (n_removed := np.sum(~filt)) > 0:
            # Add 0 values for genes that were filtered out
            missing_hvg = pd.DataFrame(
                np.zeros((n_removed, len(hvg.columns))),
                columns=hvg.columns,
            )
            missing_hvg["highly_variable"] = missing_hvg["highly_variable"].astype(bool)
            missing_hvg["gene"] = gene_list[~filt]
            hvg = pd.concat([hvg, missing_hvg], ignore_index=True)

        dfs.append(hvg)

    df = pd.concat(dfs, axis=0)

    df["highly_variable"] = df["highly_variable"].astype(int)
    df = df.groupby("gene", observed=True).agg(
        {
            "means": "mean",
            "dispersions": "mean",
            "dispersions_norm": "mean",
            "highly_variable": "sum",
        }
    )
    df["highly_variable_nbatches"] = df["highly_variable"]
    df["highly_variable_intersection"] = df["highly_variable_nbatches"] == len(batches)

    if isinstance(cutoff, int):
        # sort genes by how often they selected as hvg within each batch and
        # break ties with normalized dispersion across batches
        df.sort_values(
            ["highly_variable_nbatches", "dispersions_norm"],
            ascending=False,
            na_position="last",
            inplace=True,
        )
        df["highly_variable"] = np.arange(df.shape[0]) < cutoff
    else:
        df["dispersions_norm"] = df["dispersions_norm"].fillna(0)  # similar to Seurat
        df["highly_variable"] = cutoff.in_bounds(df["means"], df["dispersions_norm"])

    return df


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


def _highly_variable_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: float | None = None,
    n_top_genes: int = 2000,
    batch_key: str | None = None,
    check_values: bool = True,
    layer: str | None = None,
):
    """
    Select highly variable genes using analytic Pearson residuals.
    Pearson residuals of a negative binomial offset model are computed
    (with overdispersion `theta` shared across genes). By default, overdispersion
    `theta=100` is used and residuals are clipped to `sqrt(n_obs)`. Finally, genes
    are ranked by residual variance.
    Expects raw count input.
    """
    X = _get_obs_rep(adata, layer=layer)
    _check_gpu_X(X, require_cf=True)
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    if n_top_genes is None:
        n_top_genes = 2000
        warnings.warn(
            "`flavor='pearson_residuals'` expects `n_top_genes`  to be defined, defaulting to 2000 HVGs",
            UserWarning,
        )
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    else:
        batch_info = adata.obs[batch_key].values

    n_batches = len(np.unique(batch_info))
    residual_gene_vars = []
    if issparse(X):
        from ._kernels._pr_kernels import _csc_hvg_res, _sparse_sum_csc

        sum_csc = _sparse_sum_csc(X.dtype)
        csc_hvg_res = _csc_hvg_res(X.dtype)
    else:
        from ._kernels._pr_kernels import _dense_hvg_res

        dense_hvg_res = _dense_hvg_res(X.dtype)

    for b in np.unique(batch_info):
        if issparse(X):
            X_batch = X[batch_info == b].tocsc()
            nnz_per_gene = cp.diff(X_batch.indptr).ravel()
        else:
            X_batch = cp.array(X[batch_info == b], dtype=X.dtype)
            nnz_per_gene = cp.sum(X_batch != 0, axis=0).ravel()
        nonzero_genes = cp.array(nnz_per_gene >= 1)
        X_batch = X_batch[:, nonzero_genes]
        if clip is None:
            n = X_batch.shape[0]
            clip = cp.sqrt(n, dtype=X.dtype)
        if clip < 0:
            raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")

        clip = cp.array([clip], dtype=X.dtype)
        theta = cp.array([theta], dtype=X.dtype)
        residual_gene_var = cp.zeros(X_batch.shape[1], dtype=X.dtype, order="C")
        if issparse(X_batch):
            sums_genes = cp.zeros(X_batch.shape[1], dtype=X.dtype)
            sums_cells = cp.zeros(X_batch.shape[0], dtype=X.dtype)
            block = (32,)
            grid = (int(math.ceil(X_batch.shape[1] / block[0])),)

            sum_csc(
                grid,
                block,
                (
                    X_batch.indptr,
                    X_batch.indices,
                    X_batch.data,
                    sums_genes,
                    sums_cells,
                    X_batch.shape[1],
                ),
            )
            sum_total = sums_genes.sum().squeeze()
            csc_hvg_res(
                grid,
                block,
                (
                    X_batch.indptr,
                    X_batch.indices,
                    X_batch.data,
                    sums_genes,
                    sums_cells,
                    residual_gene_var,
                    sum_total,
                    clip,
                    theta,
                    X_batch.shape[1],
                    X_batch.shape[0],
                ),
            )
        else:
            sums_genes = cp.sum(X_batch, axis=0, dtype=X.dtype).ravel()
            sums_cells = cp.sum(X_batch, axis=1, dtype=X.dtype).ravel()
            sum_total = sums_genes.sum().squeeze()
            block = (32,)
            grid = (int(math.ceil(X_batch.shape[1] / block[0])),)
            dense_hvg_res(
                grid,
                block,
                (
                    cp.array(X_batch, dtype=X.dtype, order="F"),
                    sums_genes,
                    sums_cells,
                    residual_gene_var,
                    sum_total,
                    clip,
                    theta,
                    X_batch.shape[1],
                    X_batch.shape[0],
                ),
            )

        unmasked_residual_gene_var = cp.zeros(len(nonzero_genes))
        unmasked_residual_gene_var[nonzero_genes] = residual_gene_var
        residual_gene_vars.append(unmasked_residual_gene_var.reshape(1, -1))

    residual_gene_vars = cp.concatenate(residual_gene_vars, axis=0)
    # Get rank per gene within each batch
    # argsort twice gives ranks, small rank means most variable
    ranks_residual_var = cp.argsort(cp.argsort(-residual_gene_vars, axis=1), axis=1)
    ranks_residual_var = ranks_residual_var.astype(X.dtype)
    # count in how many batches a genes was among the n_top_genes
    highly_variable_nbatches = cp.sum(
        (ranks_residual_var < n_top_genes).astype(int), axis=0
    ).get()
    ranks_residual_var[ranks_residual_var >= n_top_genes] = np.nan
    ranks_residual_var = ranks_residual_var.get()
    ranks_masked_array = np.ma.masked_invalid(ranks_residual_var)
    # Median rank across batches, ignoring batches in which gene was not selected
    medianrank_residual_var = np.ma.median(ranks_masked_array, axis=0).filled(np.nan)
    means, variances = _get_mean_var(X, axis=0)
    means, variances = means.get(), variances.get()
    df = pd.DataFrame.from_dict(
        {
            "means": means,
            "variances": variances,
            "residual_variances": cp.mean(residual_gene_vars, axis=0).get(),
            "highly_variable_rank": medianrank_residual_var,
            "highly_variable_nbatches": highly_variable_nbatches.astype(np.int64),
            "highly_variable_intersection": highly_variable_nbatches == n_batches,
        }
    )
    df = df.set_index(adata.var_names)
    df.sort_values(
        ["highly_variable_nbatches", "highly_variable_rank"],
        ascending=[False, True],
        na_position="last",
        inplace=True,
    )
    high_var = np.zeros(df.shape[0], dtype=bool)
    high_var[:n_top_genes] = True
    df["highly_variable"] = high_var
    df = df.loc[adata.var_names, :]

    computed_on = layer if layer else "adata.X"
    adata.uns["hvg"] = {"flavor": "pearson_residuals", "computed_on": computed_on}
    adata.var["means"] = df["means"].values
    adata.var["variances"] = df["variances"].values
    adata.var["residual_variances"] = df["residual_variances"]
    adata.var["highly_variable_rank"] = df["highly_variable_rank"].values
    if batch_key is not None:
        adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].values
        adata.var["highly_variable_intersection"] = df[
            "highly_variable_intersection"
        ].values
    adata.var["highly_variable"] = df["highly_variable"].values


def _poisson_gene_selection(
    adata: AnnData,
    *,
    layer: str | None = None,
    n_top_genes: int = None,
    n_samples: int = 10000,
    batch_key: str = None,
    minibatch_size: int = 1000,
    check_values: bool = True,
    **kwargs,
) -> None:
    """
    Rank and select genes based on the enrichment of zero counts.
    Enrichment is considered by comparing data to a Poisson count model.
    This is based on M3Drop: https://github.com/tallulandrews/M3Drop
    The method accounts for library size internally, a raw count matrix should be provided.
    Instead of Z-test, enrichment of zeros is quantified by posterior
    probabilities from a binomial model, computed through sampling.

    Parameters
    ----------
    adata
        AnnData object
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    n_top_genes
        How many variable genes to select.
    n_samples
        The number of Binomial samples to use to estimate posterior probability
        of enrichment of zeros for each gene.
    batch_key
        key in adata.obs that contains batch info. If None, do not use batch info.
        Default: ``None``.
    minibatch_size
        Size of temporary matrix for incremental calculation. Larger is faster but
        requires more RAM or GPU memory. (The default should be fine unless
        there are hundreds of millions cells or millions of genes.)

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields
    highly_variable : bool
        boolean indicator of highly-variable genes
    **observed_fraction_zeros**
        fraction of observed zeros per gene
    **expected_fraction_zeros**
        expected fraction of observed zeros per gene
    prob_zero_enrichment : float
        Probability of zero enrichment, median across batches in the case of multiple batches
    prob_zero_enrichment_rank : float
        Rank of the gene according to probability of zero enrichment, median rank in the case of multiple batches
    prob_zero_enriched_nbatches : int
        If batch_key is given, this denotes in how many batches genes are detected as zero enriched
    """
    try:
        import torch
    except ImportError:
        raise ImportError("Please install pytorch package via `pip install pytorch")
    if n_top_genes is None:
        n_top_genes = 2000
        warnings.warn(
            "`flavor='seurat_v3'` expects `n_top_genes`  to be defined, defaulting to 2000 HVGs",
            UserWarning,
        )

    X = _get_obs_rep(adata, layer=layer)
    _check_gpu_X(X)
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    else:
        batch_info = adata.obs[batch_key].values

    prob_zero_enrichments = []
    obs_frac_zeross = []
    exp_frac_zeross = []

    with torch.no_grad():
        for b in np.unique(batch_info):
            X_batch = X[batch_info == b]
            total_counts = torch.tensor(X_batch.sum(1).ravel(), device="cuda")
            X_batch = X_batch.tocsc()
            # Calculate empirical statistics.
            sum_0 = X_batch.sum(axis=0).ravel()
            scaled_means = torch.tensor(sum_0 / sum_0.sum(), device="cuda")

            observed_fraction_zeros = torch.tensor(
                cp.asarray(
                    1.0 - cp.diff(X_batch.indptr).ravel() / X_batch.shape[0]
                ).ravel(),
                device="cuda",
            )
            # Calculate probability of zero for a Poisson model.
            # Perform in batches to save memory.
            minibatch_size = min(total_counts.shape[0], minibatch_size)
            n_batches = total_counts.shape[0] // minibatch_size

            expected_fraction_zeros = torch.zeros(scaled_means.shape, device="cuda")

            for i in range(n_batches):
                total_counts_batch = total_counts[
                    i * minibatch_size : (i + 1) * minibatch_size
                ]
                # Use einsum for outer product.
                expected_fraction_zeros += torch.exp(
                    -torch.einsum("i,j->ij", [scaled_means, total_counts_batch])
                ).sum(1)

            total_counts_batch = total_counts[(i + 1) * minibatch_size :]
            expected_fraction_zeros += torch.exp(
                -torch.einsum("i,j->ij", [scaled_means, total_counts_batch])
            ).sum(1)
            expected_fraction_zeros /= X_batch.shape[0]

            # Compute probability of enriched zeros through sampling from Binomial distributions.
            observed_zero = torch.distributions.Binomial(probs=observed_fraction_zeros)
            expected_zero = torch.distributions.Binomial(probs=expected_fraction_zeros)

            # extra_zeros = torch.zeros(expected_fraction_zeros.shape, device="cuda")

            extra_zeros = observed_zero.sample((n_samples,)) > expected_zero.sample(
                (n_samples,)
            )
            # for i in range(n_samples):
            #    extra_zeros += observed_zero.sample() > expected_zero.sample()

            extra_zeros = extra_zeros.sum(0)
            prob_zero_enrichment = (extra_zeros / n_samples).cpu().numpy()

            obs_frac_zeros = observed_fraction_zeros.cpu().numpy()
            exp_frac_zeros = expected_fraction_zeros.cpu().numpy()

            # Clean up memory (tensors seem to stay in GPU unless actively deleted).
            del scaled_means
            del total_counts
            del expected_fraction_zeros
            del observed_fraction_zeros
            del extra_zeros
            torch.cuda.empty_cache()

            prob_zero_enrichments.append(prob_zero_enrichment.reshape(1, -1))
            obs_frac_zeross.append(obs_frac_zeros.reshape(1, -1))
            exp_frac_zeross.append(exp_frac_zeros.reshape(1, -1))

    # Combine per batch results
    prob_zero_enrichments = np.concatenate(prob_zero_enrichments, axis=0)
    obs_frac_zeross = np.concatenate(obs_frac_zeross, axis=0)
    exp_frac_zeross = np.concatenate(exp_frac_zeross, axis=0)

    ranked_prob_zero_enrichments = prob_zero_enrichments.argsort(axis=1).argsort(axis=1)
    median_prob_zero_enrichments = np.median(prob_zero_enrichments, axis=0)

    median_obs_frac_zeross = np.median(obs_frac_zeross, axis=0)
    median_exp_frac_zeross = np.median(exp_frac_zeross, axis=0)

    median_ranked = np.median(ranked_prob_zero_enrichments, axis=0)

    num_batches_zero_enriched = np.sum(
        ranked_prob_zero_enrichments >= (adata.shape[1] - n_top_genes), axis=0
    )

    df = pd.DataFrame(index=np.array(adata.var_names))
    df["observed_fraction_zeros"] = median_obs_frac_zeross
    df["expected_fraction_zeros"] = median_exp_frac_zeross
    df["prob_zero_enriched_nbatches"] = num_batches_zero_enriched
    df["prob_zero_enrichment"] = median_prob_zero_enrichments
    df["prob_zero_enrichment_rank"] = median_ranked

    df["highly_variable"] = False
    sort_columns = ["prob_zero_enriched_nbatches", "prob_zero_enrichment_rank"]
    top_genes = df.nlargest(n_top_genes, sort_columns).index
    df.loc[top_genes, "highly_variable"] = True

    adata.uns["hvg"] = {"flavor": "poisson_zeros"}
    adata.var["highly_variable"] = df["highly_variable"].values
    adata.var["observed_fraction_zeros"] = df["observed_fraction_zeros"].values
    adata.var["expected_fraction_zeros"] = df["expected_fraction_zeros"].values
    adata.var["prob_zero_enriched_nbatches"] = df["prob_zero_enriched_nbatches"].values
    adata.var["prob_zero_enrichment"] = df["prob_zero_enrichment"].values
    adata.var["prob_zero_enrichment_rank"] = df["prob_zero_enrichment_rank"].values

    if batch_key is not None:
        adata.var["prob_zero_enriched_nbatches"] = df[
            "prob_zero_enriched_nbatches"
        ].values
