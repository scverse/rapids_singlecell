import math
import warnings
from typing import Optional, Union

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy.sparse import issparse
from scanpy.get import _get_obs_rep

from rapids_singlecell.cunnData import cunnData

from ._utils import _check_gpu_X, _check_nonnegative_integers, _get_mean_var


def highly_variable_genes(
    adata: Union[AnnData, cunnData],
    layer: str = None,
    min_mean: float = 0.0125,
    max_mean: float = 3,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    n_top_genes: int = None,
    flavor: str = "seurat",
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
    Expects logarithmized data, except when `flavor='seurat_v3','pearson_residuals','poisson_gene_selection'`, in which count data is expected.

    Reimplentation of scanpy's function.
    Depending on flavor, this reproduces the R-implementations of Seurat, Cell Ranger, Seurat v3 and Pearson Residuals.
    Flavor `poisson_gene_selection` is an implementation of scvi, which is based on M3Drop. It requiers gpu accelerated pytorch to be installed.

    For these dispersion-based methods, the normalized dispersion is obtained by scaling
    with the mean and standard deviation of the dispersions for genes falling into a given
    bin for mean expression of genes. This means that for each bin of mean expression,
    highly variable genes are selected.

    For Seurat v3, a normalized variance for each gene is computed. First, the data
    are standardized (i.e., z-score normalization per feature) with a regularized
    standard deviation. Next, the normalized variance is computed as the variance
    of each gene after the transformation. Genes are ranked by the normalized variance.

    Parameters
    ----------
        adata
            AnnData/ cunnData object
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
            Choose the flavor (`seurat`, `cell_ranger`, `seurat_v3`, `pearson_residuals`, `poisson_gene_selection`) for identifying highly variable genes. For the dispersion based methods
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
        upates `adata.var` with the following fields:

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
    if flavor == "seurat_v3":
        _highly_variable_genes_seurat_v3(
            adata=adata,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            span=span,
            check_values=check_values,
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
        if batch_key is None:
            X = _get_obs_rep(adata, layer=layer)
            _check_gpu_X(X)
            df = _highly_variable_genes_single_batch(
                X.copy(),
                min_disp=min_disp,
                max_disp=max_disp,
                min_mean=min_mean,
                max_mean=max_mean,
                n_top_genes=n_top_genes,
                n_bins=n_bins,
                flavor=flavor,
            )
        else:
            adata.obs[batch_key] = adata.obs[batch_key].astype("category")
            batches = adata.obs[batch_key].cat.categories
            df = []
            genes = adata.var.index.to_numpy()
            X = adata.layers[layer] if layer is not None else adata.X
            _check_gpu_X(X)
            for batch in batches:
                if not isinstance(X, cp.ndarray):
                    X_batch = X[adata.obs[batch_key] == batch,].tocsc()
                    nnz_per_gene = cp.diff(X_batch.indptr).ravel()
                else:
                    X_batch = X[adata.obs[batch_key] == batch,].copy()
                    nnz_per_gene = cp.sum(X_batch > 0, axis=0).ravel()
                good_genes = cp.where(nnz_per_gene >= 1)[0]
                bad_genes = cp.where(nnz_per_gene < 1)[0]
                X_batch = X_batch[:, good_genes]
                good_genes = good_genes.get()
                bad_genes = bad_genes.get()
                inter_genes = genes[good_genes]
                other_gens_inter = genes[bad_genes]
                hvg_inter = _highly_variable_genes_single_batch(
                    X_batch,
                    min_disp=min_disp,
                    max_disp=max_disp,
                    min_mean=min_mean,
                    max_mean=max_mean,
                    n_top_genes=n_top_genes,
                    n_bins=n_bins,
                    flavor=flavor,
                )
                hvg_inter["gene"] = inter_genes
                missing_hvg = pd.DataFrame(
                    np.zeros((len(other_gens_inter), len(hvg_inter.columns))),
                    columns=hvg_inter.columns,
                )
                missing_hvg["highly_variable"] = missing_hvg["highly_variable"].astype(
                    bool
                )
                missing_hvg["gene"] = other_gens_inter
                # hvg = hvg_inter.append(missing_hvg, ignore_index=True)
                hvg = pd.concat([hvg_inter, missing_hvg], ignore_index=True)
                idxs = np.concatenate((good_genes, bad_genes))
                hvg = hvg.loc[np.argsort(idxs)]
                df.append(hvg)

            df = pd.concat(df, axis=0)
            df["highly_variable"] = df["highly_variable"].astype(int)
            df = df.groupby("gene").agg(
                {
                    "means": np.nanmean,
                    "dispersions": np.nanmean,
                    "dispersions_norm": np.nanmean,
                    "highly_variable": np.nansum,
                }
            )
            df.rename(
                columns={"highly_variable": "highly_variable_nbatches"}, inplace=True
            )
            df["highly_variable_intersection"] = df["highly_variable_nbatches"] == len(
                batches
            )
            if n_top_genes is not None:
                # sort genes by how often they selected as hvg within each batch and
                # break ties with normalized dispersion across batches
                df = df.sort_values(
                    ["highly_variable_nbatches", "dispersions_norm"],
                    ascending=False,
                    na_position="last",
                )

                high_var = np.zeros(df.shape[0])
                high_var[:n_top_genes] = True
                df["highly_variable"] = high_var.astype(bool)
                df = df.loc[genes]
            else:
                df = df.loc[genes]
                dispersion_norm = df.dispersions_norm.values
                dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
                gene_subset = np.logical_and.reduce(
                    (
                        df.means > min_mean,
                        df.means < max_mean,
                        df.dispersions_norm > min_disp,
                        df.dispersions_norm < max_disp,
                    )
                )
                df["highly_variable"] = gene_subset

        adata.var["highly_variable"] = df["highly_variable"].values
        adata.var["means"] = df["means"].values
        adata.var["dispersions"] = df["dispersions"].values
        adata.var["dispersions_norm"] = df["dispersions_norm"].values
        adata.uns["hvg"] = {"flavor": flavor}
        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df[
                "highly_variable_nbatches"
            ].values
            adata.var["highly_variable_intersection"] = df[
                "highly_variable_intersection"
            ].values


def _highly_variable_genes_single_batch(
    X,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
    max_disp=np.inf,
    n_top_genes=None,
    flavor="seurat",
    n_bins=20,
):
    """\
        See `highly_variable_genes`.

    Returns
    -------
    A DataFrame that contains the columns
    `highly_variable`, `means`, `dispersions`, and `dispersions_norm`.
    """
    if flavor == "seurat":
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
    if flavor == "seurat":
        df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
        disp_grouped = df.groupby("mean_bin")["dispersions"]
        disp_mean_bin = disp_grouped.mean()
        disp_std_bin = disp_grouped.std(ddof=1)
        # retrieve those genes that have nan std, these are the ones where
        # only a single gene fell in the bin and implicitly set them to have
        # a normalized disperion of 1
        one_gene_per_bin = disp_std_bin.isnull()
        np.where(one_gene_per_bin[df["mean_bin"].values])[0].tolist()

        # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,
        # but there’s still a dtype error without “.value”.
        disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[
            one_gene_per_bin.values
        ].values
        disp_mean_bin[one_gene_per_bin.values] = 0
        # actually do the normalization
        df["dispersions_norm"] = (
            df["dispersions"].values  # use values here as index differs
            - disp_mean_bin[df["mean_bin"].values].values
        ) / disp_std_bin[df["mean_bin"].values].values

    elif flavor == "cell_ranger":
        from statsmodels import robust

        df["mean_bin"] = pd.cut(
            df["means"],
            np.r_[-np.inf, np.percentile(df["means"], np.arange(10, 105, 5)), np.inf],
        )
        disp_grouped = df.groupby("mean_bin")["dispersions"]
        disp_median_bin = disp_grouped.median()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            disp_mad_bin = disp_grouped.apply(robust.mad)
            df["dispersions_norm"] = (
                df["dispersions"].values - disp_median_bin[df["mean_bin"].values].values
            ) / disp_mad_bin[df["mean_bin"].values].values

    dispersion_norm = df["dispersions_norm"].values
    if n_top_genes is not None:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[
            ::-1
        ].sort()  # interestingly, np.argpartition is slightly slower
        if n_top_genes > X.shape[1]:
            n_top_genes = X.shape[1]
        if n_top_genes > dispersion_norm.size:
            warnings.warn(
                "`n_top_genes` > number of normalized dispersions, returning all genes with normalized dispersions.",
                UserWarning,
            )
            n_top_genes = dispersion_norm.size
        disp_cut_off = dispersion_norm[n_top_genes - 1]
        gene_subset = np.nan_to_num(df["dispersions_norm"].values) >= disp_cut_off
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        gene_subset = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )
    df["highly_variable"] = gene_subset
    return df


def _highly_variable_genes_seurat_v3(
    adata: cunnData,
    layer: Optional[str] = None,
    n_top_genes: int = None,
    batch_key: Optional[str] = None,
    span: float = 0.3,
    check_values=True,
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
        X_batch = X_batch.astype(cp.float64)
        batch_counts = X_batch
        N = X_batch.shape[0]
        vmax = cp.sqrt(N)
        clip_val = reg_std * vmax + mean
        if issparse(batch_counts):
            mask = batch_counts.data > clip_val[batch_counts.indices]
            batch_counts.data[mask] = clip_val[batch_counts.indices[mask]]
            squared_batch_counts_sum = cp.array(batch_counts.power(2).sum(axis=0))
            batch_counts_sum = cp.array(batch_counts.sum(axis=0))
        else:
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
    sorted_index = (
        df[["highly_variable_rank", "highly_variable_nbatches"]]
        .sort_values(
            ["highly_variable_rank", "highly_variable_nbatches"],
            ascending=[True, False],
            na_position="last",
        )
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
    adata.uns["hvg"] = {"flavor": "seurat_v3"}


def _highly_variable_pearson_residuals(
    adata: cunnData,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    check_values: bool = True,
    layer: Optional[str] = None,
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
    _check_gpu_X(X)
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
    adata: cunnData,
    layer: Optional[str] = None,
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
    probabilites from a binomial model, computed through sampling.

    Parameters
    ----------
    adata
        AnnData/ cunnData object (with sparse X matrix).
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    n_top_genes
        How many variable genes to select.
    n_samples
        The number of Binomial samples to use to estimate posterior probability
        of enrichment of zeros for each gene.
    batch_key
        key in adata.obs that contains batch info. If None, do not use batch info.
        Defatult: ``None``.
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
