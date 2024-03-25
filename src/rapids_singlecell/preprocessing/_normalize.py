from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import cupy as cp
from cupyx.scipy import sparse
from scanpy.get import _get_obs_rep, _set_obs_rep

from ._utils import _check_gpu_X, _check_nonnegative_integers

if TYPE_CHECKING:
    from anndata import AnnData


def normalize_total(
    adata: AnnData,
    *,
    target_sum: int | None = None,
    layer: int | str = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | sparse.csr_matrix | cp.ndarray | None:
    """
    Normalizes rows in matrix so they sum to `target_sum`

    Parameters
    ----------
        adata
            AnnData object

        target_sum
            If `None`, after normalization, each observation (cell) has a total count equal to the median of total counts for observations (cells) before normalization.

        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.

        inplace
            Whether to update `adata` or return the matrix.

        copy
            Whether to return a copy or update `adata`. Not compatible with inplace=False.

    Returns
    -------
    Returns a normalized copy or  updates `adata` with a normalized version of \
    the original `adata.X` and `adata.layers['layer']`, depending on `inplace`.

    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()
    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X)

    if not inplace:
        X = X.copy()

    if sparse.isspmatrix_csc(X):
        X = X.tocsr()
    if not target_sum:
        if sparse.issparse(X):
            from ._kernels._norm_kernel import _get_sparse_sum_major

            counts_per_cell = cp.zeros(X.shape[0], dtype=X.dtype)
            sum_kernel = _get_sparse_sum_major(X.dtype)
            sum_kernel(
                (X.shape[0],),
                (64,),
                (X.indptr, X.data, counts_per_cell, X.shape[0]),
            )
        else:
            counts_per_cell = X.sum(axis=1)
        target_sum = cp.median(counts_per_cell)

    if sparse.isspmatrix_csr(X):
        from ._kernels._norm_kernel import _mul_csr

        mul_kernel = _mul_csr(X.dtype)
        mul_kernel(
            (math.ceil(X.shape[0] / 128),),
            (128,),
            (X.indptr, X.data, X.shape[0], int(target_sum)),
        )

    else:
        from ._kernels._norm_kernel import _mul_dense

        if not X.flags.c_contiguous:
            X = cp.asarray(X, order="C")
        mul_kernel = _mul_dense(X.dtype)
        mul_kernel(
            (math.ceil(X.shape[0] / 128),),
            (128,),
            (X, X.shape[0], X.shape[1], int(target_sum)),
        )

    if inplace:
        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata
    elif not inplace:
        return X


def log1p(
    adata: AnnData,
    *,
    layer: str | None = None,
    obsm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | sparse.spmatrix | cp.ndarray | None:
    """
    Calculated the natural logarithm of one plus the sparse matrix.

    Parameters
    ----------
        adata:
            AnnData object

        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.

        obsm
            Entry of obsm to transform.

        inplace
            Whether to update `adata` or return the matrix.

        copy
            Whether to return a copy or update `adata`. Not compatible with inplace=False.

    Returns
    -------
            The resulting sparse matrix after applying the natural logarithm of one plus the input matrix. \
            If `copy` is set to True, returns the new sparse matrix. Otherwise, updates the `adata` object \
            in-place and returns None.

    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()
    X = _get_obs_rep(adata, layer=layer, obsm=obsm)

    _check_gpu_X(X)

    if isinstance(X, cp.ndarray):
        X = cp.log1p(X)
    else:
        X = X.log1p()

    adata.uns["log1p"] = {"base": None}
    if inplace:
        _set_obs_rep(adata, X, layer=layer, obsm=obsm)

    if copy:
        return adata
    elif not inplace:
        return X


def normalize_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: float | None = None,
    check_values: bool = True,
    layer: str | None = None,
    inplace: bool = True,
) -> cp.ndarray | None:
    """
    Applies analytic Pearson residual normalization, based on Lause21.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to `sqrt(n_obs)`
    and overdispersion `theta=100` is used.

    Parameters
    ----------
        adata:
            AnnData object
        theta
            The negative binomial overdispersion parameter theta for Pearson residuals.
            Higher values correspond to less overdispersion (var = mean + mean^2/theta), and theta=np.Inf corresponds to a Poisson model.
        clip
            Determines if and how residuals are clipped:
            If None, residuals are clipped to the interval [-sqrt(n_obs), sqrt(n_obs)], where n_obs is the number of cells in the dataset (default behavior).
            If any scalar c, residuals are clipped to the interval [-c, c]. Set clip=np.Inf for no clipping.
        check_values
            If True, checks if counts in selected layer are integers as expected by this function,
            and return a warning if non-integers are found. Otherwise, proceed without checking. Setting this to False can speed up code for large datasets.
        layer
            Layer to use as input instead of X. If None, X is used.
        inplace
            If True, update AnnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    -------
        If `inplace=True`, `adata.X` or the selected layer in `adata.layers` is updated with the normalized values. \
        If `inplace=False` the normalized matrix is returned.

    """
    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X, require_cf=True)

    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    computed_on = layer if layer else "adata.X"
    settings_dict = {"theta": theta, "clip": clip, "computed_on": computed_on}
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    if clip is None:
        n = X.shape[0]
        clip = cp.sqrt(n, dtype=X.dtype)
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")
    theta = cp.array([1 / theta], dtype=X.dtype)
    clip = cp.array([clip], dtype=X.dtype)
    sums_cells = cp.zeros(X.shape[0], dtype=X.dtype)
    sums_genes = cp.zeros(X.shape[1], dtype=X.dtype)

    if sparse.issparse(X):
        residuals = cp.zeros(X.shape, dtype=X.dtype)
        if sparse.isspmatrix_csc(X):
            from ._kernels._pr_kernels import _sparse_norm_res_csc, _sparse_sum_csc

            block = (8,)
            grid = (int(math.ceil(X.shape[1] / block[0])),)
            sum_csc = _sparse_sum_csc(X.dtype)
            sum_csc(
                grid,
                block,
                (X.indptr, X.indices, X.data, sums_genes, sums_cells, X.shape[1]),
            )
            sum_total = 1 / sums_genes.sum().squeeze()
            norm_res = _sparse_norm_res_csc(X.dtype)
            norm_res(
                grid,
                block,
                (
                    X.indptr,
                    X.indices,
                    X.data,
                    sums_cells,
                    sums_genes,
                    residuals,
                    sum_total,
                    clip,
                    theta,
                    X.shape[0],
                    X.shape[1],
                ),
            )
        elif sparse.isspmatrix_csr(X):
            from ._kernels._pr_kernels import _sparse_norm_res_csr, _sparse_sum_csr

            block = (8,)
            grid = (int(math.ceil(X.shape[0] / block[0])),)
            sum_csr = _sparse_sum_csr(X.dtype)
            sum_csr(
                grid,
                block,
                (X.indptr, X.indices, X.data, sums_genes, sums_cells, X.shape[0]),
            )
            sum_total = 1 / sums_genes.sum().squeeze()
            norm_res = _sparse_norm_res_csr(X.dtype)
            norm_res(
                grid,
                block,
                (
                    X.indptr,
                    X.indices,
                    X.data,
                    sums_cells,
                    sums_genes,
                    residuals,
                    sum_total,
                    clip,
                    theta,
                    X.shape[0],
                    X.shape[1],
                ),
            )
        else:
            raise ValueError(
                "Please transform you sparse matrix into CSR or CSC format."
            )
    else:
        from ._kernels._pr_kernels import _norm_res_dense, _sum_dense

        residuals = cp.zeros(X.shape, dtype=X.dtype)
        block = (8, 8)
        grid = (
            math.ceil(residuals.shape[0] / block[0]),
            math.ceil(residuals.shape[1] / block[1]),
        )
        sum_dense = _sum_dense(X.dtype)
        sum_dense(
            grid,
            block,
            (X, sums_cells, sums_genes, residuals.shape[0], residuals.shape[1]),
        )
        sum_total = 1 / sums_genes.sum().squeeze()
        norm_res = _norm_res_dense(X.dtype)
        norm_res(
            grid,
            block,
            (
                X,
                residuals,
                sums_cells,
                sums_genes,
                sum_total,
                clip,
                theta,
                residuals.shape[0],
                residuals.shape[1],
            ),
        )

    if inplace is True:
        adata.uns["pearson_residuals_normalization"] = settings_dict
        _set_obs_rep(adata, residuals, layer=layer)
    else:
        return residuals
