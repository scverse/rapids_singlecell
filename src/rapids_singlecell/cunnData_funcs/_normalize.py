import math
import warnings
from typing import Optional

import cupy as cp
import cupyx as cpx
from anndata import AnnData
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell.cunnData import cunnData

from ._utils import _check_gpu_X, _check_nonnegative_integers


def normalize_total(
    cudata: cunnData, target_sum: int, layer: Optional[str] = None, inplace: bool = True
) -> Optional[cpx.scipy.sparse.csr_matrix]:
    """
    Normalizes rows in matrix so they sum to `target_sum`

    Parameters
    ----------
        cudata:
            cunnData object

        target_sum :
            Each row will be normalized to sum to this value

        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.

        inplace
            Whether to update `cudata` or return the normalized matrix.


    Returns
    -------
    Returns a normalized copy or  updates `cudata` with a normalized version of \
    the original `cudata.X` and `cudata.layers['layer']`, depending on `inplace`.

    """
    X = _get_obs_rep(cudata, layer=layer)

    if isinstance(X, AnnData):
        _check_gpu_X(X)

    if not inplace:
        X = X.copy()

    from ._kernels._norm_kernel import _mul_kernel_csr

    mul_kernel = _mul_kernel_csr(X.dtype)
    mul_kernel(
        (math.ceil(X.shape[0] / 32.0),),
        (32,),
        (X.indptr, X.data, X.shape[0], int(target_sum)),
    )

    if inplace:
        _set_obs_rep(cudata, X, layer=layer)
    else:
        return X


def log1p(
    cudata: cunnData, layer: Optional[str] = None, copy: bool = False
) -> Optional[cpx.scipy.sparse.spmatrix]:
    """
    Calculated the natural logarithm of one plus the sparse matrix.

    Parameters
    ----------
        cudata
            cunnData object

        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.

        copy
            Whether to return a copy or update `cudata`.

    Returns
    -------
            The resulting sparse matrix after applying the natural logarithm of one plus the input matrix. \
            If `copy` is set to True, returns the new sparse matrix. Otherwise, updates the `cudata` object \
            in-place and returns None.

    """
    X = _get_obs_rep(cudata, layer=layer)

    if isinstance(X, AnnData):
        _check_gpu_X(X)

    X = X.log1p()
    cudata.uns["log1p"] = {"base": None}
    if not copy:
        _set_obs_rep(cudata, X, layer=layer)
    else:
        return X


def normalize_pearson_residuals(
    cudata: cunnData,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[cp.ndarray]:
    """
    Applies analytic Pearson residual normalization, based on Lause21.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to `sqrt(n_obs)`
    and overdispersion `theta=100` is used.

    Parameters
    ----------
        cudata
            cunnData object
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
            If True, update cunnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    -------
        If `inplace=True`, `cudata.X` or the selected layer in `cudata.layers` is updated with the normalized values. \
        If `inplace=False` the normalized matrix is returned.

    """
    X = _get_obs_rep(cudata, layer=layer)

    _check_gpu_X(X)

    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    computed_on = layer if layer else "cudata.X"
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

    if cpx.scipy.sparse.issparse(X):
        residuals = cp.zeros(X.shape, dtype=X.dtype)
        if cpx.scipy.sparse.isspmatrix_csc(X):
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
        elif cpx.scipy.sparse.isspmatrix_csr(X):
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
        cudata.uns["pearson_residuals_normalization"] = settings_dict
        _set_obs_rep(cudata, residuals, layer=layer)
    else:
        return residuals
