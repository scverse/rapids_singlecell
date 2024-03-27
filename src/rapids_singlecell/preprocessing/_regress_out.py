from __future__ import annotations

import math
from typing import Literal

import cupy as cp
from anndata import AnnData
from cuml.linear_model import LinearRegression
from cupyx.scipy import sparse
from scanpy._utils import view_to_actual
from scanpy.get import _get_obs_rep, _set_obs_rep

from ._utils import _check_gpu_X


def regress_out(
    adata: AnnData,
    keys: str | list,
    *,
    layer: str | None = None,
    inplace: bool = True,
    batchsize: int | Literal["all"] | None = 100,
    verbose: bool = False,
) -> cp.ndarray | None:
    """
    Use linear regression to adjust for the effects of unwanted noise
    and variation.

    Parameters
    ----------
        adata
            AnnData object

        keys
            Keys for numerical observation annotation on which to regress on.

        layer
            Layer to regress instead of `X`. If `None`, `X` is regressed.

        inplace
            Whether to update `adata` or return the corrected matrix of `adata.X` and `adata.layers`.

        batchsize
            Number of genes that should be processed together. \
            If `'all'` all genes will be processed together if `.n_obs` <100000. \
            If `None` each gene will be analysed separately. \
            Will be ignored if cuML version < 22.12

        verbose
            Print debugging information

    Returns
    -------
    Returns a corrected copy or  updates `adata` with a corrected version of the \
    original `adata.X` and `adata.layers['layer']`, depending on `inplace`.
    """
    if batchsize != "all" and type(batchsize) not in [int, type(None)]:
        raise ValueError("batchsize must be `int`, `None` or `'all'`")

    if isinstance(adata, AnnData):
        view_to_actual(adata)

    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X)

    if sparse.issparse(X) and not sparse.isspmatrix_csc(X):
        X = X.tocsc()

    dim_regressor = 2
    if isinstance(keys, list):
        dim_regressor = len(keys) + 1

    regressors = cp.ones(X.shape[0] * dim_regressor).reshape(
        (X.shape[0], dim_regressor), order="F"
    )
    if dim_regressor == 2:
        regressors[:, 1] = cp.array(adata.obs[keys]).ravel()
    else:
        for i in range(dim_regressor - 1):
            regressors[:, i + 1] = cp.array(adata.obs[keys[i]]).ravel()

    outputs = cp.empty(X.shape, dtype=X.dtype, order="F")

    cuml_supports_multi_target = LinearRegression._get_tags()["multioutput"]

    if cuml_supports_multi_target and batchsize:
        if batchsize == "all" and X.shape[0] < 100000:
            if sparse.issparse(X):
                X = X.todense()
            lr = LinearRegression(
                fit_intercept=False, output_type="cupy", algorithm="svd"
            )
            lr.fit(regressors, X, convert_dtype=True)
            outputs[:] = X - lr.predict(regressors)
        else:
            if batchsize == "all":
                batchsize = 100
            n_batches = math.ceil(X.shape[1] / batchsize)
            for batch in range(n_batches):
                start_idx = batch * batchsize
                stop_idx = min(batch * batchsize + batchsize, X.shape[1])
                if sparse.issparse(X):
                    arr_batch = X[:, start_idx:stop_idx].todense()
                else:
                    arr_batch = X[:, start_idx:stop_idx].copy()
                lr = LinearRegression(
                    fit_intercept=False, output_type="cupy", algorithm="svd"
                )
                lr.fit(regressors, arr_batch, convert_dtype=True)
                outputs[:, start_idx:stop_idx] = arr_batch - lr.predict(regressors)
    else:
        if X.shape[0] < 100000 and sparse.issparse(X):
            X = X.todense()
        for i in range(X.shape[1]):
            if verbose and i % 500 == 0:
                print(f"Regressed {i} out of {X.shape[1]}")

            y = X[:, i]
            outputs[:, i] = _regress_out_chunk(regressors, y)

    if inplace:
        _set_obs_rep(adata, outputs, layer=layer)
    else:
        return outputs


def _regress_out_chunk(X, y):
    """
    Performs a data_cunk.shape[1] number of local linear regressions,
    replacing the data in the original chunk w/ the regressed result.

    Parameters
    ----------
    X : cupy.ndarray of shape (n_cells, 3)
        Matrix of regressors
    y : cupy.sparse.spmatrix of shape (n_cells,)
        Sparse matrix containing a single column of the cellxgene matrix

    Returns
    -------
    dense_mat : cupy.ndarray of shape (n_cells,)
        Adjusted column
    """
    if sparse.issparse(y):
        y = y.todense()

    lr = LinearRegression(fit_intercept=False, output_type="cupy")
    lr.fit(X, y, convert_dtype=True)
    return y.reshape(
        y.shape[0],
    ) - lr.predict(X).reshape(y.shape[0])
