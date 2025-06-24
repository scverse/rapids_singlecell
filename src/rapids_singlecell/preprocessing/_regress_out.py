from __future__ import annotations

import math
from typing import Literal, Union

import cupy as cp
from anndata import AnnData
from cuml.linear_model import LinearRegression
from cupyx.scipy import sparse
from scanpy._utils import view_to_actual
from scanpy.get import _get_obs_rep, _set_obs_rep

from ._utils import _check_gpu_X, _sparse_to_dense


def regress_out(
    adata: AnnData,
    keys: str | list,
    *,
    layer: str | None = None,
    inplace: bool = True,
    batchsize: int | Literal["all"] | None = None,
    verbose: bool = False,
) -> Union[cp.ndarray, None]:  # noqa: UP007
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

    if not sparse.issparse(X) and inplace is False:
        X = cp.array(X)
    # Create regressors
    dim_regressor = 2
    if isinstance(keys, list):
        dim_regressor = len(keys) + 1

    regressors = cp.ones(X.shape[0] * dim_regressor, dtype=X.dtype).reshape(
        (X.shape[0], dim_regressor), order="F"
    )
    if dim_regressor == 2:
        regressors[:, 1] = cp.array(adata.obs[keys]).ravel()
    else:
        for i in range(dim_regressor - 1):
            regressors[:, i + 1] = cp.array(adata.obs[keys[i]], dtype=X.dtype).ravel()
    # Set default batch size based on the number of samples in X
    if batchsize is None:
        batchsize = 100 if X.shape[0] > 100000 else "all"

    # Validate the choice of "all" batch size
    if batchsize == "all":
        if cp.linalg.det(regressors.T @ regressors) == 0:
            batchsize = 100

    # Do reggression
    if batchsize == "all":
        if sparse.issparse(X):
            X = _sparse_to_dense(X, order="C")
        else:
            X = cp.ascontiguousarray(X)
        inv_gram_matrix = cp.linalg.inv(regressors.T @ regressors)
        coeff = inv_gram_matrix @ (regressors.T @ X)
        cp.cublas.gemm("N", "N", regressors, coeff, alpha=-1, beta=1, out=X)
        # X -= regressors @ coeff

    else:
        if sparse.issparse(X):
            X = _sparse_to_dense(X, order="F")
        else:
            X = cp.asfortranarray(X)
        n_batches = math.ceil(X.shape[1] / batchsize)
        for batch in range(n_batches):
            start_idx = batch * batchsize
            stop_idx = min(batch * batchsize + batchsize, X.shape[1])

            arr_batch = X[:, start_idx:stop_idx].copy()
            lr = LinearRegression(
                fit_intercept=False, output_type="cupy", algorithm="svd"
            )
            lr.fit(regressors, arr_batch, convert_dtype=True)
            X[:, start_idx:stop_idx] = arr_batch - lr.predict(regressors)

    # X = cp.ascontiguousarray(X)
    if inplace:
        _set_obs_rep(adata, X, layer=layer)
    else:
        return X
