from __future__ import annotations

import math

import cupy as cp
import dask
from cuml.internals.memory_utils import with_cupy_rmm

from rapids_singlecell._compat import (
    _get_dask_client,
    _meta_dense,
)
from rapids_singlecell.preprocessing._utils import _get_mean_var


class PCA_sparse_dask:
    def __init__(self, n_components, client) -> None:
        self.n_components = n_components
        self.client = _get_dask_client(client)

    def fit(self, x):
        if self.n_components is None:
            n_rows = x.shape[0]
            n_cols = x.shape[1]
            self.n_components_ = min(n_rows, n_cols)
        else:
            self.n_components_ = self.n_components

        self.n_samples_ = x.shape[0]
        self.n_features_in_ = x.shape[1] if x.ndim == 2 else 1
        self.dtype = x.dtype
        covariance, self.mean_, _ = _cov_sparse_dask(self.client, x=x, return_mean=True)
        self.explained_variance_, self.components_ = cp.linalg.eigh(
            covariance, UPLO="U"
        )
        # NOTE: We reverse the eigen vector and eigen values here
        # because cupy provides them in ascending order. Make a copy otherwise
        # it is not C_CONTIGUOUS anymore and would error when converting to
        # CumlArray
        self.explained_variance_ = self.explained_variance_[::-1]

        self.components_ = cp.flip(self.components_, axis=1)

        self.components_ = self.components_.T[: self.n_components_, :]

        self.explained_variance_ratio_ = self.explained_variance_ / cp.sum(
            self.explained_variance_
        )
        if self.n_components_ < min(self.n_samples_, self.n_features_in_):
            self.noise_variance_ = self.explained_variance_[self.n_components_ :].mean()
        else:
            self.noise_variance_ = cp.array([0.0])
        self.explained_variance_ = self.explained_variance_[: self.n_components_]

        self.explained_variance_ratio_ = self.explained_variance_ratio_[
            : self.n_components_
        ]
        return self

    def transform(self, X):
        from ._kernels._pca_sparse_kernel import denser_kernel

        kernel = denser_kernel(X.dtype)
        kernel.compile()

        def _transform(X_part, mean_, components_):
            dense = cp.zeros(X_part.shape, dtype=X.dtype)
            max_nnz = cp.diff(X_part.indptr).max()
            tpb = (32, 32)
            bpg_x = math.ceil(X_part.shape[0] / tpb[0])
            bpg_y = math.ceil(max_nnz / tpb[1])
            bpg = (bpg_x, bpg_y)
            kernel(
                bpg,
                tpb,
                (
                    X_part.indptr,
                    X_part.indices,
                    X_part.data,
                    dense,
                    X_part.shape[0],
                    X_part.shape[1],
                ),
            )
            dense = dense - mean_
            X_pca = dense.dot(components_.T)
            return X_pca

        X_pca = X.map_blocks(
            _transform,
            mean_=self.mean_,
            components_=self.components_,
            dtype=X.dtype,
            chunks=(X.chunks[0], self.n_components_),
            meta=_meta_dense(X.dtype),
        )

        self.components_ = self.components_.get()
        self.explained_variance_ = self.explained_variance_.get()
        self.explained_variance_ratio_ = self.explained_variance_ratio_.get()
        return X_pca

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


@with_cupy_rmm
def _cov_sparse_dask(client, x, return_gram=False, return_mean=False):
    """
    Computes the mean and the covariance of matrix X of
    the form Cov(X, X) = E(XX) - E(X)E(X)

    This is a temporary fix for
    cuml issue #5475 and cupy issue #7699,
    where the operation `x.T.dot(x)` did not work for
    larger sparse matrices.

    Parameters
    ----------

    x : cupyx.scipy.sparse of size (m, n)
    return_gram : boolean (default = False)
        If True, gram matrix of the form (1 / n) * X.T.dot(X)
        will be returned.
        When True, a copy will be created
        to store the results of the covariance.
        When False, the local gram matrix result
        will be overwritten
    return_mean: boolean (default = False)
        If True, the Maximum Likelihood Estimate used to
        calculate the mean of X and X will be returned,
        of the form (1 / n) * mean(X) and (1 / n) * mean(X)

    Returns
    -------

    result : cov(X, X) when return_gram and return_mean are False
            cov(X, X), gram(X, X) when return_gram is True,
            return_mean is False
            cov(X, X), mean(X), mean(X) when return_gram is False,
            return_mean is True
            cov(X, X), gram(X, X), mean(X), mean(X)
            when return_gram is True and return_mean is True
    """

    from ._kernels._pca_sparse_kernel import (
        _copy_kernel,
        _cov_kernel,
        _gramm_kernel_csr,
    )

    compute_mean_cov = _gramm_kernel_csr(x.dtype)
    compute_mean_cov.compile()

    @dask.delayed
    def __gram_block(x_part):
        n_cols = x_part.shape[1]
        gram_matrix = cp.zeros(n_cols, dtype=x.dtype)

        block = (128,)
        grid = (x_part.shape[0],)
        compute_mean_cov(
            grid,
            block,
            (
                x_part.indptr,
                x_part.indices,
                x_part.data,
                x_part.shape[0],
                n_cols,
                gram_matrix,
            ),
        )
        return gram_matrix

    blocks = x.to_delayed().ravel()
    gram_chunk_matrices = dask.array.map_blocks(
        __gram_block,
        x,
        meta=cp.array((1.0,), dtype=x.dtype),
        dtype=x.dtype,
        chunks=((x.shape[1],) * len(blocks), (x.shape[1],)),
    )
    gram_chunk_matrices = gram_chunk_matrices.reshape(
        len(blocks), x.shape[1], x.shape[1]
    )
    gram_matrix = gram_chunk_matrices.sum(axis=0).compute()
    mean_x, _ = _get_mean_var(x, client=client)
    mean_x = mean_x.astype(x.dtype)
    copy_gram = _copy_kernel(x.dtype)
    block = (32, 32)
    grid = (math.ceil(x.shape[1] / block[0]), math.ceil(x.shape[1] / block[1]))
    copy_gram(
        grid,
        block,
        (gram_matrix, x.shape[1]),
    )

    gram_matrix *= 1 / x.shape[0]

    if return_gram:
        cov_result = cp.zeros(
            (gram_matrix.shape[0], gram_matrix.shape[0]),
            dtype=gram_matrix.dtype,
        )
    else:
        cov_result = gram_matrix

    compute_cov = _cov_kernel(gram_matrix.dtype)

    block_size = (32, 32)
    grid_size = (math.ceil(gram_matrix.shape[0] / 8),) * 2
    compute_cov(
        grid_size,
        block_size,
        (cov_result, gram_matrix, mean_x, mean_x, gram_matrix.shape[0]),
    )

    if not return_gram and not return_mean:
        return cov_result
    elif return_gram and not return_mean:
        return cov_result, gram_matrix
    elif not return_gram and return_mean:
        return cov_result, mean_x, mean_x
    elif return_gram and return_mean:
        return cov_result, gram_matrix, mean_x, mean_x
