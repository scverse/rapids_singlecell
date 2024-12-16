# This file will be removed in Q3 2025 when in favor of the CUML implementation

from __future__ import annotations

import math

import cupy as cp


class PCA_sparse:
    def __init__(self, n_components) -> None:
        self.n_components = n_components

    def fit(self, x):
        if self.n_components is None:
            n_rows = x.shape[0]
            n_cols = x.shape[1]
            self.n_components_ = min(n_rows, n_cols)
        else:
            self.n_components_ = self.n_components

        _check_matrix_for_zero_genes(x)
        self.n_samples_ = x.shape[0]
        self.n_features_in_ = x.shape[1] if x.ndim == 2 else 1
        self.dtype = x.data.dtype

        covariance, self.mean_, _ = _cov_sparse(x=x, return_mean=True)

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

        self.explained_variance_ = self.explained_variance_[: self.n_components_]

        self.explained_variance_ratio_ = self.explained_variance_ratio_[
            : self.n_components_
        ]

        return self

    def transform(self, X):
        precomputed_mean_impact = self.mean_ @ self.components_.T
        mean_impact = cp.ones(
            (X.shape[0], 1), dtype=cp.float32
        ) @ precomputed_mean_impact.reshape(1, -1)
        X_transformed = X.dot(self.components_.T) - mean_impact
        # X = X - self.mean_
        # X_transformed = X.dot(self.components_.T)
        self.components_ = self.components_.get()
        self.explained_variance_ = self.explained_variance_.get()
        self.explained_variance_ratio_ = self.explained_variance_ratio_.get()
        return X_transformed.get()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def _cov_sparse(x, return_gram=False, return_mean=False):
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

    gram_matrix = cp.zeros((x.shape[1], x.shape[1]), dtype=x.data.dtype)

    block = (128,)
    grid = (x.shape[0],)
    compute_mean_cov = _gramm_kernel_csr(x.data.dtype)
    compute_mean_cov(
        grid,
        block,
        (
            x.indptr,
            x.indices,
            x.data,
            x.shape[0],
            x.shape[1],
            gram_matrix,
        ),
    )

    copy_gram = _copy_kernel(x.data.dtype)
    block = (32, 32)
    grid = (math.ceil(x.shape[1] / block[0]), math.ceil(x.shape[1] / block[1]))
    copy_gram(
        grid,
        block,
        (gram_matrix, x.shape[1]),
    )

    mean_x = x.sum(axis=0) * (1 / x.shape[0])
    gram_matrix *= 1 / x.shape[0]

    if return_gram:
        cov_result = cp.zeros(
            (gram_matrix.shape[0], gram_matrix.shape[0]),
            dtype=gram_matrix.dtype,
        )
    else:
        cov_result = gram_matrix

    compute_cov = _cov_kernel(x.dtype)

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


def _check_matrix_for_zero_genes(X):
    gene_ex = cp.zeros(X.shape[1], dtype=cp.int32)

    from ._kernels._pca_sparse_kernel import _zero_genes_kernel

    block = (32,)
    grid = (int(math.ceil(X.nnz / block[0])),)
    _zero_genes_kernel(
        grid,
        block,
        (
            X.indices,
            gene_ex,
            X.nnz,
        ),
    )
    if cp.any(gene_ex == 0):
        raise ValueError(
            "There are genes with zero expression. "
            "Please remove them before running PCA."
        )
