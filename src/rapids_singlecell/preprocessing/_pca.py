import math
from typing import Optional, Union

import cupy as cp
import numpy as np
from anndata import AnnData
from cuml.decomposition import PCA, TruncatedSVD
from cuml.internals.input_utils import sparse_scipy_to_cp
from cupyx.scipy.sparse import csr_matrix, isspmatrix_csr
from cupyx.scipy.sparse import issparse as cpissparse
from scipy.sparse import issparse

from rapids_singlecell.cunnData import cunnData


def pca(
    adata: Union[AnnData, cunnData],
    layer: str = None,
    n_comps: Optional[int] = None,
    zero_center: bool = True,
    random_state: Union[int, None] = 0,
    use_highly_variable: Optional[bool] = None,
    chunked: bool = False,
    chunk_size: int = None,
) -> None:
    """
    Performs PCA using the cuml decomposition function.

    Parameters
    ----------
        adata :
            AnnData/ cunnData object

        layer
            If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.

        n_comps
            Number of principal components to compute. Defaults to 50, or 1 - minimum
            dimension size of selected representation

        zero_center
            If `True`, compute standard PCA from covariance matrix.
            If `False`, omit zero-centering variables

        random_state
            Change to use different initial states for the optimization.

        use_highly_variable
            Whether to use highly variable genes only, stored in
            `.var['highly_variable']`.
            By default uses them if they have been determined beforehand.

        chunked
            If `True`, perform an incremental PCA on segments of `chunk_size`.
            The incremental PCA automatically zero centers and ignores settings of
            `random_seed` and `svd_solver`. If `False`, perform a full PCA.

        chunk_size
            Number of observations to include in each chunk.
            Required if `chunked=True` was passed.

    Returns
    -------
        adds fields to `adata` :
            `.obsm['X_pca']`
                PCA representation of data.
            `.varm['PCs']`
                The principal components containing the loadings.
            `.uns['pca']['variance_ratio']`
                Ratio of explained variance.
            `.uns['pca']['variance']`
                Explained variance, equivalent to the eigenvalues of the
                covariance matrix.
    """
    if use_highly_variable is True and "highly_variable" not in adata.var.keys():
        raise ValueError(
            "Did not find adata.var['highly_variable']. "
            "Either your data already only consists of highly-variable genes "
            "or consider running `highly_variable_genes` first."
        )

    X = adata.layers[layer] if layer is not None else adata.X

    if use_highly_variable is None:
        use_highly_variable = True if "highly_variable" in adata.var.keys() else False

    if use_highly_variable:
        X = X[:, adata.var["highly_variable"]]

    if n_comps is None:
        min_dim = min(X.shape[0], X.shape[1])
        if 50 >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = 50

    if chunked:
        from cuml.decomposition import IncrementalPCA

        X_pca = np.zeros((X.shape[0], n_comps), X.dtype)

        pca_func = IncrementalPCA(
            n_components=n_comps, output_type="numpy", batch_size=chunk_size
        )
        pca_func.fit(X)

        n_batches = math.ceil(X.shape[0] / chunk_size)
        for batch in range(n_batches):
            start_idx = batch * chunk_size
            stop_idx = min(batch * chunk_size + chunk_size, X.shape[0])
            chunk = X[start_idx:stop_idx, :]
            if issparse(chunk) or cpissparse(chunk):
                chunk = chunk.toarray()
            X_pca[start_idx:stop_idx] = pca_func.transform(chunk)
    else:
        if zero_center:
            if cpissparse(X) or issparse(X):
                if issparse(X):
                    X = sparse_scipy_to_cp(X, dtype=X.dtype)
                    X = csr_matrix(X)
                pca_func = PCA_sparse(n_components=n_comps)
                X_pca = pca_func.fit_transform(X)
            else:
                pca_func = PCA(
                    n_components=n_comps, random_state=random_state, output_type="numpy"
                )
                X_pca = pca_func.fit_transform(X)

        elif not zero_center:
            pca_func = TruncatedSVD(
                n_components=n_comps, random_state=random_state, output_type="numpy"
            )
            X_pca = pca_func.fit_transform(X)

    adata.obsm["X_pca"] = X_pca
    adata.uns["pca"] = {
        "variance": pca_func.explained_variance_,
        "variance_ratio": pca_func.explained_variance_ratio_,
    }
    if use_highly_variable:
        adata.varm["PCs"] = np.zeros(shape=(adata.n_vars, n_comps))
        adata.varm["PCs"][adata.var["highly_variable"]] = pca_func.components_.T
    else:
        adata.varm["PCs"] = pca_func.components_.T


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

        if not isspmatrix_csr(x):
            x = x.tocsr()
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
        X = X - self.mean_
        X_transformed = X.dot(self.components_.T)
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
