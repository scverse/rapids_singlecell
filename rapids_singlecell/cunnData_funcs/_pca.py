import math
from typing import Optional, Union

import cupy as cp
import numpy as np
from anndata import AnnData
from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.decomposition import PCA, TruncatedSVD
from cuml.internals.input_utils import sparse_scipy_to_cp
from cupyx.scipy.sparse import csr_matrix, isspmatrix_csr
from cupyx.scipy.sparse import issparse as cpissparse
from scipy.sparse import issparse

from rapids_singlecell.cunnData import cunnData


def pca(
    cudata: Union[cunnData, AnnData],
    layer: str = None,
    n_comps: Optional[int] = None,
    zero_center: bool = True,
    random_state: Union[int, None] = 0,
    use_highly_variable: Optional[bool] = None,
    chunked: bool = False,
    chunk_size: int = None,
) -> None:
    """
    Performs PCA using the cuML decomposition function for the :class:`~rapids_singlecell.cunnData.cunnData` object.

    Parameters
    ----------
        cudata :
            cunnData, AnnData object

        layer
            If provided, use `cudata.layers[layer]` for expression values instead of `cudata.X`.

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
        adds fields to `cudata` :
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
    if use_highly_variable is True and "highly_variable" not in cudata.var.keys():
        raise ValueError(
            "Did not find cudata.var['highly_variable']. "
            "Either your data already only consists of highly-variable genes "
            "or consider running `highly_variable_genes` first."
        )

    X = cudata.layers[layer] if layer is not None else cudata.X

    if use_highly_variable is None:
        use_highly_variable = True if "highly_variable" in cudata.var.keys() else False

    if use_highly_variable:
        X = X[:, cudata.var["highly_variable"]]

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
                    X = sparse_scipy_to_cp(X)
                    X = csr_matrix(X)
                pca_func = PCA_sparse(n_components=n_comps)
                X_pca = pca_func.fit_transform(X)
            pca_func = PCA(
                n_components=n_comps, random_state=random_state, output_type="numpy"
            )
            X_pca = pca_func.fit_transform(X)

        elif not zero_center:
            pca_func = TruncatedSVD(
                n_components=n_comps, random_state=random_state, output_type="numpy"
            )
            X_pca = pca_func.fit_transform(X)

    cudata.obsm["X_pca"] = X_pca
    cudata.uns["pca"] = {
        "variance": pca_func.explained_variance_,
        "variance_ratio": pca_func.explained_variance_ratio_,
    }
    if use_highly_variable:
        cudata.varm["PCs"] = np.zeros(shape=(cudata.n_vars, n_comps))
        cudata.varm["PCs"][cudata.var["highly_variable"]] = pca_func.components_.T
    else:
        cudata.varm["PCs"] = pca_func.components_.T


class PCA_sparse:
    def __init__(self, n_components) -> None:
        self.n_components = n_components

    def fit(self, x):
        if self.n_components is None:
            print(
                "Warning(`fit`): As of v0.16, PCA invoked without an"
                " n_components argument defaults to using"
                " min(n_samples, n_features) rather than 1"
            )
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
        gram_matrix = cp.zeros((x.shape[1], x.shape[1]), dtype=self.dtype)

        block = (128,)
        grid = (x.shape[0],)

        compute_mean_cov = _cov_kernel_sparse_xx(self.dtype)
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
        copy_gram = _warp_copy_kernel(x.data.dtype)
        block = (32, 32)
        grid = (math.ceil(x.shape[1] / block[0]), math.ceil(x.shape[1] / block[1]))
        copy_gram(
            grid,
            block,
            (gram_matrix, x.shape[1]),
        )

        mean_x = x.sum(axis=0) * (1 / x.shape[0])
        gram_matrix *= 1 / x.shape[0]
        cov_result = gram_matrix

        compute_cov = _cov_kernel(self.dtype)

        block_size = (8, 8)
        grid_size = (
            math.ceil(gram_matrix.shape[0] / 8),
            math.ceil(gram_matrix.shape[1] / 8),
        )
        compute_cov(
            grid_size,
            block_size,
            (cov_result, gram_matrix, mean_x, mean_x, gram_matrix.shape[0]),
        )

        covariance, self.mean_ = cov_result, mean_x
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
        return X_transformed.get()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


cov_kernel_str_sparse_xx = r"""
(const int *indptr,const int *index, {0} *data,int nrows,int ncols, {0}  * out) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows) return;

    int start = indptr[row];
    int end = indptr[row + 1];

    for (int idx1 = start; idx1 < end; idx1++)
    {
        int index1 = index[idx1];
        {0} data1 = data[idx1];
        for(int idx2 = idx1+col; idx2 < end; idx2 += blockDim.x){
            int index2 = index[idx2];
            {0} data2 = data[idx2];
            atomicAdd(&out[index1*ncols+index2], data1*data2);
        }
    }
}
"""

copy_kernel = r"""
({0} *out, int ncols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= ncols || col >= ncols) return;

    if (row > col) {
        out[row*ncols+col] = out[col*ncols+row]; // Copy the upper triangle to the lower triangle
    }
}
"""

cov_kernel_str = r"""
({0} *cov_values, {0} *gram_matrix, {0} *mean_x, {0} *mean_y, int n_cols) {

    int rid = blockDim.x * blockIdx.x + threadIdx.x;
    int cid = blockDim.y * blockIdx.y + threadIdx.y;

    if(rid >= n_cols || cid >= n_cols) return;

    cov_values[rid * n_cols + cid] = \
        gram_matrix[rid * n_cols + cid] - mean_x[rid] * mean_y[cid];
}
"""


def _cov_kernel_sparse_xx(dtype):
    return cuda_kernel_factory(
        cov_kernel_str_sparse_xx, (dtype,), "cov_kernel_sprase_xx"
    )


def _warp_copy_kernel(dtype):
    return cuda_kernel_factory(copy_kernel, (dtype,), "_copy_kernel")


def _cov_kernel(dtype):
    return cuda_kernel_factory(cov_kernel_str, (dtype,), "cov_kernel")
