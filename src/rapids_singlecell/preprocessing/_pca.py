from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cuml.internals.input_utils import sparse_scipy_to_cp
from cupyx.scipy.sparse import csr_matrix, isspmatrix_csr
from cupyx.scipy.sparse import issparse as cpissparse
from scanpy._utils import Empty, _empty
from scanpy.preprocessing._pca import _handle_mask_var
from scipy.sparse import issparse

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.get import _get_obs_rep

from ._utils import _check_gpu_X

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray


def pca(
    adata: AnnData,
    n_comps: int | None = None,
    *,
    layer: str = None,
    zero_center: bool = True,
    svd_solver: str | None = None,
    random_state: int | None = 0,
    mask_var: NDArray[np.bool] | str | None | Empty = _empty,
    use_highly_variable: bool | None = None,
    dtype: str = "float32",
    chunked: bool = False,
    chunk_size: int = None,
    key_added: str | None = None,
    copy: bool = False,
) -> None | AnnData:
    """
    Performs PCA using the cuml decomposition function.

    Parameters
    ----------
        adata
            AnnData object

        n_comps
            Number of principal components to compute. Defaults to 50, or 1 - minimum \
            dimension size of selected representation

        layer
            If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.

        zero_center
            If `True`, compute standard PCA from covariance matrix. \
            If `False`, omit zero-centering variables

        svd_solver
            Solver to use for the PCA computation. \
            Must be one of {'full', 'jacobi', 'auto'}. \
            Defaults to 'auto'.

        random_state
            Change to use different initial states for the optimization.

        mask_var
            Mask to use for the PCA computation. \
            If `None`, all variables are used. \
            If `np.ndarray`, use the provided mask. \
            If `str`, use the mask stored in `adata.var[mask_var]`.

        use_highly_variable
            Whether to use highly variable genes only, stored in \
            `.var['highly_variable']`. \
            By default uses them if they have been determined beforehand.

        dtype
            Numpy data type string to which to convert the result.

        chunked
            If `True`, perform an incremental PCA on segments of `chunk_size`. \
            The incremental PCA automatically zero centers and ignores settings of \
            `random_seed` and `svd_solver`. If `False`, perform a full PCA.

        chunk_size
            Number of observations to include in each chunk. \
            Required if `chunked=True` was passed.

        key_added
            If not specified, the embedding is stored as
            :attr:`~anndata.AnnData.obsm`\\ `['X_pca']`, the loadings as
            :attr:`~anndata.AnnData.varm`\\ `['PCs']`, and the the parameters in
            :attr:`~anndata.AnnData.uns`\\ `['pca']`.
            If specified, the embedding is stored as
            :attr:`~anndata.AnnData.obsm`\\ ``[key_added]``, the loadings as
            :attr:`~anndata.AnnData.varm`\\ ``[key_added]``, and the the parameters in
            :attr:`~anndata.AnnData.uns`\\ ``[key_added]``.

        copy
            Whether to return a copy or update `adata`.

    Returns
    -------
        adds fields to `adata`:

            `.obsm['X_pca' | key_added]`
                PCA representation of data.
            `.varm['PCs' | key_added]`
                The principal components containing the loadings.
            `.uns['pca' | key_added]['variance_ratio']`
                Ratio of explained variance.
            `.uns['pca' | key_added]['variance']`
                Explained variance, equivalent to the eigenvalues of the \
                covariance matrix.
    """
    if use_highly_variable is True and "highly_variable" not in adata.var.keys():
        raise ValueError(
            "Did not find adata.var['highly_variable']. "
            "Either your data already only consists of highly-variable genes "
            "or consider running `highly_variable_genes` first."
        )
    if copy:
        adata = adata.copy()

    X = _get_obs_rep(adata, layer=layer)

    mask_var_param, mask_var = _handle_mask_var(adata, mask_var, use_highly_variable)
    del use_highly_variable
    X = X[:, mask_var] if mask_var is not None else X

    if svd_solver is None:
        svd_solver = "auto"

    if n_comps is None:
        min_dim = min(X.shape[0], X.shape[1])
        if 50 >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = 50
    if isinstance(X, DaskArray):
        if chunked:
            raise ValueError(
                "Dask arrays are not supported for chunked PCA computation."
            )
        _check_gpu_X(X, allow_dask=True)
        if not zero_center:
            raise ValueError("Dask arrays do not support non-zero centered PCA.")
        if isinstance(X._meta, cp.ndarray):
            from cuml.dask.decomposition import PCA

            if svd_solver == "auto":
                svd_solver = "jacobi"
            pca_func = PCA(n_components=n_comps, svd_solver=svd_solver, whiten=False)
            X_pca = pca_func.fit_transform(X)
            # cuml-issue #5883
            X_pca = X_pca.compute_chunk_sizes()
        elif isinstance(X._meta, csr_matrix):
            from ._sparse_pca._dask_sparse_pca import PCA_sparse_dask

            pca_func = PCA_sparse_dask(n_components=n_comps)
            pca_func = pca_func.fit(X)
            X_pca = pca_func.transform(X)

    elif zero_center:
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
        elif cpissparse(X) or issparse(X):
            if issparse(X):
                X = sparse_scipy_to_cp(X, dtype=X.dtype)
            from ._sparse_pca._sparse_pca import PCA_sparse

            if not isspmatrix_csr(X):
                X = X.tocsr()
            pca_func = PCA_sparse(n_components=n_comps)
            X_pca = pca_func.fit_transform(X)
        else:
            from cuml.decomposition import PCA

            pca_func = PCA(
                n_components=n_comps,
                svd_solver=svd_solver,
                random_state=random_state,
                output_type="numpy",
            )
            X_pca = pca_func.fit_transform(X)

    else:  # not zero_center
        from cuml.decomposition import TruncatedSVD

        pca_func = TruncatedSVD(
            n_components=n_comps,
            random_state=random_state,
            algorithm=svd_solver,
            output_type="numpy",
        )
        X_pca = pca_func.fit_transform(X)

    if X_pca.dtype.descr != np.dtype(dtype).descr:
        X_pca = X_pca.astype(dtype)

    key_obsm, key_varm, key_uns = (
        ("X_pca", "PCs", "pca") if key_added is None else [key_added] * 3
    )
    adata.obsm[key_obsm] = X_pca
    adata.uns[key_uns] = {
        "params": {
            "zero_center": zero_center,
            "use_highly_variable": mask_var_param == "highly_variable",
            "mask_var": mask_var_param,
            **({"layer": layer} if layer is not None else {}),
        },
        "variance": _as_numpy(pca_func.explained_variance_),
        "variance_ratio": _as_numpy(pca_func.explained_variance_ratio_),
    }
    if mask_var is not None:
        adata.varm[key_varm] = np.zeros(shape=(adata.n_vars, n_comps))
        adata.varm[key_varm][mask_var] = _as_numpy(pca_func.components_.T)
    else:
        adata.varm[key_varm] = _as_numpy(pca_func.components_.T)

    if copy:
        return adata


def _as_numpy(X):
    if isinstance(X, cp.ndarray):
        return X.get()
    else:
        return X
