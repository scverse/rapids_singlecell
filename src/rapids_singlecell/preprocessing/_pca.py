from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cuml.internals.input_utils import sparse_scipy_to_cp
from cupyx.scipy.sparse import issparse as cpissparse
from cupyx.scipy.sparse import issparse as issparse_cupy
from cupyx.scipy.sparse import isspmatrix_csr
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
    **kwargs,
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
            Must be one of {'full', 'jacobi', 'auto', 'covariance_eigh', 'lanczos', 'randomized'}. \
            Defaults to 'auto'. Sparse matrices will use `'covariance_eigh'` or `'lanczos'`.

            .. note::
                For sparse matrices, `'covariance_eigh'` is recommended for fewer than ~10,000
                features (best around 2-5k). For larger feature counts (>10,000), use `'lanczos'`
                or `'randomized'` which scale better with the number of features.

            `'covariance_eigh'`
                Classic eigendecomposition of the covariance matrix. Fast for sparse matrices
                with fewer than ~10,000 features due to optimized GPU kernels. Good balance of
                speed and accuracy. Works with Dask arrays (must be CSR or dense, chunked
                as `(N, adata.shape[1])`).
            `'lanczos'`
                Lanczos bidiagonalization SVD with implicit restarts. Memory efficient for
                large sparse matrices with many features (>10,000). Faster than covariance_eigh
                when n_vars is large and n_comps is small relative to n_vars. Does not support
                Dask arrays. Best singular value accuracy (exact for converged components),
                but orthogonality of components may degrade for high n_comps due to accumulated
                rounding errors in the iterative process.
            `'randomized'`
                Randomized SVD based on Halko et al. (2009) with GPU-optimized CholeskyQR2
                orthogonalization from Tomás et al. (2024). Uses efficient GPU GEMM operations.
                Faster than Lanczos but approximate. Excellent orthogonality of components
                (uses explicit reorthogonalization), but singular values are approximate.
                Accuracy depends on the number of power iterations (default 2). Does not
                support Dask arrays.
            `'full'`
                From `cuml` for dense arrays uses a eigendecomposition of the covariance matrix then discards components.
            `'jacobi'`
                From `cuml` for dense arrays. Jacobi is much faster as it iteratively corrects, but is less accurate.
            `'auto'`
                Automatically chooses the best solver based on the shape of the data matrix.
                For sparse matrices: uses `'lanczos'` when n_vars > 10,000 and n_comps < n_vars/10,
                otherwise uses `'covariance_eigh'`. For dense dask arrays uses `'covariance_eigh'`.

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

        **kwargs
            Additional arguments for specific SVD solvers.
            For ``svd_solver='randomized'``:

            - ``n_oversamples`` : Extra random vectors for better approximation.
              Higher values improve accuracy. Default is 10.
            - ``n_iter`` : Number of power iterations. Higher values improve
              accuracy for matrices with slowly decaying singular values. Default is 2.

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

    mask_var_param, mask_var = _handle_mask_var(
        adata, mask_var, use_highly_variable=use_highly_variable
    )
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

    # Auto-select sparse solver based on matrix dimensions
    # Lanczos is faster when n_vars is large (>10000) and k is small
    # Covariance is faster for smaller matrices due to optimized kernels
    if svd_solver == "auto" and (cpissparse(X) or issparse(X)):
        n_vars = X.shape[1]
        if n_vars > 10000 and n_comps < n_vars // 10:
            svd_solver = "lanczos"
        else:
            svd_solver = "covariance_eigh"

    if isinstance(X, DaskArray):
        if chunked:
            raise ValueError(
                "Dask arrays are not supported for chunked PCA computation."
            )
        _check_gpu_X(X, allow_dask=True)
        if svd_solver in ("lanczos", "randomized"):
            raise NotImplementedError(
                f"'{svd_solver}' SVD solver does not support Dask arrays. "
                "Use svd_solver='covariance_eigh' instead."
            )
        if svd_solver == "auto":
            svd_solver = "covariance_eigh"
        if (
            isinstance(X._meta, cp.ndarray)
            and svd_solver != "covariance_eigh"
            and zero_center
        ):
            pca_func, X_pca = _run_cuml_pca_dask(X, n_comps, svd_solver=svd_solver)
        else:
            pca_func, X_pca = _run_covariance_pca(X, n_comps, zero_center)

    elif zero_center:
        if chunked:
            pca_func, X_pca = _run_chunked_pca(X, n_comps, chunk_size)
        elif cpissparse(X) or issparse(X):
            if svd_solver in ("lanczos", "randomized"):
                pca_func, X_pca = _run_sparse_svd_pca(
                    X,
                    n_comps,
                    svd_solver,
                    zero_center=zero_center,
                    random_state=random_state,
                    n_oversamples=kwargs.get("n_oversamples"),
                    n_iter=kwargs.get("n_iter"),
                )
            else:
                pca_func, X_pca = _run_covariance_pca(X, n_comps, zero_center)
        else:
            pca_func, X_pca = _run_cuml_pca(X, n_comps, svd_solver=svd_solver)

    else:  # not zero_center
        if cpissparse(X) or issparse(X):
            if svd_solver in ("lanczos", "randomized"):
                pca_func, X_pca = _run_sparse_svd_pca(
                    X,
                    n_comps,
                    svd_solver,
                    zero_center=zero_center,
                    random_state=random_state,
                    n_oversamples=kwargs.get("n_oversamples"),
                    n_iter=kwargs.get("n_iter"),
                )
            else:
                pca_func, X_pca = _run_covariance_pca(X, n_comps, zero_center)
        else:
            pca_func, X_pca = _run_cuml_tsvd(X, n_comps, svd_solver=svd_solver)

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


def _run_covariance_pca(X, n_comps, zero_center):
    """Run PCA using covariance matrix eigendecomposition."""
    if issparse(X):
        X = sparse_scipy_to_cp(X, dtype=X.dtype)
    from ._sparse_pca._sparse_pca import PCA_sparse

    if not isspmatrix_csr(X) and not isinstance(X, DaskArray):
        X = X.tocsr()
    if issparse_cupy(X):
        X.sort_indices()

    pca_func = PCA_sparse(n_components=n_comps, zero_center=zero_center)
    X_pca = pca_func.fit_transform(X)
    return pca_func, X_pca


def _run_sparse_svd_pca(
    X,
    n_comps,
    svd_solver,
    *,
    zero_center: bool = True,
    random_state: int = 0,
    n_oversamples: int | None = None,
    n_iter: int | None = None,
):
    """Run PCA using SVD solvers (lanczos, randomized)."""
    if issparse(X):
        X = sparse_scipy_to_cp(X, dtype=X.dtype)
    from ._sparse_pca._sparse_svd_pca import PCA_sparse_svd

    if not isspmatrix_csr(X):
        X = X.tocsr()
    if issparse_cupy(X):
        X.sort_indices()

    # Build kwargs for PCA_sparse_svd
    kwargs = {
        "n_components": n_comps,
        "svd_solver": svd_solver,
        "zero_center": zero_center,
        "random_state": random_state,
    }
    if n_oversamples is not None:
        kwargs["n_oversamples"] = n_oversamples
    if n_iter is not None:
        kwargs["n_iter"] = n_iter

    pca_func = PCA_sparse_svd(**kwargs)
    X_pca = pca_func.fit_transform(X)
    return pca_func, X_pca


def _run_chunked_pca(X, n_comps, chunk_size):
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

    return pca_func, X_pca


def _run_cuml_pca(X, n_comps, *, svd_solver: str):
    from cuml.decomposition import PCA

    if svd_solver == "covariance_eigh":
        svd_solver = "auto"
    pca_func = PCA(
        n_components=n_comps,
        svd_solver=svd_solver,
        output_type="numpy",
    )
    X_pca = pca_func.fit_transform(X)
    return pca_func, X_pca


def _run_cuml_tsvd(X, n_comps, *, svd_solver: str = "auto"):
    from cuml.decomposition import TruncatedSVD

    pca_func = TruncatedSVD(
        n_components=n_comps,
        algorithm=svd_solver,
        output_type="numpy",
    )
    X_pca = pca_func.fit_transform(X)
    return pca_func, X_pca


def _run_cuml_pca_dask(X, n_comps, *, svd_solver: str):
    from cuml.dask.decomposition import PCA

    pca_func = PCA(n_components=n_comps, svd_solver=svd_solver, whiten=False)
    X_pca = pca_func.fit_transform(X)
    # cuml-issue #5883
    X_pca = X_pca.compute_chunk_sizes()
    return pca_func, X_pca
