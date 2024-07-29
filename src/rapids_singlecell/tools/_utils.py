from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy.sparse import issparse, isspmatrix_csc, isspmatrix_csr

from . import pca


def _choose_representation(adata, use_rep=None, n_pcs=None):
    if use_rep is None and n_pcs == 0:  # backwards compat for specifying `.X`
        use_rep = "X"
    if use_rep is None:
        if adata.n_vars > 50:
            if "X_pca" in adata.obsm.keys():
                if n_pcs is not None and n_pcs > adata.obsm["X_pca"].shape[1]:
                    raise ValueError(
                        "`X_pca` does not have enough PCs. Rerun `rsc.pp.pca` with adjusted `n_comps`."
                    )
                X = adata.obsm["X_pca"][:, :n_pcs]
            else:
                n_pcs_pca = n_pcs if n_pcs is not None else 50

                pca(adata, n_comps=n_pcs_pca)
                X = adata.obsm["X_pca"][:, :n_pcs]
        else:
            X = adata.X
    else:
        if use_rep in adata.obsm.keys() and n_pcs is not None:
            if n_pcs > adata.obsm[use_rep].shape[1]:
                raise ValueError(
                    f"{use_rep} does not have enough Dimensions. Provide a "
                    "Representation with equal or more dimensions than"
                    "`n_pcs` or lower `n_pcs` "
                )
            X = adata.obsm[use_rep][:, :n_pcs]
        elif use_rep in adata.obsm.keys() and n_pcs is None:
            X = adata.obsm[use_rep]
        elif use_rep == "X":
            X = adata.X
        else:
            raise ValueError(
                f"Did not find {use_rep} in `.obsm.keys()`. "
                "You need to compute it first."
            )
    return X


def _nan_mean_minor(X, major, minor, *, mask=None, n_features=None):
    from ._kernels._nan_mean_kernels import _get_nan_mean_minor

    mean = cp.zeros(minor, dtype=cp.float64)
    nans = cp.zeros(minor, dtype=cp.int32)
    tpb = (32,)
    bpg_x = math.ceil(X.nnz / 32)

    bpg = (bpg_x,)
    get_mean_var_minor = _get_nan_mean_minor(X.data.dtype)
    get_mean_var_minor(bpg, tpb, (X.indices, X.data, mean, nans, mask, X.nnz))
    mean /= n_features - nans
    return mean


def _nan_mean_major(X, major, minor, *, mask=None, n_features=None):
    from ._kernels._nan_mean_kernels import _get_nan_mean_major

    mean = cp.zeros(major, dtype=cp.float64)
    nans = cp.zeros(major, dtype=cp.int32)
    block = (64,)
    grid = (major,)
    get_mean_var_major = _get_nan_mean_major(X.data.dtype)
    get_mean_var_major(
        grid, block, (X.indptr, X.indices, X.data, mean, nans, mask, major, minor)
    )
    mean /= n_features - nans

    return mean


def _nan_mean(X, axis=0, *, mask=None, n_features=None):
    if issparse(X):
        if axis == 0:
            if isspmatrix_csr(X):
                major = X.shape[0]
                minor = X.shape[1]
                n_features = major
                if mask is None:
                    mask = cp.ones(X.shape[1], dtype=cp.bool_)
                mean = _nan_mean_minor(
                    X, major, minor, mask=mask, n_features=n_features
                )
            elif isspmatrix_csc(X):
                if mask is not None:
                    X = X[:, mask]
                major = X.shape[1]
                minor = X.shape[0]
                mask = cp.ones(X.shape[0], dtype=cp.bool_)
                n_features = minor
                mean = _nan_mean_major(
                    X, major, minor, mask=mask, n_features=n_features
                )
        elif axis == 1:
            if isspmatrix_csr(X):
                major = X.shape[0]
                minor = X.shape[1]
                if mask is None:
                    mask = cp.ones(X.shape[1], dtype=cp.bool_)
                n_features = minor if n_features is None else n_features
                mean = _nan_mean_major(
                    X, major, minor, mask=mask, n_features=n_features
                )
            elif isspmatrix_csc(X):
                if mask is not None:
                    X = X[:, mask]
                major = X.shape[1]
                minor = X.shape[0]
                mask = cp.ones(X.shape[0], dtype=cp.bool_)
                n_features = major
                mean = _nan_mean_minor(
                    X, major, minor, mask=mask, n_features=n_features
                )
    else:
        if mask is None:
            mask = cp.ones(X.shape[1], dtype=cp.bool_)
        mean = cp.nanmean(X[:, mask], axis=axis, dtype=cp.float64)
    return mean
