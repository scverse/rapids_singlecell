from __future__ import annotations

import math

import cupy as cp
from cupyx.scipy.sparse import issparse, isspmatrix_csc, isspmatrix_csr

from rapids_singlecell._compat import DaskArray

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


def _nan_mean_minor_dask_sparse(X, major, minor, *, mask=None, n_features=None):
    from ._kernels._nan_mean_kernels import _get_nan_mean_minor

    kernel = _get_nan_mean_minor(X.dtype)
    kernel.compile()

    def __nan_mean_minor(X_part):
        mean = cp.zeros(minor, dtype=cp.float64)
        nans = cp.zeros(minor, dtype=cp.int32)
        tpb = (32,)
        bpg_x = math.ceil(X_part.nnz / 32)
        bpg = (bpg_x,)
        kernel(bpg, tpb, (X_part.indices, X_part.data, mean, nans, mask, X_part.nnz))
        return cp.vstack([mean, nans.astype(cp.float64)])[None, ...]

    n_blocks = X.blocks.size
    mean, nans = X.map_blocks(
        __nan_mean_minor,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (2,), (minor,)),
        dtype=cp.float64,
        meta=cp.array([]),
    ).sum(axis=0)
    mean /= n_features - nans
    return mean


def _nan_mean_major_dask_sparse(X, major, minor, *, mask=None, n_features=None):
    from ._kernels._nan_mean_kernels import _get_nan_mean_major

    kernel = _get_nan_mean_major(X.dtype)
    kernel.compile()

    def __nan_mean_major(X_part):
        major_part = X_part.shape[0]
        mean = cp.zeros(major_part, dtype=cp.float64)
        nans = cp.zeros(major_part, dtype=cp.int32)
        block = (64,)
        grid = (major_part,)
        kernel(
            grid,
            block,
            (
                X_part.indptr,
                X_part.indices,
                X_part.data,
                mean,
                nans,
                mask,
                major_part,
                minor,
            ),
        )
        return cp.stack([mean, nans.astype(cp.float64)], axis=1)

    output = X.map_blocks(
        __nan_mean_major,
        chunks=(X.chunks[0], (2,)),
        dtype=cp.float64,
        meta=cp.array([]),
    )
    mean = output[:, 0]
    nans = output[:, 1]
    mean /= n_features - nans
    return mean


def _nan_mean_dense_dask(X, axis, *, mask, n_features):
    def __nan_mean_dense(X_part):
        X_to_use = X_part[:, mask].astype(cp.float64)
        sum = cp.nansum(X_to_use, axis=axis).ravel()
        nans = cp.sum(cp.isnan(X_to_use), axis=axis).ravel()
        if axis == 1:
            return cp.stack([sum, nans.astype(cp.float64)], axis=1)
        else:
            return cp.vstack([sum, nans.astype(cp.float64)])[None, ...]

    n_blocks = X.blocks.size
    output = X.map_blocks(
        __nan_mean_dense,
        new_axis=(1,) if axis - 1 else None,
        chunks=(X.chunks[0], (2,)) if axis else ((1,) * n_blocks, (2,), (X.shape[1],)),
        dtype=cp.float64,
        meta=cp.array([]),
    )
    if axis == 0:
        mean, nans = output.sum(axis=0)
    else:
        mean = output[:, 0]
        nans = output[:, 1]
    mean /= n_features - nans
    return mean


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
    elif isinstance(X, DaskArray):
        if isspmatrix_csr(X._meta):
            major, minor = X.shape
            if mask is None:
                mask = cp.ones(X.shape[1], dtype=cp.bool_)
            if axis == 0:
                n_features = major
                mean = _nan_mean_minor_dask_sparse(
                    X, major, minor, mask=mask, n_features=n_features
                )
            elif axis == 1:
                n_features = minor if n_features is None else n_features
                mean = _nan_mean_major_dask_sparse(
                    X, major, minor, mask=mask, n_features=n_features
                )
            else:
                raise ValueError("axis must be either 0 or 1")
        elif isinstance(X._meta, cp.ndarray):
            if mask is None:
                mask = cp.ones(X.shape[1], dtype=cp.bool_)
            if n_features is None:
                n_features = X.shape[axis]
            mean = _nan_mean_dense_dask(X, axis, mask=mask, n_features=n_features)
            # raise NotImplementedError("Dask dense arrays are not supported yet")
        else:
            raise ValueError(
                "Type not supported. Please provide a CuPy ndarray or a CuPy sparse matrix. Or a Dask array with a CuPy ndarray or a CuPy sparse matrix as meta."
            )
    else:
        if mask is None:
            mask = cp.ones(X.shape[1], dtype=cp.bool_)
        mean = cp.nanmean(X[:, mask], axis=axis, dtype=cp.float64)
    return mean
