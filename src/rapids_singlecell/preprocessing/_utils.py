from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import cupy as cp
from cupyx.scipy.sparse import issparse, isspmatrix_csc, isspmatrix_csr, spmatrix

if TYPE_CHECKING:
    from anndata import AnnData


def _sparse_to_dense(X: spmatrix, order: Literal["C", "F"] | None = None) -> cp.ndarray:
    if order is None:
        order = "C"
    from ._kernels._sparse2dense import _sparse2densekernel

    if isspmatrix_csr(X):
        major, minor = X.shape[0], X.shape[1]
        switcher = 1 if order == "C" else 0
    elif isspmatrix_csc(X):
        major, minor = X.shape[1], X.shape[0]
        switcher = 0 if order == "C" else 1
    else:
        raise ValueError("Input matrix must be a sparse `csc` or `csr` matrix")
    sparse2dense = _sparse2densekernel(X.dtype)

    dense = cp.zeros(X.shape, order=order, dtype=X.dtype)
    max_nnz = cp.diff(X.indptr).max()
    tpb = (32, 32)
    bpg_x = math.ceil(major / tpb[0])
    bpg_y = math.ceil(max_nnz / tpb[1])
    bpg = (bpg_x, bpg_y)

    sparse2dense(
        bpg,
        tpb,
        (X.indptr, X.indices, X.data, dense, major, minor, switcher),
    )
    return dense


def _mean_var_major(X, major, minor):
    from ._kernels._mean_var_kernel import _get_mean_var_major

    mean = cp.zeros(major, dtype=cp.float64)
    var = cp.zeros(major, dtype=cp.float64)
    block = (64,)
    grid = (major,)
    get_mean_var_major = _get_mean_var_major(X.data.dtype)
    get_mean_var_major(
        grid, block, (X.indptr, X.indices, X.data, mean, var, major, minor)
    )
    mean = mean / minor
    var = var / minor
    var -= cp.power(mean, 2)
    var *= minor / (minor - 1)
    return mean, var


def _mean_var_minor(X, major, minor):
    from ._kernels._mean_var_kernel import _get_mean_var_minor

    mean = cp.zeros(minor, dtype=cp.float64)
    var = cp.zeros(minor, dtype=cp.float64)
    block = (32,)
    grid = (int(math.ceil(X.nnz / block[0])),)
    get_mean_var_minor = _get_mean_var_minor(X.data.dtype)
    get_mean_var_minor(grid, block, (X.indices, X.data, mean, var, major, X.nnz))

    var = (var - mean**2) * (major / (major - 1))
    return mean, var


def _mean_var_dense(X, axis):
    from ._kernels._mean_var_kernel import mean_sum, sq_sum

    var = sq_sum(X, axis=axis)
    mean = mean_sum(X, axis=axis)
    mean = mean / X.shape[axis]
    var = var / X.shape[axis]
    var -= cp.power(mean, 2)
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def _get_mean_var(X, axis=0):
    if issparse(X):
        if axis == 0:
            if isspmatrix_csr(X):
                major = X.shape[0]
                minor = X.shape[1]
                mean, var = _mean_var_minor(X, major, minor)
            elif isspmatrix_csc(X):
                major = X.shape[1]
                minor = X.shape[0]
                mean, var = _mean_var_major(X, major, minor)
        elif axis == 1:
            if isspmatrix_csr(X):
                major = X.shape[0]
                minor = X.shape[1]
                mean, var = _mean_var_major(X, major, minor)
            elif isspmatrix_csc(X):
                major = X.shape[1]
                minor = X.shape[0]
                mean, var = _mean_var_minor(X, major, minor)
    else:
        mean, var = _mean_var_dense(X, axis)
    return mean, var


def _check_nonnegative_integers(X):
    if issparse(X):
        data = X.data
    else:
        data = X
    """Checks values of data to ensure it is count data"""
    # Check no negatives
    if cp.signbit(data).any():
        return False
    elif cp.any(~cp.equal(cp.mod(data, 1), 0)):
        return False
    else:
        return True


def _check_gpu_X(X, require_cf=False):
    if isinstance(X, cp.ndarray):
        return True
    elif issparse(X):
        if X.has_canonical_format or not require_cf:
            return True
        else:
            X.sort_indices()
            X.sum_duplicates()
    else:
        raise TypeError(
            "The input is not a CuPy ndarray or CuPy sparse matrix. "
            "Rapids-singlecell only supports GPU matrices in this function, "
            "so your input must be either a CuPy ndarray or a CuPy sparse matrix. "
            "Please checkout `rapids_singlecell.get.anndata_to_GPU` to convert your data to GPU. "
            "If you're working with CPU-based matrices, please consider using Scanpy instead."
        )


def _check_use_raw(adata: AnnData, use_raw: None | bool, layer: str | None) -> bool:
    """
    Normalize checking `use_raw`.

    My intentention here is to also provide a single place to throw a deprecation warning from in future.
    """
    if use_raw is not None:
        return use_raw
    if layer is not None:
        return False
    return adata.raw is not None
