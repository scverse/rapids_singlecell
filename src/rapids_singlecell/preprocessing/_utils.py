import math

import cupy as cp
from cupyx.scipy.sparse import issparse, isspmatrix_csc, isspmatrix_csr


def _mean_var_major(X, major, minor):
    from ._kernels._mean_var_kernel import _get_mean_var_major

    mean = cp.zeros(major, dtype=cp.float64)
    var = cp.zeros(major, dtype=cp.float64)
    block = (32,)
    grid = (int(math.ceil(major / block[0])),)
    get_mean_var_major = _get_mean_var_major(X.data.dtype)
    get_mean_var_major(
        grid, block, (X.indptr, X.indices, X.data, mean, var, major, minor)
    )
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
            else:
                mean = X.mean(axis=0)
                var = X.var(axis=0)
                major = X.shape[1]
                var = (var - mean**2) * (major / (major - 1))
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
        mean = X.mean(axis=axis)
        var = X.var(axis=axis)
        var *= X.shape[axis] / (X.shape[axis] - 1)
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


def _check_gpu_X(X):
    if isinstance(X, cp.ndarray):
        return True
    elif issparse(X):
        return True
    else:
        raise TypeError(
            "The input is not a CuPy ndarray or CuPy sparse matrix. "
            "Rapids-singlecell only supports GPU matrices, "
            "so your input must be either a CuPy ndarray or a CuPy sparse matrix. "
            "If you're working with CPU-based matrices, please consider using Scanpy instead."
        )
