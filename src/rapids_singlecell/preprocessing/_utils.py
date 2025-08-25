from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import issparse, isspmatrix_csc, isspmatrix_csr, spmatrix
from natsort import natsorted
from pandas.api.types import infer_dtype

from rapids_singlecell._compat import DaskArray

if TYPE_CHECKING:
    from anndata import AnnData

    from rapids_singlecell._utils import AnyRandom


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


def _sanitize_column(adata: AnnData, column: str):
    dont_sanitize = adata.is_view or adata.isbacked
    if infer_dtype(adata.obs[column]) == "string":
        c = pd.Categorical(adata.obs[column])
        sorted_categories = natsorted(c.categories)
        if not np.array_equal(c.categories, sorted_categories):
            c = c.reorder_categories(sorted_categories)
        if dont_sanitize:
            msg = (
                "Please call `.strings_to_categoricals()` on full "
                "AnnData, not on this view. You might encounter this"
                "error message while copying or writing to disk."
            )
            raise RuntimeError(msg)
        adata.obs[column] = c


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
    from ._kernels._mean_var_kernel import _get_mean_var_minor_fast

    mean = cp.zeros(minor, dtype=cp.float64)
    var = cp.zeros(minor, dtype=cp.float64)
    block = 256
    sm = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["multiProcessorCount"]
    grid = (min(max((X.nnz + block - 1) // block, sm * 4), 65535),)
    shmem_bytes = 1024 * 4 + 1024 * 8 * 2  # keys + two double arrays

    get_mean_var_minor = _get_mean_var_minor_fast(X.data.dtype)
    get_mean_var_minor(
        grid, (block,), (X.nnz, X.indices, X.data, mean, var), shared_mem=shmem_bytes
    )
    mean /= major
    var /= major
    var -= mean**2
    var *= major / (major - 1)
    return mean, var


def _mean_var_minor_dask(X, major, minor):
    """
    Implements sum operation for dask array when the backend is cupy sparse csr matrix
    """

    from rapids_singlecell.preprocessing._kernels._mean_var_kernel import (
        _get_mean_var_minor,
    )

    get_mean_var_minor = _get_mean_var_minor(X.dtype)
    get_mean_var_minor.compile()

    def __mean_var(X_part):
        mean = cp.zeros(minor, dtype=cp.float64)
        var = cp.zeros(minor, dtype=cp.float64)
        block = (32,)
        grid = (int(math.ceil(X_part.nnz / block[0])),)
        get_mean_var_minor(
            grid, block, (X_part.indices, X_part.data, mean, var, major, X_part.nnz)
        )
        return cp.vstack([mean, var])[None, ...]  # new axis for summing

    n_blocks = X.blocks.size
    mean, var = X.map_blocks(
        __mean_var,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (2,), (minor,)),
        dtype=cp.float64,
        meta=cp.array([]),
    ).sum(axis=0)
    var = (var - mean**2) * (major / (major - 1))
    return mean, var


# todo: Implement this dynamically for csc matrix as well
def _mean_var_major_dask(X, major, minor):
    """
    Implements sum operation for dask array when the backend is cupy sparse csr matrix
    """

    from rapids_singlecell.preprocessing._kernels._mean_var_kernel import (
        _get_mean_var_major,
    )

    get_mean_var_major = _get_mean_var_major(X.dtype)
    get_mean_var_major.compile()

    def __mean_var(X_part):
        mean = cp.zeros(X_part.shape[0], dtype=cp.float64)
        var = cp.zeros(X_part.shape[0], dtype=cp.float64)
        block = (64,)
        grid = (X_part.shape[0],)
        get_mean_var_major(
            grid,
            block,
            (
                X_part.indptr,
                X_part.indices,
                X_part.data,
                mean,
                var,
                X_part.shape[0],
                minor,
            ),
        )
        return cp.stack([mean, var], axis=1)

    output = X.map_blocks(
        __mean_var,
        chunks=(X.chunks[0], (2,)),
        dtype=cp.float64,
        meta=cp.array([]),
    )
    mean = output[:, 0]
    var = output[:, 1]
    mean = mean / minor
    var = var / minor
    var -= mean**2
    var *= minor / (minor - 1)
    return mean, var


def _mean_var_dense_dask(X, axis):
    """
    Implements sum operation for dask array when the backend is cupy dense matrix
    """
    from ._kernels._mean_var_kernel import mean_sum, sq_sum

    def __mean_var(X_part):
        var = sq_sum(X_part, axis=axis)
        mean = mean_sum(X_part, axis=axis)
        if axis == 0:
            return cp.vstack([mean, var])[None, ...]
        else:
            return cp.stack([mean, var], axis=1)

    n_blocks = X.blocks.size
    mean_var = X.map_blocks(
        __mean_var,
        new_axis=(1,) if axis - 1 else None,
        chunks=(X.chunks[0], (2,)) if axis else ((1,) * n_blocks, (2,), (X.shape[1],)),
        dtype=cp.float64,
        meta=cp.array([]),
    )

    if axis == 0:
        mean, var = mean_var.sum(axis=0)
    else:
        mean = mean_var[:, 0]
        var = mean_var[:, 1]

    mean = mean / X.shape[axis]
    var = var / X.shape[axis]
    var -= mean**2
    var *= X.shape[axis] / (X.shape[axis] - 1)
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
            raise ValueError("axis must be either 0 or 1")
    elif isinstance(X, DaskArray):
        if isspmatrix_csr(X._meta):
            if axis == 0:
                major = X.shape[0]
                minor = X.shape[1]
                mean, var = _mean_var_minor_dask(X, major, minor)
            if axis == 1:
                major = X.shape[0]
                minor = X.shape[1]
                mean, var = _mean_var_major_dask(X, major, minor)
        elif isinstance(X._meta, cp.ndarray):
            mean, var = _mean_var_dense_dask(X, axis)
        else:
            raise ValueError(
                "Type not supported. Please provide a CuPy ndarray or a CuPy sparse matrix. Or a Dask array with a CuPy ndarray or a CuPy sparse matrix as meta."
            )
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


def _check_gpu_X(X, *, require_cf=False, allow_dask=False, allow_csc=True):
    if isinstance(X, DaskArray):
        if allow_dask:
            return _check_gpu_X(X._meta, allow_csc=False)
        else:
            raise TypeError(
                "The input is a DaskArray. "
                "Rapids-singlecell doesn't support DaskArray in this function, "
                "so your input must be a CuPy ndarray or a CuPy sparse matrix. "
            )
    elif isinstance(X, cp.ndarray):
        return True
    elif isspmatrix_csc(X) or isspmatrix_csr(X):
        if not allow_csc and isspmatrix_csc(X):
            raise TypeError(
                "When using Dask, only CuPy ndarrays and CSR matrices are supported as "
                "meta arrays. Please convert your data to CSR format if it is in CSC."
            )
        elif not require_cf:
            return True
        elif X.has_canonical_format:
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


def _check_use_raw(adata: AnnData, layer: str | None, *, use_raw: None | bool) -> bool:
    """
    Normalize checking `use_raw`.

    My intentention here is to also provide a single place to throw a deprecation warning from in future.
    """
    if use_raw is not None:
        return use_raw
    if layer is not None:
        return False
    return adata.raw is not None


def get_random_state(seed: AnyRandom) -> np.random.RandomState:
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)
