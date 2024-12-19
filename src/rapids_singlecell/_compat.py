from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix
from dask.array import Array as DaskArray  # noqa: F401
from scipy.sparse import csc_matrix as csc_matrix_cpu
from scipy.sparse import csr_matrix as csr_matrix_cpu


def _meta_dense(dtype):
    return cp.zeros([0], dtype=dtype)


def _meta_sparse(dtype):
    return csr_matrix(cp.array((1.0,), dtype=dtype))


def _meta_dense(dtype):
    return cp.zeros([0], dtype=dtype)


def _meta_sparse(dtype):
    return csr_matrix(cp.array((1.0,), dtype=dtype))


def _meta_dense_cpu(dtype):
    return np.zeros([0], dtype=dtype)


def _meta_sparse_csr_cpu(dtype):
    return csr_matrix_cpu(np.array((1.0,), dtype=dtype))


def _meta_sparse_csc_cpu(dtype):
    return csc_matrix_cpu(np.array((1.0,), dtype=dtype))
