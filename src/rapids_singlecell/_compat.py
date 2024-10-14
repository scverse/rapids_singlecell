from __future__ import annotations

import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from dask.array import Array as DaskArray  # noqa: F401


def _meta_dense(dtype):
    return cp.zeros([0], dtype=dtype)


def _meta_sparse(dtype):
    return csr_matrix(cp.array((1.0,), dtype=dtype))
