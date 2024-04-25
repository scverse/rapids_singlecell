from __future__ import annotations

import cupy as cp
from cupyx.scipy.sparse import csr_matrix

try:
    from dask.array import Array as DaskArray
except ImportError:

    class DaskArray:
        pass


try:
    from dask.distributed import Client as DaskClient
except ImportError:

    class DaskClient:
        pass


def _get_dask_client(client=None):
    from dask.distributed import default_client

    if client is None or not isinstance(client, DaskClient):
        return default_client()
    return client


def _meta_dense(dtype):
    return cp.zeros([0], dtype=dtype)


def _meta_sparse(dtype):
    return csr_matrix(cp.array((1.0,), dtype=dtype))
