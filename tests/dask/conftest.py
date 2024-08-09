from __future__ import annotations

import cupy as cp
from cupyx.scipy import sparse as cusparse
from anndata.tests.helpers import as_dense_dask_array, as_sparse_dask_array
import pytest

from dask_cuda import LocalCUDACluster
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny
from dask.distributed import Client


def as_sparse_cupy_dask_array(X):
    da = as_sparse_dask_array(X)
    da = da.rechunk((da.shape[0]//2, da.shape[1]))
    da = da.map_blocks(cusparse.csr_matrix, dtype = X.dtype)
    return da

def as_dense_cupy_dask_array(X):
    X = as_dense_dask_array(X)
    X = X.map_blocks(cp.array)
    X = X.rechunk((X.shape[0]//2, X.shape[1]))
    return X

from dask_cuda import initialize
from dask_cuda import LocalCUDACluster
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny
from dask.distributed import Client

@pytest.fixture(scope="module")
def cluster():

    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES ="0",
        protocol="tcp",
        scheduler_port=0,
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def client(cluster):

    client = Client(cluster)
    yield client
    client.close()
