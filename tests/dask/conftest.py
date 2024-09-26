from __future__ import annotations

import pytest
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny


@pytest.fixture(scope="module")
def cluster():
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0",
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
