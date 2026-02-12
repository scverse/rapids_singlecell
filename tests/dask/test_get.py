from __future__ import annotations

import cupy as cp
import dask.array as da
import numpy as np
import pytest
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scanpy.datasets import pbmc3k_processed
from scipy import sparse

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._utils import _check_gpu_X
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_get_anndata(client, data_kind):
    adata = pbmc3k_processed()
    dask_adata = adata.copy()
    if data_kind == "sparse":
        adata.X = rsc.get.X_to_GPU(sparse.csr_matrix(adata.X.astype(np.float64)))
        dask_adata.X = as_sparse_cupy_dask_array(dask_adata.X.astype(np.float64))
    elif data_kind == "dense":
        adata.X = cp.array(adata.X.astype(np.float64))
        dask_adata.X = as_dense_cupy_dask_array(dask_adata.X.astype(np.float64))
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    assert type(adata.X) is type(dask_adata.X._meta)

    if data_kind == "sparse":
        cp.testing.assert_array_equal(
            adata.X.toarray(), dask_adata.X.compute().toarray()
        )
    else:
        cp.testing.assert_array_equal(adata.X, dask_adata.X.compute())

    rsc.get.anndata_to_CPU(dask_adata)
    rsc.get.anndata_to_CPU(adata)

    assert type(adata.X) is type(dask_adata.X._meta)

    if data_kind == "sparse":
        cp.testing.assert_array_equal(
            adata.X.toarray(), dask_adata.X.compute().toarray()
        )
    else:
        cp.testing.assert_array_equal(adata.X, dask_adata.X.compute())
    rsc.get.anndata_to_GPU(dask_adata)
    rsc.get.anndata_to_GPU(adata)

    assert type(adata.X) is type(dask_adata.X._meta)

    if data_kind == "sparse":
        cp.testing.assert_array_equal(
            adata.X.toarray(), dask_adata.X.compute().toarray()
        )
    else:
        cp.testing.assert_array_equal(adata.X, dask_adata.X.compute())


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_check_gpu_X_column_chunked(data_kind):
    """Test that _check_gpu_X rejects Dask arrays chunked along the column axis."""
    X = np.random.default_rng(0).random((100, 50))
    if data_kind == "sparse":
        X_gpu = cp_csr_matrix(sparse.csr_matrix(X))
    else:
        X_gpu = cp.array(X)

    # Column-chunked: should raise
    col_chunked = da.from_array(X_gpu, chunks=(50, 25))
    with pytest.raises(ValueError, match="chunked along the column axis"):
        _check_gpu_X(col_chunked, allow_dask=True)

    # Row-chunked only: should pass
    row_chunked = da.from_array(X_gpu, chunks=(50, 50))
    _check_gpu_X(row_chunked, allow_dask=True)
