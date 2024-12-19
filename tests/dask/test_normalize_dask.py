from __future__ import annotations

import cupy as cp
import pytest
import scanpy as sc
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k

import rapids_singlecell as rsc
from rapids_singlecell._testing import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_normalize_total(client, data_kind):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    dask_data = adata.copy()

    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
        adata.X = cusparse.csr_matrix(adata.X)
    elif data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)
        adata.X = cp.array(adata.X.toarray())
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.normalize_total(adata)
    rsc.pp.normalize_total(dask_data)

    if data_kind == "sparse":
        adata_X = adata.X.toarray()
        dask_X = dask_data.X.compute().toarray()
    else:
        adata_X = adata.X
        dask_X = dask_data.X.compute()

    cp.testing.assert_allclose(adata_X, dask_X)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_log1p(client, data_kind):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    dask_data = adata.copy()

    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
        adata.X = cusparse.csr_matrix(adata.X)
    elif data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)
        adata.X = cp.array(adata.X.toarray())
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.log1p(adata)
    rsc.pp.log1p(dask_data)

    if data_kind == "sparse":
        adata_X = adata.X.toarray()
        dask_X = dask_data.X.compute().toarray()
    else:
        adata_X = adata.X
        dask_X = dask_data.X.compute()

    cp.testing.assert_allclose(adata_X, dask_X)
