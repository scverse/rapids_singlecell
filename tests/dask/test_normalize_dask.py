from __future__ import annotations

import cupy as cp
import scanpy as sc
from conftest import as_dense_cupy_dask_array, as_sparse_cupy_dask_array
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k

import rapids_singlecell as rsc


def test_normalize_sparse(client):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.normalize_total(adata)
    rsc.pp.normalize_total(dask_data)
    cp.testing.assert_allclose(adata.X.toarray(), dask_data.X.compute().toarray())


def test_normalize_dense(client):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X)
    adata.X = cp.array(adata.X.toarray())
    rsc.pp.normalize_total(adata)
    rsc.pp.normalize_total(dask_data)
    cp.testing.assert_allclose(adata.X, dask_data.X.compute())


def test_log1p_sparse(client):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.log1p(adata)
    rsc.pp.log1p(dask_data)
    cp.testing.assert_allclose(adata.X.toarray(), dask_data.X.compute().toarray())


def test_log1p_dense(client):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X)
    adata.X = cp.array(adata.X.toarray())
    rsc.pp.log1p(adata)
    rsc.pp.log1p(dask_data)
    cp.testing.assert_allclose(adata.X, dask_data.X.compute())
