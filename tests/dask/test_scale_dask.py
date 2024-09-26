from __future__ import annotations

import cupy as cp
import numpy as np
import scanpy as sc
from conftest import as_dense_cupy_dask_array, as_sparse_cupy_dask_array
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k

import rapids_singlecell as rsc


def _get_anndata():
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)
    return adata.copy()


def test_zc_sparse(client):
    adata = _get_anndata()
    mask = np.random.randint(0, 2, adata.shape[0], dtype=np.bool_)
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X.astype(np.float64))
    adata.X = cusparse.csr_matrix(adata.X.astype(np.float64))
    rsc.pp.scale(adata, mask_obs=mask, max_value=10)
    rsc.pp.scale(dask_data, mask_obs=mask, max_value=10)
    cp.testing.assert_allclose(adata.X, dask_data.X.compute())


def test_nzc_sparse(client):
    adata = _get_anndata()
    mask = np.random.randint(0, 2, adata.shape[0], dtype=np.bool_)
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.scale(adata, zero_center=False, mask_obs=mask, max_value=10)
    rsc.pp.scale(dask_data, zero_center=False, mask_obs=mask, max_value=10)
    cp.testing.assert_allclose(adata.X.toarray(), dask_data.X.compute().toarray())


def test_zc_dense(client):
    adata = _get_anndata()
    mask = np.random.randint(0, 2, adata.shape[0], dtype=np.bool_)
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X.astype(np.float64))
    adata.X = cp.array(adata.X.toarray().astype(np.float64))
    rsc.pp.scale(adata, mask_obs=mask, max_value=10)
    rsc.pp.scale(dask_data, mask_obs=mask, max_value=10)
    cp.testing.assert_allclose(adata.X, dask_data.X.compute())


def test_nzc_dense(client):
    adata = _get_anndata()
    mask = np.random.randint(0, 2, adata.shape[0], dtype=np.bool_)
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X.astype(np.float64))
    adata.X = cp.array(adata.X.toarray().astype(np.float64))
    rsc.pp.scale(adata, zero_center=False, mask_obs=mask, max_value=10)
    rsc.pp.scale(dask_data, zero_center=False, mask_obs=mask, max_value=10)
    cp.testing.assert_allclose(adata.X, dask_data.X.compute())
