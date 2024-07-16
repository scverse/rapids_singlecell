from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cusparse
from scipy import sparse
from conftest import as_dense_cupy_dask_array, as_sparse_cupy_dask_array
import rapids_singlecell as rsc
import scanpy as sc

from scanpy.datasets import pbmc3k

def test_zc_sparse(client):
    adata = _get_anndata()
    mask = np.random.randint(0,2,adata.shape[0],dtype = np.bool_)
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X.astype(np.float64))
    adata.X = cusparse.csr_matrix(adata.X.astype(np.float64))
    rsc.pp.scale(adata, mask_obs = mask, max_value = 10)
    rsc.pp.scale(dask_data, mask_obs = mask, max_value = 10)
    cp.testing.assert_allclose(adata.X, dask_data.X.compute())

def test_nzc_sparse(client):
    adata = _get_anndata()
    mask = np.random.randint(0,2,adata.shape[0],dtype = np.bool_)
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.scale(adata, zero_center = False, mask_obs = mask, max_value = 10)
    rsc.pp.scale(dask_data,zero_center = False, mask_obs = mask, max_value = 10)
    cp.testing.assert_allclose(adata.X.toarray(), dask_data.X.compute().toarray())

def test_zc_dense(client):
    adata = _get_anndata()
    mask = np.random.randint(0,2,adata.shape[0],dtype = np.bool_)
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X.astype(np.float64))
    adata.X = cp.array(adata.X.toarray().astype(np.float64))
    rsc.pp.scale(adata, mask_obs = mask, max_value = 10)
    rsc.pp.scale(dask_data, mask_obs = mask, max_value = 10)
    cp.testing.assert_allclose(adata.X, dask_data.X.compute())

def test_nzc_dense(client):
    adata = _get_anndata()
    mask = np.random.randint(0,2,adata.shape[0],dtype = np.bool_)
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X.astype(np.float64))
    adata.X = cp.array(adata.X.toarray().astype(np.float64))
    rsc.pp.scale(adata, zero_center = False, mask_obs = mask, max_value = 10)
    rsc.pp.scale(dask_data, zero_center = False, mask_obs = mask, max_value = 10)
    cp.testing.assert_allclose(adata.X, dask_data.X.compute())
