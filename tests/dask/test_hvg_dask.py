from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cusparse
import scanpy as sc
import pandas as pd
from conftest import as_dense_cupy_dask_array, as_sparse_cupy_dask_array
import rapids_singlecell as rsc

from scanpy.datasets import pbmc3k

def _get_anndata():
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata

def test_seurat_sparse(client):
    adata = _get_anndata()
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.highly_variable_genes(adata)
    rsc.pp.highly_variable_genes(dask_data)
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(adata.var["dispersions_norm"], dask_data.var["dispersions_norm"])


def test_seurat_sparse_batch(client):
    adata = _get_anndata()
    adata.obs["batch"] = (
        "source_" + pd.array([*range(1, 6), 5]).repeat(500).astype("string")
    )[: adata.n_obs]
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.highly_variable_genes(adata, batch_key="batch")
    rsc.pp.highly_variable_genes(dask_data,batch_key="batch")
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(adata.var["dispersions_norm"], dask_data.var["dispersions_norm"])

def test_cr_sparse(client):
    adata = _get_anndata()
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.highly_variable_genes(adata, flavor="cell_ranger")
    rsc.pp.highly_variable_genes(dask_data, flavor="cell_ranger")
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(adata.var["dispersions_norm"], dask_data.var["dispersions_norm"])

def test_cr_sparse_batch(client):
    adata = _get_anndata()
    adata.obs["batch"] = (
        "source_" + pd.array([*range(1, 6), 5]).repeat(500).astype("string")
    )[: adata.n_obs]
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.highly_variable_genes(adata, batch_key="batch", flavor="cell_ranger")
    rsc.pp.highly_variable_genes(dask_data,batch_key="batch", flavor="cell_ranger")
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(adata.var["dispersions_norm"], dask_data.var["dispersions_norm"])

def test_cr_dense(client):
    adata = _get_anndata()
    adata.X = adata.X.astype("float64")
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
    adata.X = cp.array(adata.X.toarray())
    rsc.pp.highly_variable_genes(adata, flavor="cell_ranger")
    rsc.pp.highly_variable_genes(dask_data, flavor="cell_ranger")
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(adata.var["dispersions_norm"], dask_data.var["dispersions_norm"])

def test_seurat_dense(client):
    adata = _get_anndata()
    adata.X = adata.X.astype("float64")
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
    adata.X = cp.array(adata.X.toarray())
    rsc.pp.highly_variable_genes(adata)
    rsc.pp.highly_variable_genes(dask_data)
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(adata.var["dispersions_norm"], dask_data.var["dispersions_norm"])


def test_cr_dense_batch(client):
    adata = _get_anndata()
    adata.obs["batch"] = (
        "source_" + pd.array([*range(1, 6), 5]).repeat(500).astype("string")
    )[: adata.n_obs]
    adata.X = adata.X.astype("float64")
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
    adata.X = cp.array(adata.X.toarray())
    rsc.pp.highly_variable_genes(adata, batch_key="batch", flavor="cell_ranger")
    rsc.pp.highly_variable_genes(dask_data,batch_key="batch", flavor="cell_ranger")
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(adata.var["dispersions_norm"], dask_data.var["dispersions_norm"])

def test_seurat_dense_batch(client):
    adata = _get_anndata()
    adata.obs["batch"] = (
        "source_" + pd.array([*range(1, 6), 5]).repeat(500).astype("string")
    )[: adata.n_obs]
    adata.X = adata.X.astype("float64")
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
    adata.X =  cp.array(adata.X.toarray())
    rsc.pp.highly_variable_genes(adata, batch_key="batch")
    rsc.pp.highly_variable_genes(dask_data,batch_key="batch")
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(adata.var["dispersions_norm"], dask_data.var["dispersions_norm"])
