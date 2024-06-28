from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cusparse
from conftest import as_dense_cupy_dask_array, as_sparse_cupy_dask_array
import rapids_singlecell as rsc

from scanpy.datasets import pbmc3k_processed

def test_pca_sparse_dask(client):
    sparse_ad = pbmc3k_processed()
    default = pbmc3k_processed()
    sparse_ad.X = cusparse.csr_matrix(sparse_ad.X.astype(np.float64))
    default.X = as_sparse_cupy_dask_array(default.X.astype(np.float64))
    rsc.pp.pca(sparse_ad)
    rsc.pp.pca(default)

    cp.testing.assert_allclose(
        np.abs(sparse_ad.obsm["X_pca"]),
        cp.abs(default.obsm["X_pca"].compute()),
        rtol=1e-7,
        atol=1e-6,
    )

    cp.testing.assert_allclose(
        np.abs(sparse_ad.varm["PCs"]), np.abs(default.varm["PCs"]), rtol=1e-7, atol=1e-6
    )

    cp.testing.assert_allclose(
        np.abs(sparse_ad.uns["pca"]["variance_ratio"]),
        np.abs(default.uns["pca"]["variance_ratio"]),
        rtol=1e-7,
        atol=1e-6,
    )

def test_pca_sparse_dask_full_pipeline(client):
    sparse_ad = pbmc3k_processed()
    default = pbmc3k_processed()
    sparse_ad.X = cusparse.csr_matrix(sparse_ad.X.astype(np.float64))
    default.X = as_sparse_cupy_dask_array(default.X.astype(np.float64))

    rsc.pp.normalize_total(sparse_ad,  target_sum=1e4)
    rsc.pp.normalize_total(default,target_sum=1e4)

    rsc.pp.log1p(sparse_ad)
    rsc.pp.log1p(default)

    rsc.pp.pca(sparse_ad)
    rsc.pp.pca(default)

    cp.testing.assert_allclose(
        np.abs(sparse_ad.obsm["X_pca"]),
        cp.abs(default.obsm["X_pca"].compute()),
        rtol=1e-7,
        atol=1e-6,
    )

    cp.testing.assert_allclose(
        np.abs(sparse_ad.varm["PCs"]), np.abs(default.varm["PCs"]), rtol=1e-7, atol=1e-6
    )

    cp.testing.assert_allclose(
        np.abs(sparse_ad.uns["pca"]["variance_ratio"]),
        np.abs(default.uns["pca"]["variance_ratio"]),
        rtol=1e-7,
        atol=1e-6,
    )

def test_pca_dense_dask(client):
    sparse_ad = pbmc3k_processed()
    default = pbmc3k_processed()
    sparse_ad.X = cp.array(sparse_ad.X.astype(np.float64))
    default.X = as_dense_cupy_dask_array(default.X.astype(np.float64))
    rsc.pp.pca(sparse_ad, svd_solver="full")
    rsc.pp.pca(default, svd_solver="full")

    cp.testing.assert_allclose(
        np.abs(sparse_ad.obsm["X_pca"]),
        cp.abs(default.obsm["X_pca"].compute()),
        rtol=1e-7,
        atol=1e-6,
    )

    cp.testing.assert_allclose(
        np.abs(sparse_ad.varm["PCs"]), np.abs(default.varm["PCs"]), rtol=1e-7, atol=1e-6
    )

    cp.testing.assert_allclose(
        np.abs(sparse_ad.uns["pca"]["variance_ratio"]),
        np.abs(default.uns["pca"]["variance_ratio"]),
        rtol=1e-7,
        atol=1e-6,
    )
