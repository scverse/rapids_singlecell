from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k

import rapids_singlecell as rsc
from rapids_singlecell._testing import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


def test_qc_metrics_sparse(client):
    adata = pbmc3k()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    dask_data = adata.copy()
    dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
    adata.X = cusparse.csr_matrix(adata.X)
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=True)
    rsc.pp.calculate_qc_metrics(dask_data, qc_vars=["mt"], log1p=True)
    np.testing.assert_allclose(
        adata.obs["n_genes_by_counts"], dask_data.obs["n_genes_by_counts"]
    )
    np.testing.assert_allclose(adata.obs["total_counts"], dask_data.obs["total_counts"])
    np.testing.assert_allclose(
        adata.obs["log1p_n_genes_by_counts"], dask_data.obs["log1p_n_genes_by_counts"]
    )
    np.testing.assert_allclose(
        adata.obs["log1p_total_counts"], dask_data.obs["log1p_total_counts"]
    )
    np.testing.assert_allclose(
        adata.obs["pct_counts_mt"], dask_data.obs["pct_counts_mt"]
    )
    np.testing.assert_allclose(
        adata.obs["total_counts_mt"], dask_data.obs["total_counts_mt"]
    )
    np.testing.assert_allclose(
        adata.obs["log1p_total_counts_mt"], dask_data.obs["log1p_total_counts_mt"]
    )
    np.testing.assert_allclose(
        adata.var["n_cells_by_counts"], dask_data.var["n_cells_by_counts"]
    )
    np.testing.assert_allclose(adata.var["total_counts"], dask_data.var["total_counts"])
    np.testing.assert_allclose(adata.var["mean_counts"], dask_data.var["mean_counts"])
    np.testing.assert_allclose(
        adata.var["pct_dropout_by_counts"], dask_data.var["pct_dropout_by_counts"]
    )
    np.testing.assert_allclose(
        adata.var["log1p_total_counts"], dask_data.var["log1p_total_counts"]
    )
    np.testing.assert_allclose(
        adata.var["log1p_mean_counts"], dask_data.var["log1p_mean_counts"]
    )


def test_qc_metrics_dense(client):
    adata = pbmc3k()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    dask_data = adata.copy()
    dask_data.X = as_dense_cupy_dask_array(dask_data.X)
    adata.X = cp.array(adata.X.toarray())
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=True)
    rsc.pp.calculate_qc_metrics(dask_data, qc_vars=["mt"], log1p=True)
    np.testing.assert_allclose(
        adata.obs["n_genes_by_counts"], dask_data.obs["n_genes_by_counts"]
    )
    np.testing.assert_allclose(adata.obs["total_counts"], dask_data.obs["total_counts"])
    np.testing.assert_allclose(
        adata.obs["log1p_n_genes_by_counts"], dask_data.obs["log1p_n_genes_by_counts"]
    )
    np.testing.assert_allclose(
        adata.obs["log1p_total_counts"], dask_data.obs["log1p_total_counts"]
    )
    np.testing.assert_allclose(
        adata.obs["pct_counts_mt"], dask_data.obs["pct_counts_mt"]
    )
    np.testing.assert_allclose(
        adata.obs["total_counts_mt"], dask_data.obs["total_counts_mt"]
    )
    np.testing.assert_allclose(
        adata.obs["log1p_total_counts_mt"], dask_data.obs["log1p_total_counts_mt"]
    )
    np.testing.assert_allclose(
        adata.var["n_cells_by_counts"], dask_data.var["n_cells_by_counts"]
    )
    np.testing.assert_allclose(adata.var["total_counts"], dask_data.var["total_counts"])
    np.testing.assert_allclose(adata.var["mean_counts"], dask_data.var["mean_counts"])
    np.testing.assert_allclose(
        adata.var["pct_dropout_by_counts"], dask_data.var["pct_dropout_by_counts"]
    )
    np.testing.assert_allclose(
        adata.var["log1p_total_counts"], dask_data.var["log1p_total_counts"]
    )
    np.testing.assert_allclose(
        adata.var["log1p_mean_counts"], dask_data.var["log1p_mean_counts"]
    )
