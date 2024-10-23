from __future__ import annotations

import cupy as cp
import pandas as pd
import pytest
import scanpy as sc
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k

import rapids_singlecell as rsc
from rapids_singlecell._testing import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


def _get_anndata():
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger"])
def test_highly_variable_genes(client, data_kind, flavor):
    adata = _get_anndata()
    adata.X = adata.X.astype("float64")
    dask_data = adata.copy()

    if data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        adata.X = cp.array(adata.X.toarray())
    elif data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        adata.X = cusparse.csr_matrix(adata.X)
    else:
        raise ValueError(f"Unknown data_kind: {data_kind}")

    rsc.pp.highly_variable_genes(adata, flavor=flavor)
    rsc.pp.highly_variable_genes(dask_data, flavor=flavor)

    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(
        adata.var["dispersions_norm"], dask_data.var["dispersions_norm"]
    )


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger"])
def test_highly_variable_genes_batched(client, data_kind, flavor):
    adata = _get_anndata()
    adata.obs["batch"] = (
        "source_" + pd.array([*range(1, 6), 5]).repeat(500).astype("string")
    )[: adata.n_obs]
    dask_data = adata.copy()

    if data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        adata.X = cp.array(adata.X.toarray())
    elif data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        adata.X = cusparse.csr_matrix(adata.X)
    else:
        raise ValueError(f"Unknown data_kind: {data_kind}")

    rsc.pp.highly_variable_genes(adata, batch_key="batch")
    rsc.pp.highly_variable_genes(dask_data, batch_key="batch")
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(
        adata.var["dispersions_norm"], dask_data.var["dispersions_norm"]
    )
