from __future__ import annotations

import cupy as cp
import pandas as pd
import pytest
from scanpy.datasets import pbmc3k_processed, pbmc68k_reduced

import rapids_singlecell as rsc
from rapids_singlecell._testing import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_rank_genes_groups_logreg(client, data_kind, dtype):
    if data_kind == "dense":
        adata = pbmc68k_reduced()
        adata.X = adata.X.astype(dtype)
        dask_data = adata.copy()
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        rsc.get.anndata_to_GPU(adata)
        groupby = "bulk_labels"
        read = "Dendritic"
    elif data_kind == "sparse":
        adata = pbmc3k_processed()
        org_var_names = adata.var_names
        adata = adata.raw.to_adata()
        adata = adata[:, org_var_names].copy()
        adata.X = adata.X.astype(dtype)
        dask_data = adata.copy()
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        rsc.get.anndata_to_GPU(adata)
        groupby = "louvain"
        read = "B cells"

    rsc.tl.rank_genes_groups_logreg(adata, groupby=groupby, use_raw=False)
    rsc.tl.rank_genes_groups_logreg(dask_data, groupby=groupby, use_raw=False)
    array_ad = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"][read]).to_numpy()[
        :10
    ]
    array_bd = pd.DataFrame(
        dask_data.uns["rank_genes_groups"]["scores"][read]
    ).to_numpy()[:10]
    cp.testing.assert_allclose(array_ad, array_bd, atol=1e-3)
