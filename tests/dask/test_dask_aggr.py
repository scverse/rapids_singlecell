from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k_processed

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("use_mask", [True, False])
def test_dask_aggr(client, data_kind, use_mask):
    adata_1 = pbmc3k_processed()
    adata_2 = pbmc3k_processed()

    if data_kind == "sparse":
        adata_1 = adata_1.raw.to_adata()
        adata_2 = adata_2.raw.to_adata()
        adata_1.X = cusparse.csr_matrix(adata_1.X.astype(np.float64))
        adata_2.X = as_sparse_cupy_dask_array(adata_2.X.astype(np.float64))
    elif data_kind == "dense":
        adata_1.X = cp.array(adata_1.X.astype(np.float64))
        adata_2.X = as_dense_cupy_dask_array(adata_2.X.astype(np.float64))
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")
    if use_mask:
        mask = adata_1.obs.louvain == "Megakaryocytes"
    else:
        mask = None

    out_in_memory = rsc.get.aggregate(
        adata_2,
        by="louvain",
        func=["sum", "mean", "var", "count_nonzero"],
        mask=mask,
    )
    out_dask = rsc.get.aggregate(
        adata_1,
        by="louvain",
        func=["sum", "mean", "var", "count_nonzero"],
        mask=mask,
    )
    for i in range(4):
        c = ["sum", "mean", "var", "count_nonzero"][i]
        a = out_in_memory.layers[c]
        b = out_dask.layers[c]
        cp.testing.assert_allclose(a, b)
