from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._utils import _check_gpu_X


@pytest.mark.parametrize(
    "mtype", [csc_matrix, csr_matrix, csc_array, csr_array, "dense"]
)
def test_utils(mtype):
    if mtype == "dense":
        adata = sc.datasets.pbmc68k_reduced()
    else:
        adata = sc.datasets.pbmc3k()
        adata.X = mtype(adata.X)
    # check X
    rsc.get.anndata_to_GPU(adata)
    rsc.preprocessing._utils._check_gpu_X(adata.X)
    rsc.get.anndata_to_CPU(adata)
    assert isinstance(adata.X, np.ndarray | csr_matrix | csc_matrix)
    # check layers
    adata.layers["test"] = adata.X.copy()
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    _check_gpu_X(adata.layers["test"])
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    assert isinstance(adata.layers["test"], np.ndarray | csr_matrix | csc_matrix)
