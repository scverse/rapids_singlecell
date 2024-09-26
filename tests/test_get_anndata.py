from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._utils import _check_gpu_X


@pytest.mark.parametrize("mtype", ["csc", "csr", "dense"])
def test_utils(mtype):
    if mtype in {"csc", "csr"}:
        adata = sc.datasets.pbmc3k()
        if mtype == "csc":
            adata.X = csc_matrix(adata.X)
    elif mtype == "dense":
        adata = sc.datasets.pbmc68k_reduced()
    # check X
    rsc.get.anndata_to_GPU(adata)
    rsc.preprocessing._utils._check_gpu_X(adata.X)
    rsc.get.anndata_to_CPU(adata)
    assert isinstance(adata.X, (np.ndarray, csr_matrix, csc_matrix))
    # check layers
    adata.layers["test"] = adata.X.copy()
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    _check_gpu_X(adata.layers["test"])
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    assert isinstance(adata.layers["test"], (np.ndarray, csr_matrix, csc_matrix))
