import numpy as np
import rapids_singlecell as rsc
import scanpy as sc
from rapids_singlecell.preprocessing._utils import _check_gpu_X
from scipy.sparse import csc_matrix, csr_matrix


def test_utils(mtype):
    if mtype in {"csc", "csr"}:
        adata = sc.datasets.pbmc3k()
        if mtype == "csc":
            adata.X = csc_matrix(adata.X)
    elif mtype == "dense":
        adata = sc.datasets.pbmc68k_reduced()
    # check X
    rsc.utils.anndata_to_GPU(adata)
    rsc.preprocessing._utils._check_gpu_X(adata.X)
    rsc.utils.anndata_to_CPU(adata)
    assert isinstance(adata.X, (np.ndarray, csr_matrix, csc_matrix))
    # check layers
    adata.layers["test"] = adata.X.copy()
    rsc.utils.anndata_to_GPU(adata, convert_all=True)
    _check_gpu_X(adata.layers["test"])
    rsc.utils.anndata_to_CPU(adata, convert_all=True)
    assert isinstance(adata.layers["test"], (np.ndarray, csr_matrix, csc_matrix))
