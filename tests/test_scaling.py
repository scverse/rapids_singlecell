from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix

import scanpy as sc
import cupy as cp
import pytest
import rapids_singlecell as rsc
import scanpy as sc


# test "data" for 3 cells * 4 genes
X_original = np.array([
    [-1, 2, 0, 0],
    [1, 2, 4, 0],
    [0, 2, 2, 0],
])  # with gene std 1,0,2,0 and center 0,2,2,0
X_scaled_original = np.array([
    [-1, 2, 0, 0],
    [1, 2, 2, 0],
    [0, 2, 1, 0],
])  # with gene std 1,0,1,0 and center 0,2,1,0
X_centered_original = np.array([
    [-1, 0, -1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
]) # with gene std 1,0,1,0 and center 0,0,0,0

X_for_mask = np.array([
    [27, 27, 27, 27],
    [27, 27, 27, 27],
    [-1, 2, 0, 0],
    [1, 2, 4, 0],
    [0, 2, 2, 0],
    [27, 27, 27, 27],
    [27, 27, 27, 27],
])
X_scaled_for_mask = np.array([
    [27, 27, 27, 27],
    [27, 27, 27, 27],
    [-1, 2, 0, 0],
    [1, 2, 2, 0],
    [0, 2, 1, 0],
    [27, 27, 27, 27],
    [27, 27, 27, 27],
])
X_centered_for_mask = np.array([
    [27, 27, 27, 27],
    [27, 27, 27, 27],
    [-1, 0, -1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
    [27, 27, 27, 27],
    [27, 27, 27, 27],
])

@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scale_simple(dtype):
    adata = sc.datasets.pbmc68k_reduced()
    adata.X = adata.raw.X.todense()

    adata[:, 0 : adata.shape[1] // 2]
    adata.X = cp.array(adata.X, dtype=dtype)

    rsc.pp.scale(adata)

    cp.testing.assert_allclose(adata.X.var(axis=0), cp.ones(adata.shape[1]), atol=0.01)
    cp.testing.assert_allclose(
        adata.X.mean(axis=0), cp.zeros(adata.shape[1]), atol=0.00001
    )


@pytest.mark.parametrize("typ", [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    ("mask_obs", "X", "X_centered", "X_scaled"),
    [
        (None, X_original, X_centered_original, X_scaled_original),
        (
            np.array((0, 0, 1, 1, 1, 0, 0), dtype=bool),
            X_for_mask,
            X_centered_for_mask,
            X_scaled_for_mask,
        ),
    ],
)
def test_scale(*, typ, dtype, mask_obs, X, X_centered, X_scaled):
    # test AnnData arguments
    # test scaling with default zero_center == True
    adata = AnnData(typ(X,dtype=dtype))
    adata0 = rsc.get.anndata_to_GPU(adata,copy= True)
    rsc.pp.scale(adata0, mask_obs=mask_obs)
    cp.testing.assert_allclose(cp_csr_matrix(adata0.X).toarray(), X_centered)
    # test scaling with explicit zero_center == True
    adata1 = rsc.get.anndata_to_GPU(adata,copy= True)
    rsc.pp.scale(adata1, zero_center=True, mask_obs=mask_obs)
    cp.testing.assert_allclose(cp_csr_matrix(adata1.X).toarray(), X_centered)
    # test scaling with explicit zero_center == False
    adata2 = rsc.get.anndata_to_GPU(adata,copy= True)
    rsc.pp.scale(adata2, zero_center=False, mask_obs=mask_obs)
    cp.testing.assert_allclose(cp_csr_matrix(adata2.X).toarray(), X_scaled)

def test_mask_string():
    adata = AnnData(np.array(X_for_mask, dtype="float32"))
    rsc.get.anndata_to_GPU(adata)
    with pytest.raises(ValueError):
        rsc.pp.scale(adata, mask_obs="mask")
    adata.obs["some cells"] = np.array((0, 0, 1, 1, 1, 0, 0), dtype=bool)
    rsc.pp.scale(adata, mask_obs="some cells")
    cp.testing.assert_allclose(adata.X, X_centered_for_mask)
    assert "mean of some cells" in adata.var.keys()
