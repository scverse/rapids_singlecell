import cupy as cp
import pytest
import rapids_singlecell as rsc
import scanpy as sc


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scale(dtype):
    adata = sc.datasets.pbmc68k_reduced()
    adata.X = adata.raw.X.todense()

    adata[:, 0 : adata.shape[1] // 2]
    adata.X = cp.array(adata.X, dtype=dtype)

    rsc.pp.scale(adata)

    cp.testing.assert_allclose(adata.X.var(axis=0), cp.ones(adata.shape[1]), atol=0.01)
    cp.testing.assert_allclose(
        adata.X.mean(axis=0), cp.zeros(adata.shape[1]), atol=0.00001
    )
