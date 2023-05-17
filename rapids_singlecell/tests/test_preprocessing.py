import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp


def test_scale():
    adata = sc.datasets.pbmc68k_reduced()
    adata.X = adata.raw.X

    v = adata[:, 0 : adata.shape[1] // 2]
    cudata = rsc.cunnData.cunnData(v)
    cudata.X = cudata.X.toarray().astype(cp.float64)

    rsc.pp.scale(cudata)

    cp.testing.assert_allclose(
        cudata.X.var(axis=0), cp.ones(cudata.shape[1]), atol=0.01
    )
    cp.testing.assert_allclose(
        cudata.X.mean(axis=0), cp.zeros(cudata.shape[1]), atol=0.00001
    )
