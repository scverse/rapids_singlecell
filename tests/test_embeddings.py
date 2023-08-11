from rapids_singlecell.scanpy_gpu import tsne, umap
from scanpy.testing._helpers.data import pbmc68k_reduced


def test_umap():
    pbmc = pbmc68k_reduced()
    del pbmc.obsm["X_umap"]
    umap(pbmc)
    assert pbmc.obsm["X_umap"].shape == (700, 2)


def test_tsne():
    pbmc = pbmc68k_reduced()
    tsne(pbmc)
    assert pbmc.obsm["X_tsne"].shape == (700, 2)
