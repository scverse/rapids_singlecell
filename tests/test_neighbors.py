import numpy as np
from anndata import AnnData
from rapids_singlecell.scanpy_gpu import neighbors
from scanpy.datasets import pbmc68k_reduced

# the input data
X = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])

distances_euclidean = [
    [0.0, 2.0, 0.0, 4.123105525970459],
    [2.0, 0.0, 0.0, 5.0],
    [0.0, 6.324555397033691, 0.0, 5.385164737701416],
    [4.123105525970459, 5.0, 0.0, 0.0],
]

connectivities_umap = [
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.5849691143165735, 0.8277419907567016],
    [0.0, 0.5849691143165735, 0.0, 1.0],
    [1.0, 0.8277419907567016, 1.0, 0.0],
]


def test_umap_connectivities_euclidean():
    adata = AnnData(X=X)
    neighbors(adata, n_neighbors=3)
    assert np.allclose(adata.obsp["distances"].toarray(), distances_euclidean)
    assert np.allclose(adata.obsp["connectivities"].toarray(), connectivities_umap)


key = "test"


def test_neighbors_key_added():
    adata = pbmc68k_reduced()
    del adata.obsp
    neighbors(adata, n_neighbors=5, random_state=0)
    neighbors(adata, n_neighbors=5, random_state=0, key_added=key)

    conns_key = adata.uns[key]["connectivities_key"]
    dists_key = adata.uns[key]["distances_key"]

    assert adata.uns["neighbors"]["params"] == adata.uns[key]["params"]
    assert np.allclose(
        adata.obsp["connectivities"].toarray(), adata.obsp[conns_key].toarray()
    )
    assert np.allclose(
        adata.obsp["distances"].toarray(), adata.obsp[dists_key].toarray()
    )
