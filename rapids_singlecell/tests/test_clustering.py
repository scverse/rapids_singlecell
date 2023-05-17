import pytest
import rapids_singlecell as rsc
from scanpy.datasets import pbmc68k_reduced


@pytest.fixture
def adata_neighbors():
    return pbmc68k_reduced()


def test_leiden_basic(adata_neighbors):
    rsc.tl.leiden(adata_neighbors, use_weights=False)
    rsc.tl.leiden(adata_neighbors, use_weights=True)


def test_louvain_basic(adata_neighbors):
    rsc.tl.louvain(adata_neighbors, use_weights=False)
    rsc.tl.louvain(adata_neighbors, use_weights=True)
