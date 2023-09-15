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


@pytest.mark.parametrize("key", ["leiden", "louvain"])
def test_clustering_subset(adata_neighbors, key):
    if key == "leiden":
        clustering = rsc.tl.leiden
    else:
        clustering = rsc.tl.louvain
    clustering(adata_neighbors, key_added=key)

    for c in adata_neighbors.obs[key].unique():
        print("Analyzing cluster ", c)
        cells_in_c = adata_neighbors.obs[key] == c
        ncells_in_c = adata_neighbors.obs[key].value_counts().loc[c]
        key_sub = str(key) + "_sub"
        clustering(
            adata_neighbors,
            restrict_to=(key, [c]),
            key_added=key_sub,
        )
        # Get new clustering labels
        new_partition = adata_neighbors.obs[key_sub]

        cat_counts = new_partition[cells_in_c].value_counts()

        # Only original cluster's cells assigned to new categories
        assert cat_counts.sum() == ncells_in_c

        # Original category's cells assigned only to new categories
        nonzero_cat = cat_counts[cat_counts > 0].index
        common_cat = nonzero_cat.intersection(adata_neighbors.obs[key].cat.categories)
        assert len(common_cat) == 0
