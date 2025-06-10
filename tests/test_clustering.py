from __future__ import annotations

import pytest
from scanpy.datasets import pbmc68k_reduced

import rapids_singlecell as rsc
from rapids_singlecell.tools._clustering import _create_graph


@pytest.fixture
def adata_neighbors():
    return pbmc68k_reduced()


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
@pytest.mark.parametrize("use_weights", [True, False])
def test_leiden_dtype(adata_neighbors, dtype, use_weights):
    if dtype == "int32":
        with pytest.raises(ValueError):
            rsc.tl.leiden(adata_neighbors, use_weights=use_weights, dtype=dtype)
    else:
        rsc.tl.leiden(adata_neighbors, use_weights=use_weights, dtype=dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
@pytest.mark.parametrize("use_weights", [True, False])
def test_louvain_dtype(adata_neighbors, dtype, use_weights):
    if dtype == "int32":
        with pytest.raises(ValueError):
            rsc.tl.louvain(adata_neighbors, use_weights=use_weights, dtype=dtype)
    else:
        rsc.tl.louvain(adata_neighbors, use_weights=use_weights, dtype=dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_create_graph_dtype(adata_neighbors, dtype):
    g = _create_graph(adata_neighbors.X, use_weights=True, dtype=dtype)
    df = g.view_edge_list()
    assert df.weight.dtype == dtype


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


def test_kmeans_basic(adata_neighbors):
    rsc.tl.kmeans(adata_neighbors)
    assert adata_neighbors.obs["kmeans"].nunique() == 8
