from __future__ import annotations

import scanpy as sc

import rapids_singlecell as rsc


def test_harmony_integrate():
    """
    Test that Harmony integrate works.

    This is a very simple test that just checks to see if the Harmony
    integrate wrapper successfully added a new field to ``adata.obsm``
    and makes sure it has the same dimensions as the original PCA table.
    """
    adata = sc.datasets.pbmc3k()
    sc.pp.recipe_zheng17(adata)
    sc.tl.pca(adata)
    adata.obs["batch"] = 1350 * ["a"] + 1350 * ["b"]
    rsc.pp.harmony_integrate(adata, "batch")
    assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape
