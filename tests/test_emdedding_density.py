from __future__ import annotations

import numpy as np
from anndata import AnnData

import rapids_singlecell as rsc


def test_embedding_density():
    # Test that density values are scaled
    # Test that the highest value is in the middle for a grid layout
    test_data = AnnData(X=np.ones((9, 10), dtype=np.float32))
    test_data.obsm["X_test"] = np.array(
        [[x, y] for x in range(3) for y in range(3)], dtype=np.float32
    )
    rsc.tl.embedding_density(test_data, "test")

    max_dens = np.max(test_data.obs["test_density"])
    min_dens = np.min(test_data.obs["test_density"])
    max_idx = test_data.obs["test_density"].idxmax()

    assert max_idx == "4"
    assert max_dens == 1
    assert min_dens == 0
