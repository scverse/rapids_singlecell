from __future__ import annotations

import cupy as cp
import pytest
import scanpy as sc

import rapids_singlecell as rsc


@pytest.mark.parametrize("correction_method", ["fast", "original"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_harmony_integrate(correction_method, dtype):
    """
    Test that Harmony integrate works.

    This is a very simple test that just checks to see if the Harmony
    integrate wrapper successfully added a new field to ``adata.obsm``
    and makes sure it has the same dimensions as the original PCA table.
    """
    adata = sc.datasets.pbmc68k_reduced()
    rsc.pp.harmony_integrate(
        adata, "bulk_labels", correction_method=correction_method, dtype=dtype
    )
    assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape
