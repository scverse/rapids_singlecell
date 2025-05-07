from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from scipy.stats import pearsonr

import rapids_singlecell as rsc


def _get_measure(x, base, norm):
    assert norm in ["r", "L2"]

    if norm == "r":
        corr, _ = pearsonr(x, base)
        return corr
    else:
        return np.linalg.norm(x - base) / np.linalg.norm(base)


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


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_harmony_integrate_algos(dtype):
    """
    Test that Harmony integrate works.

    This is a very simple test that just checks to see if the Harmony
    integrate wrapper successfully added a new field to ``adata.obsm``
    and makes sure it has the same dimensions as the original PCA table.
    """
    adata = sc.datasets.pbmc68k_reduced()
    rsc.pp.harmony_integrate(
        adata, "bulk_labels", correction_method="fast", dtype=dtype
    )
    fast = adata.obsm["X_pca_harmony"].copy()
    rsc.pp.harmony_integrate(
        adata, "bulk_labels", correction_method="original", dtype=dtype
    )
    slow = adata.obsm["X_pca_harmony"].copy()
    assert _get_measure(fast, slow, "r").min() > 0.99
    assert _get_measure(fast, slow, "L2").max() < 0.1
