from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from anndata import AnnData, read_h5ad

from rapids_singlecell.squidpy_gpu._co_oc import (
    _co_occurrence_helper,
    _find_min_max,
    co_occurrence,
)


@pytest.fixture
def adata():
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)
    dummy_adata.obs["leiden"] = dummy_adata.obs.cluster.astype("category")

    return dummy_adata


def test_co_occurrence(adata: AnnData):
    """
    check co_occurrence score and shape
    """
    co_occurrence(adata, cluster_key="leiden")

    # assert occurrence in adata.uns
    assert "leiden_co_occurrence" in adata.uns.keys()
    assert "occ" in adata.uns["leiden_co_occurrence"].keys()
    assert "interval" in adata.uns["leiden_co_occurrence"].keys()

    # assert shapes
    arr = adata.uns["leiden_co_occurrence"]["occ"]
    assert arr.ndim == 3
    assert arr.shape[2] == 49
    assert arr.shape[1] == arr.shape[0] == adata.obs["leiden"].unique().shape[0]


def test_co_occurrence_reproducibility(adata: AnnData):
    """Check co_occurrence reproducibility results."""
    arr_1, interval_1 = co_occurrence(adata, cluster_key="leiden", copy=True)
    arr_2, interval_2 = co_occurrence(adata, cluster_key="leiden", copy=True)

    np.testing.assert_array_equal(sorted(interval_1), sorted(interval_2))
    np.testing.assert_allclose(arr_1, arr_2)


@pytest.mark.parametrize("size", [1, 3])
def test_co_occurrence_explicit_interval(adata: AnnData, size: int):
    minn, maxx = _find_min_max(cp.array(adata.obsm["spatial"]))
    interval = np.linspace(minn, maxx, size)
    if size == 1:
        with pytest.raises(ValueError, match=r"Expected interval to be of length"):
            _ = co_occurrence(adata, cluster_key="leiden", copy=True, interval=interval)
    else:
        _, interval_1 = co_occurrence(
            adata, cluster_key="leiden", copy=True, interval=interval
        )

        assert interval is not interval_1
        cp.testing.assert_allclose(
            interval, interval_1
        )  # allclose because in the func, we use f32


def test_co_occurrence_fast_kernel():
    n = 100  # number of points
    k = 5  # number of labels
    # Generate random data
    spatial = cp.asarray(np.random.rand(n, 2).astype(np.float32))
    thresholds = cp.asarray(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    label_idx = cp.asarray(np.random.randint(0, k, size=n, dtype=np.int32))
    occ_prob_fast = _co_occurrence_helper(spatial, thresholds, label_idx, fast=True)
    cp.cuda.Stream.null.synchronize()
    occ_prob_slow = _co_occurrence_helper(spatial, thresholds, label_idx, fast=False)
    cp.testing.assert_allclose(occ_prob_fast, occ_prob_slow)
