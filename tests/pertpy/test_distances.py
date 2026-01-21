from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from rapids_singlecell.pertpy_gpu import Distance, EDistanceResult


@pytest.fixture
def small_adata() -> AnnData:
    rng = np.random.default_rng(0)
    n_groups = 3
    cells_per_group = 4
    n_features = 5
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)
    return adata


# ============================================================================
# Tests for Distance class API
# ============================================================================


def test_distance_class_initialization() -> None:
    """Test Distance class can be initialized with different metrics."""
    distance = Distance(metric="edistance")
    assert distance.metric == "edistance"
    assert distance.obsm_key == "X_pca"

    distance_custom = Distance(metric="edistance", obsm_key="X_custom")
    assert distance_custom.obsm_key == "X_custom"


def test_distance_class_invalid_metric() -> None:
    """Test Distance class raises error for unsupported metrics."""
    with pytest.raises(ValueError, match="Unknown metric"):
        Distance(metric="invalid_metric")


def test_distance_class_pairwise_with_bootstrap(small_adata: AnnData) -> None:
    """Test Distance.pairwise() with bootstrap."""
    distance = Distance(metric="edistance")
    result = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=10,
        random_state=42,
    )

    assert isinstance(result, EDistanceResult)
    assert result.distances is not None
    assert result.distances_var is not None
    assert result.distances.shape == result.distances_var.shape
    assert np.all(result.distances_var.values >= 0)


def test_distance_class_onesided_distances(small_adata: AnnData) -> None:
    """Test Distance.onesided_distances() method."""
    distance = Distance(metric="edistance")
    result = distance.onesided_distances(
        small_adata, groupby="group", selected_group="g0"
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 3  # 3 groups total
    assert "g0" in result.index
    assert "g1" in result.index
    assert "g2" in result.index


def test_distance_class_onesided_matches_pairwise(small_adata: AnnData) -> None:
    """Test onesided_distances matches the corresponding row from pairwise."""
    distance = Distance(metric="edistance")

    # Get pairwise distances
    pairwise_result = distance.pairwise(small_adata, groupby="group")

    # Get onesided distances for each group
    for group in ["g0", "g1", "g2"]:
        onesided = distance.onesided_distances(
            small_adata, groupby="group", selected_group=group
        )
        # Should match the row from pairwise matrix
        np.testing.assert_allclose(
            onesided.values, pairwise_result.distances.loc[group].values, atol=1e-5
        )


def test_distance_class_onesided_self_distance_zero(small_adata: AnnData) -> None:
    """Test that distance from a group to itself is zero."""
    distance = Distance(metric="edistance")

    for group in ["g0", "g1", "g2"]:
        onesided = distance.onesided_distances(
            small_adata, groupby="group", selected_group=group
        )
        # Distance to self should be 0
        assert onesided[group] == pytest.approx(0.0, abs=1e-6)


def test_distance_class_onesided_invalid_group(small_adata: AnnData) -> None:
    """Test Distance.onesided_distances() raises error for invalid group."""
    distance = Distance(metric="edistance")
    with pytest.raises(ValueError, match="not found"):
        distance.onesided_distances(
            small_adata, groupby="group", selected_group="invalid"
        )


def test_distance_class_bootstrap_two_groups(small_adata: AnnData) -> None:
    """Test Distance.bootstrap() for two specific groups."""
    distance = Distance(metric="edistance")
    mean, variance = distance.bootstrap(
        small_adata,
        groupby="group",
        group_a="g0",
        group_b="g1",
        n_bootstrap=10,
        random_state=42,
    )

    assert isinstance(mean, float)
    assert isinstance(variance, float)
    assert variance >= 0


def test_distance_class_inplace_storage(small_adata: AnnData) -> None:
    """Test Distance.pairwise() with inplace=True."""
    distance = Distance(metric="edistance")
    result = distance.pairwise(small_adata, groupby="group", inplace=True)

    # Check result is returned
    assert isinstance(result, EDistanceResult)

    # Check result is stored in uns
    key = "group_pairwise_edistance"
    assert key in small_adata.uns
    stored = small_adata.uns[key]
    np.testing.assert_allclose(stored["distances"].values, result.distances.values)


def test_distance_class_repr() -> None:
    """Test Distance.__repr__() method."""
    distance = Distance(metric="edistance", obsm_key="X_custom")
    repr_str = repr(distance)
    assert "Distance" in repr_str
    assert "edistance" in repr_str
    assert "X_custom" in repr_str


def test_calculate_blocks_per_pair() -> None:
    """Test blocks_per_pair calculation logic."""
    distance = Distance(metric="edistance")

    # Test with large num_pairs (should get 1 block per pair)
    blocks = distance._metric_impl._calculate_blocks_per_pair(num_pairs=10000)
    assert blocks >= 1

    # Test with small num_pairs (should get multiple blocks per pair, capped at 32)
    blocks = distance._metric_impl._calculate_blocks_per_pair(num_pairs=1)
    assert 1 <= blocks <= 32

    # Test custom parameters
    blocks = distance._metric_impl._calculate_blocks_per_pair(
        num_pairs=10, target_multiplier=8, max_blocks_per_pair=16
    )
    assert 1 <= blocks <= 16

    # Test edge case: num_pairs=0 should not raise error (max(1, ...) handles it)
    blocks = distance._metric_impl._calculate_blocks_per_pair(num_pairs=0)
    assert blocks >= 1
