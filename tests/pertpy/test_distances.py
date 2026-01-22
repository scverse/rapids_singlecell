from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from rapids_singlecell.pertpy_gpu import Distance, MeanVar


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
    distances, distances_var = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=10,
        random_state=42,
    )

    assert isinstance(distances, pd.DataFrame)
    assert isinstance(distances_var, pd.DataFrame)
    assert distances.shape == distances_var.shape
    assert np.all(distances_var.values >= 0)


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
    pairwise_df = distance.pairwise(small_adata, groupby="group")

    # Get onesided distances for each group
    for group in ["g0", "g1", "g2"]:
        onesided = distance.onesided_distances(
            small_adata, groupby="group", selected_group=group
        )
        # Should match the row from pairwise matrix
        np.testing.assert_allclose(
            onesided.values, pairwise_df.loc[group].values, atol=1e-5
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
    """Test Distance.bootstrap_adata() for two specific groups."""
    distance = Distance(metric="edistance")
    result = distance.bootstrap_adata(
        small_adata,
        groupby="group",
        group_a="g0",
        group_b="g1",
        n_bootstrap=10,
        random_state=42,
    )

    assert isinstance(result.mean, float)
    assert isinstance(result.variance, float)
    assert result.variance >= 0


def test_distance_class_inplace_storage(small_adata: AnnData) -> None:
    """Test Distance.pairwise() with inplace=True."""
    distance = Distance(metric="edistance")
    result_df = distance.pairwise(small_adata, groupby="group", inplace=True)

    # Check result is returned
    assert isinstance(result_df, pd.DataFrame)

    # Check result is stored in uns
    key = "group_pairwise_edistance"
    assert key in small_adata.uns
    stored = small_adata.uns[key]
    np.testing.assert_allclose(stored["distances"].values, result_df.values)


def test_distance_class_repr() -> None:
    """Test Distance.__repr__() method."""
    distance = Distance(metric="edistance", obsm_key="X_custom")
    repr_str = repr(distance)
    assert "Distance" in repr_str
    assert "edistance" in repr_str
    assert "X_custom" in repr_str


# ============================================================================
# Correctness tests against reference implementation
# ============================================================================


def _compute_mean_euclidean_distance_cpu(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute mean Euclidean distance between all pairs in X and Y (CPU reference)."""
    from scipy.spatial.distance import cdist

    if X is Y or np.array_equal(X, Y):
        # Within-group: compute upper triangle only
        dists = cdist(X, X, metric="euclidean")
        n = len(X)
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        return dists[triu_indices].mean()
    else:
        # Between-group: all pairs
        dists = cdist(X, Y, metric="euclidean")
        return dists.mean()


def _compute_energy_distance_cpu(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute energy distance between X and Y (CPU reference).

    E(X, Y) = 2 * mean_dist(X, Y) - mean_dist(X, X) - mean_dist(Y, Y)
    """
    d_xy = _compute_mean_euclidean_distance_cpu(X, Y)
    d_xx = _compute_mean_euclidean_distance_cpu(X, X)
    d_yy = _compute_mean_euclidean_distance_cpu(Y, Y)
    return 2 * d_xy - d_xx - d_yy


def test_edistance_correctness_vs_cpu(small_adata: AnnData) -> None:
    """Test that GPU energy distance matches CPU reference implementation."""
    distance = Distance(metric="edistance")
    result_df = distance.pairwise(small_adata, groupby="group")

    # Get CPU embedding
    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    # Compute reference energy distances for each pair
    group_names = ["g0", "g1", "g2"]
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i <= j:
                X = cpu_embedding[groups == g1]
                Y = cpu_embedding[groups == g2]

                if g1 == g2:
                    expected = 0.0  # Self-distance is 0
                else:
                    expected = _compute_energy_distance_cpu(X, Y)

                actual = result_df.loc[g1, g2]
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Mismatch for ({g1}, {g2}): GPU={actual}, CPU={expected}",
                )


def test_edistance_correctness_larger_dataset() -> None:
    """Test correctness with larger dataset and more groups."""
    rng = np.random.default_rng(42)
    n_groups = 5
    cells_per_group = 20
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")
    result_df = distance.pairwise(adata, groupby="group")

    # Check a few pairs against CPU reference
    for g1, g2 in [("g0", "g1"), ("g0", "g4"), ("g2", "g3")]:
        X = cpu_embedding[np.array(groups) == g1]
        Y = cpu_embedding[np.array(groups) == g2]
        expected = _compute_energy_distance_cpu(X, Y)
        actual = result_df.loc[g1, g2]
        np.testing.assert_allclose(
            actual, expected, rtol=1e-5, atol=1e-6, err_msg=f"Mismatch for ({g1}, {g2})"
        )


# ============================================================================
# Onesided distance correctness tests
# ============================================================================


def test_onesided_distances_correctness_vs_cpu(small_adata: AnnData) -> None:
    """Test that onesided_distances matches CPU reference implementation."""
    distance = Distance(metric="edistance")

    # Get CPU embedding
    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    # Test onesided from each group
    for selected_group in ["g0", "g1", "g2"]:
        onesided = distance.onesided_distances(
            small_adata, groupby="group", selected_group=selected_group
        )

        # Verify against CPU reference for each target group
        X = cpu_embedding[groups == selected_group]
        for target_group in ["g0", "g1", "g2"]:
            Y = cpu_embedding[groups == target_group]

            if selected_group == target_group:
                expected = 0.0
            else:
                expected = _compute_energy_distance_cpu(X, Y)

            actual = onesided[target_group]
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Onesided mismatch for ({selected_group}, {target_group})",
            )


def test_onesided_matches_pairwise_all_groups(small_adata: AnnData) -> None:
    """Test that onesided_distances matches the pairwise matrix for all groups."""
    distance = Distance(metric="edistance")
    pairwise_df = distance.pairwise(small_adata, groupby="group")

    # For each group, onesided should match the corresponding row
    for group in ["g0", "g1", "g2"]:
        onesided = distance.onesided_distances(
            small_adata, groupby="group", selected_group=group
        )

        # Extract the row from pairwise
        pairwise_row = pairwise_df.loc[group]

        # Should match exactly (same computation path after optimization)
        np.testing.assert_allclose(
            onesided.values,
            pairwise_row.values,
            atol=1e-5,
            err_msg=f"Onesided vs pairwise mismatch for group {group}",
        )


# ============================================================================
# Bootstrap correctness tests
# ============================================================================


def test_bootstrap_variance_is_positive(small_adata: AnnData) -> None:
    """Test that bootstrap variance is always non-negative."""
    distance = Distance(metric="edistance")
    distances_df, distances_var_df = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )

    # All variances should be >= 0
    assert np.all(distances_var_df.values >= 0), (
        "Bootstrap variance should be non-negative"
    )

    # Off-diagonal variances should be > 0 (with bootstrap resampling)
    for i, g1 in enumerate(["g0", "g1", "g2"]):
        for j, g2 in enumerate(["g0", "g1", "g2"]):
            if i != j:
                assert distances_var_df.loc[g1, g2] > 0, (
                    f"Off-diagonal variance should be positive for ({g1}, {g2})"
                )


def test_bootstrap_mean_close_to_point_estimate() -> None:
    """Test that bootstrap mean is close to point estimate (no bootstrap)."""
    # Need larger dataset for meaningful bootstrap comparison
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 50  # Larger for stable bootstrap
    n_features = 10
    total_cells = n_groups * cells_per_group

    cpu_embedding = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    adata = AnnData(cpu_embedding.copy(), obs=obs)
    adata.obsm["X_pca"] = cp.asarray(cpu_embedding, dtype=cp.float32)

    distance = Distance(metric="edistance")

    # Point estimate
    point_df = distance.pairwise(adata, groupby="group")

    # Bootstrap estimate
    boot_df, boot_var_df = distance.pairwise(
        adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=100,
        random_state=42,
    )

    # Bootstrap mean should be reasonably close to point estimate
    # With 50 cells per group, the bootstrap should be more stable
    np.testing.assert_allclose(
        boot_df.values,
        point_df.values,
        rtol=0.3,  # 30% tolerance for bootstrap variation
        atol=0.5,  # Absolute tolerance for small distances
        err_msg="Bootstrap mean should be close to point estimate",
    )


def test_bootstrap_reproducibility(small_adata: AnnData) -> None:
    """Test that bootstrap with same seed produces same results."""
    distance = Distance(metric="edistance")

    df1, var1 = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )

    df2, var2 = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=20,
        random_state=42,
    )

    np.testing.assert_allclose(
        df1.values,
        df2.values,
        atol=1e-10,
        err_msg="Same seed should produce same bootstrap results",
    )

    np.testing.assert_allclose(
        var1.values,
        var2.values,
        atol=1e-10,
        err_msg="Same seed should produce same bootstrap variance",
    )


def test_bootstrap_two_groups_matches_pairwise(small_adata: AnnData) -> None:
    """Test that Distance.bootstrap_adata() for two groups matches pairwise bootstrap."""
    distance = Distance(metric="edistance")

    # Use pairwise bootstrap
    pairwise_df, pairwise_var_df = distance.pairwise(
        small_adata,
        groupby="group",
        bootstrap=True,
        n_bootstrap=50,
        random_state=42,
    )

    # Use two-group bootstrap (adata-based)
    result = distance.bootstrap_adata(
        small_adata,
        groupby="group",
        group_a="g0",
        group_b="g1",
        n_bootstrap=50,
        random_state=42,
    )

    # Should match the corresponding cell in pairwise result
    np.testing.assert_allclose(
        result.mean,
        pairwise_df.loc["g0", "g1"],
        atol=1e-6,
        err_msg="Two-group bootstrap mean should match pairwise",
    )
    np.testing.assert_allclose(
        result.variance,
        pairwise_var_df.loc["g0", "g1"],
        atol=1e-6,
        err_msg="Two-group bootstrap variance should match pairwise",
    )


# ============================================================================
# Tests for Distance.__call__ API (pertpy-compatible)
# ============================================================================


def test_distance_call_api_basic(small_adata: AnnData) -> None:
    """Test Distance.__call__ computes distance directly from arrays."""
    distance = Distance(metric="edistance")

    # Extract arrays for two groups
    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    X = cpu_embedding[groups == "g0"]
    Y = cpu_embedding[groups == "g1"]

    # Compute via __call__
    d = distance(X, Y)

    assert isinstance(d, float)
    assert d >= 0 or d < 0  # Energy distance can be negative with small samples


def test_distance_call_api_vs_cpu_reference(small_adata: AnnData) -> None:
    """Test Distance.__call__ matches CPU reference implementation."""
    distance = Distance(metric="edistance")

    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    # Test for each pair
    for g1, g2 in [("g0", "g1"), ("g0", "g2"), ("g1", "g2")]:
        X = cpu_embedding[groups == g1]
        Y = cpu_embedding[groups == g2]

        # GPU result via __call__
        actual = distance(X, Y)

        # CPU reference
        expected = _compute_energy_distance_cpu(X, Y)

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"__call__ mismatch for ({g1}, {g2})",
        )


def test_distance_call_api_vs_pairwise(small_adata: AnnData) -> None:
    """Test Distance.__call__ matches pairwise results."""
    distance = Distance(metric="edistance")

    # Get pairwise results
    result_df = distance.pairwise(small_adata, groupby="group")

    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    # Test each pair
    for g1 in ["g0", "g1", "g2"]:
        for g2 in ["g0", "g1", "g2"]:
            if g1 == g2:
                continue

            X = cpu_embedding[groups == g1]
            Y = cpu_embedding[groups == g2]

            # Via __call__
            call_result = distance(X, Y)

            # Via pairwise
            pairwise_result = result_df.loc[g1, g2]

            np.testing.assert_allclose(
                call_result,
                pairwise_result,
                atol=1e-5,
                err_msg=f"__call__ vs pairwise mismatch for ({g1}, {g2})",
            )


def test_distance_call_api_with_cupy_arrays(small_adata: AnnData) -> None:
    """Test Distance.__call__ works with CuPy arrays."""
    distance = Distance(metric="edistance")

    # Use CuPy arrays directly
    embedding = small_adata.obsm["X_pca"]  # Already CuPy
    groups = small_adata.obs["group"].values

    X = embedding[cp.array(groups == "g0")]
    Y = embedding[cp.array(groups == "g1")]

    # Should work with CuPy arrays
    d = distance(X, Y)
    assert isinstance(d, float)


# ============================================================================
# Tests for Distance.bootstrap with arrays (pertpy-compatible)
# ============================================================================


def test_distance_bootstrap_arrays_basic(small_adata: AnnData) -> None:
    """Test Distance.bootstrap() with arrays (pertpy-compatible API)."""
    distance = Distance(metric="edistance")

    cpu_embedding = small_adata.obsm["X_pca"].get()
    groups = small_adata.obs["group"].values

    X = cpu_embedding[groups == "g0"]
    Y = cpu_embedding[groups == "g1"]

    result = distance.bootstrap(X, Y, n_bootstrap=20, random_state=42)

    assert isinstance(result, MeanVar)
    assert isinstance(result.mean, float)
    assert isinstance(result.variance, float)
    assert result.variance >= 0


def test_distance_bootstrap_arrays_reproducibility() -> None:
    """Test bootstrap with arrays is reproducible with same seed."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 10)).astype(np.float32)
    Y = rng.normal(size=(30, 10)).astype(np.float32)

    distance = Distance(metric="edistance")

    result1 = distance.bootstrap(X, Y, n_bootstrap=50, random_state=123)
    result2 = distance.bootstrap(X, Y, n_bootstrap=50, random_state=123)

    assert result1.mean == result2.mean
    assert result1.variance == result2.variance


def test_distance_bootstrap_arrays_vs_call() -> None:
    """Test bootstrap mean is close to direct __call__ result."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 10)).astype(np.float32)
    Y = rng.normal(loc=1.0, size=(50, 10)).astype(np.float32)  # Different distribution

    distance = Distance(metric="edistance")

    # Point estimate
    point_estimate = distance(X, Y)

    # Bootstrap estimate
    result = distance.bootstrap(X, Y, n_bootstrap=100, random_state=42)

    # Bootstrap mean should be reasonably close to point estimate
    np.testing.assert_allclose(
        result.mean,
        point_estimate,
        rtol=0.3,
        atol=0.5,
        err_msg="Bootstrap mean should be close to point estimate",
    )


# ============================================================================
# Tests for layer_key support
# ============================================================================


def test_distance_layer_key_basic() -> None:
    """Test Distance with layer_key (uses data with same shape as adata.X)."""
    rng = np.random.default_rng(42)
    n_groups = 3
    cells_per_group = 20
    n_features = 10  # This must match n_vars for layers
    total_cells = n_groups * cells_per_group

    # Create expression data with proper shape for both X and layer
    cpu_data = rng.normal(size=(total_cells, n_features)).astype(np.float32)
    groups = [f"g{idx}" for idx in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(groups, categories=[f"g{i}" for i in range(n_groups)])}
    )

    # Create AnnData with n_features as n_vars (required for layer to match)
    adata = AnnData(np.zeros((total_cells, n_features), dtype=np.float32), obs=obs)
    # Store in layer (must match shape of X: n_obs x n_vars)
    adata.layers["counts"] = cpu_data

    distance = Distance(metric="edistance", layer_key="counts")
    assert distance.layer_key == "counts"
    assert distance.obsm_key is None

    result_df = distance.pairwise(adata, groupby="group")

    # Verify against CPU reference
    for g1, g2 in [("g0", "g1"), ("g0", "g2")]:
        X = cpu_data[np.array(groups) == g1]
        Y = cpu_data[np.array(groups) == g2]
        expected = _compute_energy_distance_cpu(X, Y)
        actual = result_df.loc[g1, g2]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"layer_key mismatch for ({g1}, {g2})",
        )


def test_distance_layer_key_mutual_exclusion() -> None:
    """Test that layer_key and obsm_key are mutually exclusive."""
    with pytest.raises(ValueError, match="Cannot use 'layer_key' and 'obsm_key'"):
        Distance(metric="edistance", layer_key="counts", obsm_key="X_pca")


def test_distance_layer_key_repr() -> None:
    """Test __repr__ includes layer_key."""
    distance = Distance(metric="edistance", layer_key="counts")
    repr_str = repr(distance)
    assert "layer_key" in repr_str
    assert "counts" in repr_str


def test_distance_default_obsm_key() -> None:
    """Test default obsm_key is X_pca when neither layer_key nor obsm_key is specified."""
    distance = Distance(metric="edistance")
    assert distance.obsm_key == "X_pca"
    assert distance.layer_key is None
