from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from rapids_singlecell.pertpy_gpu._edistance import EDistanceResult, pertpy_edistance


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


def _compute_cpu_reference(
    adata: AnnData, obsm_key: str, group_key: str
) -> tuple[np.ndarray, np.ndarray]:
    embedding = adata.obsm[obsm_key].get()
    group_series = adata.obs[group_key]
    categories = list(group_series.cat.categories)
    k = len(categories)

    pair_means = np.zeros((k, k), dtype=np.float32)

    for i, gi in enumerate(categories):
        idx_i = np.where(group_series == gi)[0]
        for j, gj in enumerate(categories[i:], start=i):
            idx_j = np.where(group_series == gj)[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                mean_distance = 0.0
            else:
                distances = []
                for idx in idx_i:
                    diffs = embedding[idx] - embedding[idx_j]
                    distances.append(np.sqrt(np.sum(diffs**2, axis=1)))
                stacked = np.concatenate(distances)
                mean_distance = stacked.mean(dtype=np.float64)
            pair_means[i, j] = pair_means[j, i] = np.float32(mean_distance)

    edistance = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            value = 2 * pair_means[i, j] - pair_means[i, i] - pair_means[j, j]
            edistance[i, j] = edistance[j, i] = np.float32(value)

    return pair_means, edistance


def test_pertpy_edistance_matches_cpu_reference(small_adata: AnnData) -> None:
    result = pertpy_edistance(small_adata, groupby="group", obsm_key="X_pca")

    assert isinstance(result, EDistanceResult)
    assert result.distances_var is None

    _, cpu_edistance = _compute_cpu_reference(small_adata, "X_pca", "group")

    assert result.distances.shape == cpu_edistance.shape
    np.testing.assert_allclose(result.distances.values, cpu_edistance, atol=1e-5)
    assert np.allclose(result.distances.values, result.distances.values.T)


def test_pertpy_edistance_inplace_populates_uns(small_adata: AnnData) -> None:
    key = "group_pairwise_edistance"
    result = pertpy_edistance(
        small_adata,
        groupby="group",
        obsm_key="X_pca",
        inplace=True,
    )

    assert isinstance(result, EDistanceResult)
    assert key in small_adata.uns
    stored = small_adata.uns[key]
    assert set(stored.keys()) == {"distances", "distances_var"}
    np.testing.assert_allclose(stored["distances"].values, result.distances.values)
    assert stored["distances_var"] is None


def test_pertpy_edistance_bootstrap_returns_variance(small_adata: AnnData) -> None:
    result = pertpy_edistance(
        small_adata,
        groupby="group",
        obsm_key="X_pca",
        bootstrap=True,
        n_bootstrap=8,
        random_state=11,
    )

    assert isinstance(result, EDistanceResult)
    assert result.distances_var is not None
    assert result.distances.shape == result.distances_var.shape
    assert np.allclose(result.distances.values, result.distances.values.T)
    assert np.allclose(result.distances_var.values, result.distances_var.values.T)
    assert np.all(result.distances_var.values >= 0)


def test_pertpy_edistance_requires_categorical_obs(small_adata: AnnData) -> None:
    bad = small_adata.copy()
    bad.obs["group"] = bad.obs["group"].astype(str)

    with pytest.raises(TypeError):
        pertpy_edistance(bad, groupby="group", obsm_key="X_pca")


@pytest.mark.parametrize("missing_key", ["missing", "other"])
def test_pertpy_edistance_missing_group_raises(
    small_adata: AnnData, missing_key: str
) -> None:
    with pytest.raises(KeyError):
        pertpy_edistance(small_adata, groupby=missing_key, obsm_key="X_pca")
