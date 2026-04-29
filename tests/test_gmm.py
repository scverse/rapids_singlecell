from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score as ARI

from rapids_singlecell.squidpy_gpu._gmm import gmm_fit_predict


def _well_separated(n_per: int, K: int, d: int, sep: float, seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(K, d))
    X = np.vstack(
        [rng.normal(loc=c, scale=1.0, size=(n_per, d)) for c in centers]
    ).astype(np.float32)
    y = np.repeat(np.arange(K), n_per)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_kmeans_init_recovers_well_separated_clusters():
    """kmeans init should land at near-truth on well-separated data."""
    X_np, y = _well_separated(n_per=300, K=5, d=20, sep=6.0, seed=0)
    labels = cp.asnumpy(
        gmm_fit_predict(cp.asarray(X_np), n_components=5, random_state=0, init="kmeans")
    )
    assert ARI(y, labels) >= 0.95


def test_random_from_data_init_runs():
    """random_from_data may land at a worse local optimum than kmeans, but should
    still produce a non-trivial partition on well-separated data."""
    X_np, y = _well_separated(n_per=300, K=5, d=20, sep=6.0, seed=0)
    labels = cp.asnumpy(
        gmm_fit_predict(
            cp.asarray(X_np), n_components=5, random_state=0, init="random_from_data"
        )
    )
    assert ARI(y, labels) >= 0.4
    assert len(set(labels.tolist())) >= 2


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8)).astype(np.float32)
    labels = gmm_fit_predict(cp.asarray(X), n_components=4, random_state=0)
    assert labels.shape == (500,)
    assert labels.dtype == cp.int32
    assert int(labels.max()) < 4
    assert int(labels.min()) >= 0


@pytest.mark.parametrize("init", ["kmeans", "random_from_data"])
def test_determinism_same_seed(init):
    rng = np.random.default_rng(1)
    X = cp.asarray(rng.standard_normal((800, 10)).astype(np.float32))
    a = cp.asnumpy(gmm_fit_predict(X, n_components=5, random_state=42, init=init))
    b = cp.asnumpy(gmm_fit_predict(X, n_components=5, random_state=42, init=init))
    np.testing.assert_array_equal(a, b)


def test_invalid_init_raises():
    X = cp.asarray(np.zeros((100, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="init"):
        gmm_fit_predict(X, n_components=3, init="bogus")


def test_n_components_one_returns_single_label():
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((200, 4)).astype(np.float32))
    labels = cp.asnumpy(gmm_fit_predict(X, n_components=1, random_state=0))
    assert set(labels.tolist()) == {0}


def test_float64_input_accepted():
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((300, 6)).astype(np.float64))
    labels = gmm_fit_predict(X, n_components=3, random_state=0)
    assert labels.shape == (300,)
