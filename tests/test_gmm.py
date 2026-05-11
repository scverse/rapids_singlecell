from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.mixture import GaussianMixture

from rapids_singlecell.squidpy_gpu._gmm import (
    _e_step,
    _GMMCudaWorkspace,
    _m_step,
    _precision_cholesky,
    _resolve_backend,
    _use_cuda_cublas_e_step,
    _use_cuda_e_step,
    gmm_fit_predict,
)


def _well_separated(n_per: int, K: int, d: int, sep: float, seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(K, d))
    X = np.vstack(
        [rng.normal(loc=c, scale=1.0, size=(n_per, d)) for c in centers]
    ).astype(np.float32)
    y = np.repeat(np.arange(K), n_per)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _pca_like_mixture(n_per: int, K: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(K, d))
    rows = []
    for k in range(K):
        # A compact low-rank perturbation of the identity gives each synthetic
        # cell state its own correlated PCA-space geometry.
        factors = rng.normal(scale=0.3 + 0.05 * k, size=(d, 3))
        cov = np.eye(d) * (0.35 + 0.03 * k) + factors @ factors.T
        rows.append(rng.multivariate_normal(centers[k], cov, size=n_per))
    X = np.vstack(rows).astype(np.float32)
    return np.ascontiguousarray(X[rng.permutation(len(X))])


def test_kmeans_init_recovers_well_separated_clusters():
    """kmeans init should land at near-truth on well-separated data."""
    X_np, y = _well_separated(n_per=300, K=5, d=20, sep=6.0, seed=0)
    labels = cp.asnumpy(
        gmm_fit_predict(cp.asarray(X_np), n_components=5, random_state=0, init="kmeans")
    )
    assert ARI(y, labels) >= 0.95


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_full_cov_gmm_matches_sklearn_on_singlecell_embedding(backend):
    if backend == "cuda" and _resolve_backend("auto") != "cuda":
        pytest.skip("_gmm_cuda extension is not available")

    X = _pca_like_mixture(n_per=250, K=5, d=16, seed=7)
    sk_labels = GaussianMixture(
        n_components=5,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        random_state=0,
    ).fit_predict(X)
    rsc_labels = cp.asnumpy(
        gmm_fit_predict(
            cp.asarray(X),
            n_components=5,
            random_state=0,
            init="kmeans",
            backend=backend,
            kmeans_n_init=1,
        )
    )

    assert ARI(sk_labels, rsc_labels) >= 0.99


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


def test_invalid_backend_raises():
    X = cp.asarray(np.zeros((100, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="backend"):
        gmm_fit_predict(X, n_components=3, backend="bogus")


def test_invalid_kmeans_n_init_raises():
    X = cp.asarray(np.zeros((100, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="kmeans_n_init"):
        gmm_fit_predict(X, n_components=3, kmeans_n_init=0)


def test_auto_backend_uses_cuda_when_available():
    if _resolve_backend("auto") != "cuda":
        pytest.skip("_gmm_cuda extension is not available")

    assert _resolve_backend("auto") == "cuda"
    assert _resolve_backend("cuda") == "cuda"


def test_cuda_e_step_routing_uses_fused_kernels_for_wide_embeddings():
    assert _use_cuda_e_step(16)
    assert _use_cuda_e_step(32)
    assert _use_cuda_e_step(50)
    assert _use_cuda_e_step(64)
    assert _use_cuda_e_step(80)
    assert _use_cuda_e_step(96)
    assert _use_cuda_e_step(128)
    assert _use_cuda_e_step(256)
    assert _use_cuda_e_step(384)
    assert _use_cuda_e_step(512)
    assert not _use_cuda_e_step(768)
    assert _use_cuda_e_step(512, cp.float32)
    assert not _use_cuda_e_step(512, cp.float64)
    assert _use_cuda_e_step(64, cp.float64)
    assert not _use_cuda_cublas_e_step(256, cp.float32)
    assert _use_cuda_cublas_e_step(384, cp.float32)
    assert _use_cuda_cublas_e_step(512, cp.float32)
    assert not _use_cuda_cublas_e_step(512, cp.float64)


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


def test_cuda_backend_matches_cupy_steps():
    if _resolve_backend("auto") != "cuda":
        pytest.skip("_gmm_cuda extension is not available")

    rng = cp.random.RandomState(0)
    n, d, K = 40_000, 6, 3  # large enough to exercise the cuBLAS M-step path
    X = rng.standard_normal((n, d), dtype=cp.float32)
    logits = rng.standard_normal((n, K), dtype=cp.float32)
    resp = cp.exp(logits - cp.log(cp.exp(logits).sum(axis=1, keepdims=True)))

    w_c, m_c, cov_c = _m_step(X, resp, 1e-6, backend="cupy")
    w_g, m_g, cov_g = _m_step(X, resp, 1e-6, backend="cuda")

    assert cp.max(cp.abs(w_c - w_g)).item() < 1e-5
    assert cp.max(cp.abs(m_c - m_g)).item() < 1e-5
    assert cp.max(cp.abs(cov_c - cov_g)).item() < 1e-4

    weights = cp.full(K, 1 / K, dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = A @ A.transpose(0, 2, 1) + cp.eye(d, dtype=cp.float32)[None] * 0.1
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _e_step(X, weights, means, prec_chol, log_det_half, backend="cupy")
    r_g, ll_g = _e_step(X, weights, means, prec_chol, log_det_half, backend="cuda")
    r_f, ll_f = _GMMCudaWorkspace(X, K).e_step_fused(
        weights, means, prec_chol, log_det_half
    )

    assert cp.max(cp.abs(r_c - r_g)).item() < 1e-4
    assert cp.abs(ll_c - ll_g).item() < 1e-4
    assert cp.max(cp.abs(r_c - r_f)).item() < 1e-4
    assert cp.abs(ll_c - ll_f).item() < 1e-4


def test_cuda_large_e_step_matches_cupy_for_large_feature_count():
    if _resolve_backend("auto") != "cuda":
        pytest.skip("_gmm_cuda extension is not available")

    rng = cp.random.RandomState(2)
    n, d, K = 2048, 96, 4
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.asarray([0.15, 0.2, 0.3, 0.35], dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float32)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _e_step(X, weights, means, prec_chol, log_det_half, backend="cupy")
    r_g, ll_g = _e_step(X, weights, means, prec_chol, log_det_half, backend="cuda")

    assert cp.max(cp.abs(r_c - r_g)).item() < 5e-4
    assert cp.abs(ll_c - ll_g).item() < 5e-4


def test_cuda_512_e_step_matches_cupy_for_cublas_route():
    if _resolve_backend("auto") != "cuda":
        pytest.skip("_gmm_cuda extension is not available")

    rng = cp.random.RandomState(5)
    n, d, K = 384, 512, 3
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.asarray([0.2, 0.3, 0.5], dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float32)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _e_step(X, weights, means, prec_chol, log_det_half, backend="cupy")
    r_g, ll_g = _e_step(X, weights, means, prec_chol, log_det_half, backend="cuda")
    r_b, ll_b = _GMMCudaWorkspace(X, K).e_step_cublas(
        weights, means, prec_chol, log_det_half
    )

    assert cp.max(cp.abs(r_c - r_g)).item() < 1e-3
    assert cp.abs(ll_c - ll_g).item() < 1e-3
    assert cp.max(cp.abs(r_c - r_b)).item() < 1e-3
    assert cp.abs(ll_c - ll_b).item() < 1e-3


def test_cuda_fixed_e_step_matches_cupy_for_medium_regime():
    if _resolve_backend("auto") != "cuda":
        pytest.skip("_gmm_cuda extension is not available")

    rng = cp.random.RandomState(4)
    n, d, K = 1024, 16, 8
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.full(K, 1 / K, dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float32)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _e_step(X, weights, means, prec_chol, log_det_half, backend="cupy")
    r_g, ll_g = _e_step(X, weights, means, prec_chol, log_det_half, backend="cuda")

    assert cp.max(cp.abs(r_c - r_g)).item() < 5e-4
    assert cp.abs(ll_c - ll_g).item() < 5e-4


def test_cuda_fused_e_step_matches_cupy_for_50_pc_regime():
    if _resolve_backend("auto") != "cuda":
        pytest.skip("_gmm_cuda extension is not available")

    rng = cp.random.RandomState(6)
    n, d, K = 1024, 50, 12
    X = rng.standard_normal((n, d), dtype=cp.float32)
    weights = cp.full(K, 1 / K, dtype=cp.float32)
    means = rng.standard_normal((K, d), dtype=cp.float32)
    A = rng.standard_normal((K, d, d), dtype=cp.float32)
    cov = (A @ A.transpose(0, 2, 1)) / d + cp.eye(d, dtype=cp.float32)[None] * 0.5
    prec_chol, log_det_half = _precision_cholesky(cov)

    r_c, ll_c = _e_step(X, weights, means, prec_chol, log_det_half, backend="cupy")
    r_d, ll_d = _GMMCudaWorkspace(X, K).e_step(weights, means, prec_chol, log_det_half)

    assert cp.max(cp.abs(r_c - r_d)).item() < 5e-4
    assert cp.abs(ll_c - ll_d).item() < 5e-4


def test_cuda_backend_runs_large_feature_count():
    if _resolve_backend("auto") != "cuda":
        pytest.skip("_gmm_cuda extension is not available")

    rng = np.random.default_rng(3)
    X = cp.asarray(rng.standard_normal((360, 80)).astype(np.float32))
    labels = gmm_fit_predict(
        X,
        n_components=3,
        random_state=0,
        max_iter=2,
        reg_covar=1e-2,
        init="random_from_data",
        backend="cuda",
    )

    assert labels.shape == (360,)
    assert labels.dtype == cp.int32
