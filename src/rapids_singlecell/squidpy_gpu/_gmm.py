"""Minimal GMM (full covariance) for the cellcharter niche flavor.

Mirrors :class:`sklearn.mixture.GaussianMixture` with
``covariance_type="full"``. Two init strategies are exposed: ``"kmeans"`` (default,
cuML KMeans warm-start) and ``"random_from_data"`` (sklearn-equivalent for parity
testing). A future cuML-backed or fused-CUDA GMM can replace this.

Implementation notes
--------------------
- E-step uses a cached precision Cholesky (computed once per M-step) and a
  per-component dense matmul. This avoids the per-component triangular solve.
- ``_precision_cholesky`` is a batched ``inv``+``cholesky`` — no Python K-loop.
- ``cupyx.scipy.special.logsumexp`` for the stable softmax over components.
- Convergence is the change in mean log-likelihood.
"""

from __future__ import annotations

from typing import Literal

import cupy as cp
import numpy as np
from cupyx.scipy.special import logsumexp

_LOG_2PI = float(np.log(2.0 * np.pi))


def gmm_fit_predict(
    X: cp.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
    max_iter: int = 100,
    tol: float = 1e-3,
    reg_covar: float = 1e-6,
    init: Literal["kmeans", "random_from_data"] = "kmeans",
) -> cp.ndarray:
    """Fit a full-covariance GMM and return cluster labels.

    Parameters
    ----------
    X
        Cupy array, shape ``(n_samples, n_features)``, float32 or float64.
    n_components
        Number of mixture components ``K``.
    random_state
        Seed for initialization.
    max_iter
        Maximum EM iterations.
    tol
        Convergence threshold on the change in mean log-likelihood.
    reg_covar
        Regularization added to each component covariance diagonal.
    init
        ``"kmeans"`` (default) uses cuML KMeans for warm-start; usually much
        better than ``"random_from_data"``, which mirrors sklearn for parity.
    """
    X = cp.ascontiguousarray(X)
    n_samples, _ = X.shape
    K = int(n_components)

    weights, means, covariances = _initialize(X, K, random_state, reg_covar, init)
    prec_chol, log_det_prec_half = _precision_cholesky(covariances)

    prev_ll = -cp.inf
    converged = False
    for _ in range(max_iter):
        log_resp, ll = _e_step(X, weights, means, prec_chol, log_det_prec_half)
        if cp.abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll
        weights, means, covariances = _m_step(X, log_resp, reg_covar)
        prec_chol, log_det_prec_half = _precision_cholesky(covariances)

    if not converged:
        log_resp, _ = _e_step(X, weights, means, prec_chol, log_det_prec_half)
    return log_resp.argmax(axis=1).astype(cp.int32)


def _initialize(
    X: cp.ndarray,
    K: int,
    random_state: int,
    reg_covar: float,
    init: str,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    n, d = X.shape
    eye_reg = reg_covar * cp.eye(d, dtype=X.dtype)

    if init == "random_from_data":
        # sklearn parity: pick K rows as means, equal weights, reg-only covariance.
        rng = np.random.default_rng(random_state)
        idx = cp.asarray(rng.choice(n, size=K, replace=False))
        means = X[idx].copy()
        weights = cp.full(K, 1.0 / K, dtype=X.dtype)
        covariances = cp.broadcast_to(eye_reg, (K, d, d)).copy()
        return weights, means, covariances

    if init != "kmeans":
        raise ValueError(f"init must be 'kmeans' or 'random_from_data', got {init!r}")

    from cuml.cluster import KMeans

    # Match sklearn's n_init=10 inside its GaussianMixture's kmeans init.
    # n_init=1 was empirically prone to degenerate inits on structureless data,
    # which then collapsed EM to a single component.
    km = KMeans(n_clusters=K, random_state=random_state, n_init=10, max_iter=100)
    km.fit(X)
    labels = cp.asarray(km.labels_)
    means = cp.asarray(km.cluster_centers_, dtype=X.dtype)

    weights = cp.zeros(K, dtype=X.dtype)
    covariances = cp.empty((K, d, d), dtype=X.dtype)
    for k in range(K):
        mask = labels == k
        cnt = int(mask.sum())
        if cnt == 0:
            # KMeans can return empty clusters; fall back to a tiny uniform component.
            weights[k] = 1.0 / n
            covariances[k] = eye_reg
            continue
        weights[k] = cnt / n
        diff = X[mask] - means[k]
        covariances[k] = (diff.T @ diff) / cnt + eye_reg
    return weights, means, covariances


def _precision_cholesky(
    covariances: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return ``(prec_chol, log|Σ⁻¹|/2)`` where ``prec_chol @ prec_chol.T = Σ⁻¹``.

    Two batched LAPACK calls — no Python K-loop.
    """
    precisions = cp.linalg.inv(covariances)
    prec_chol = cp.linalg.cholesky(precisions)
    log_det_half = cp.sum(cp.log(cp.diagonal(prec_chol, axis1=1, axis2=2)), axis=1)
    return prec_chol, log_det_half


@cp.fuse(kernel_name="_gmm_log_pdf_const")
def _log_pdf_const(mahal: cp.ndarray, log_det_half: cp.ndarray, half_d_log2pi):
    return -0.5 * mahal + log_det_half - half_d_log2pi


def _e_step(
    X: cp.ndarray,
    weights: cp.ndarray,
    means: cp.ndarray,
    prec_chol: cp.ndarray,
    log_det_half: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = means.shape[0]

    log_prob = cp.empty((n, K), dtype=X.dtype)
    half_d_log2pi = X.dtype.type(0.5 * d * _LOG_2PI)
    for k in range(K):
        # mahal = || (X - μ_k) @ prec_chol[k] ||²
        y = (X - means[k]) @ prec_chol[k]
        mahal = cp.einsum("ij,ij->i", y, y)
        log_prob[:, k] = _log_pdf_const(mahal, log_det_half[k], half_d_log2pi)
    log_prob = log_prob + cp.log(weights)

    log_total = logsumexp(log_prob, axis=1, keepdims=True)
    log_resp = log_prob - log_total
    return log_resp, log_total.mean()


def _m_step(
    X: cp.ndarray,
    log_resp: cp.ndarray,
    reg_covar: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = log_resp.shape[1]

    resp = cp.exp(log_resp)
    N_k = resp.sum(axis=0) + 10.0 * cp.finfo(X.dtype).eps  # (K,)

    weights = N_k / n
    means = (resp.T @ X) / N_k[:, None]

    covariances = cp.empty((K, d, d), dtype=X.dtype)
    eye_reg = reg_covar * cp.eye(d, dtype=X.dtype)
    for k in range(K):
        diff = X - means[k]
        covariances[k] = ((resp[:, k : k + 1] * diff).T @ diff) / N_k[k] + eye_reg
    return weights, means, covariances
