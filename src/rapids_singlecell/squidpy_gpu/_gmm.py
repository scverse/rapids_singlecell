"""Minimal GMM (full covariance) for the cellcharter niche flavor.

Mirrors :class:`sklearn.mixture.GaussianMixture` with
``covariance_type="full"``. Two init strategies are exposed: ``"kmeans"`` (default,
cuML KMeans warm-start) and ``"random_from_data"`` (sklearn-equivalent for parity
testing). The default ``"auto"`` backend uses a nanobind/CUDA EM path when the
compiled extension is available, with a CuPy fallback for environments without
compiled extensions.

Implementation notes
--------------------
- CUDA E-step uses a cached precision Cholesky (computed once per M-step) and a
  custom Mahalanobis + responsibility kernel for the common <=64-PC case.
- CUDA M-step reuses preallocated workspaces and computes full covariances from
  upper-triangle tiled reductions.
- ``_precision_cholesky`` is a batched ``inv``+``cholesky`` — no Python K-loop.
- Convergence is the change in mean log-likelihood.
"""

from __future__ import annotations

import importlib
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
    backend: Literal["auto", "cupy", "cuda"] = "auto",
    kmeans_n_init: int = 1,
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
    backend
        ``"auto"`` (default) uses the nanobind/CUDA EM backend when the compiled
        extension is available, otherwise falls back to CuPy. ``"cupy"`` uses
        CuPy + cuBLAS for the covariance update. ``"cuda"`` forces the custom
        CUDA kernels for the E-step and M-step reductions.
    kmeans_n_init
        Number of cuML KMeans restarts for ``init="kmeans"``. The default ``1``
        matches sklearn's GaussianMixture default and keeps cellcharter fast;
        increase this for difficult or noisy initialization landscapes.
    """
    if backend not in {"auto", "cupy", "cuda"}:
        raise ValueError("backend must be one of 'auto', 'cupy', or 'cuda'.")
    if int(kmeans_n_init) < 1:
        raise ValueError("kmeans_n_init must be >= 1.")

    X = cp.ascontiguousarray(X)
    K = int(n_components)
    backend = _resolve_backend(backend)

    if backend == "cuda" and X.shape[1] <= 64:
        return _fit_predict_cuda(
            X,
            K,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
            init=init,
            kmeans_n_init=int(kmeans_n_init),
        )

    weights, means, covariances = _initialize(
        X, K, random_state, reg_covar, init, int(kmeans_n_init)
    )
    prec_chol, log_det_prec_half = _precision_cholesky(covariances)

    prev_ll = -cp.inf
    converged = False
    for _ in range(max_iter):
        resp, ll = _e_step(
            X, weights, means, prec_chol, log_det_prec_half, backend=backend
        )
        if cp.abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll
        weights, means, covariances = _m_step(X, resp, reg_covar, backend=backend)
        prec_chol, log_det_prec_half = _precision_cholesky(covariances)

    if not converged:
        resp, _ = _e_step(
            X, weights, means, prec_chol, log_det_prec_half, backend=backend
        )
    return resp.argmax(axis=1).astype(cp.int32)


def _fit_predict_cuda(
    X: cp.ndarray,
    K: int,
    *,
    random_state: int,
    max_iter: int,
    tol: float,
    reg_covar: float,
    init: str,
    kmeans_n_init: int,
) -> cp.ndarray:
    weights, means, covariances = _initialize(
        X, K, random_state, reg_covar, init, kmeans_n_init
    )
    weights = cp.ascontiguousarray(weights)
    means = cp.ascontiguousarray(means)
    covariances = cp.ascontiguousarray(covariances)
    workspace = _GMMCudaWorkspace(X, K)
    prec_chol, log_det_prec_half = _precision_cholesky(covariances)

    prev_ll = -np.inf
    converged = False
    for _ in range(max_iter):
        resp, ll = workspace.e_step(weights, means, prec_chol, log_det_prec_half)
        ll_value = float(ll)
        if abs(ll_value - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll_value
        workspace.m_step(resp, weights, means, covariances, reg_covar)
        prec_chol, log_det_prec_half = _precision_cholesky(covariances)

    if not converged:
        resp, _ = workspace.e_step(weights, means, prec_chol, log_det_prec_half)
    return resp.argmax(axis=1).astype(cp.int32)


class _GMMCudaWorkspace:
    def __init__(self, X: cp.ndarray, K: int):
        n, d = X.shape
        self._gc = _get_gmm_cuda()
        self.X = X
        self.n = int(n)
        self.d = int(d)
        self.K = int(K)
        self.stream = cp.cuda.get_current_stream().ptr
        self.log_prob = cp.empty((n, K), dtype=X.dtype)
        self.resp = cp.empty((n, K), dtype=X.dtype)
        self.ll_per_cell = cp.empty(n, dtype=X.dtype)
        self.N_k = cp.empty(K, dtype=X.dtype)
        self.num = cp.empty((K, d), dtype=X.dtype)

    def e_step(
        self,
        weights: cp.ndarray,
        means: cp.ndarray,
        prec_chol: cp.ndarray,
        log_det_half: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        self._gc.e_step(
            self.X,
            cp.ascontiguousarray(weights.astype(self.X.dtype, copy=False)),
            cp.ascontiguousarray(means),
            cp.ascontiguousarray(prec_chol),
            cp.ascontiguousarray(log_det_half.astype(self.X.dtype, copy=False)),
            self.log_prob,
            self.resp,
            self.ll_per_cell,
            n=self.n,
            d=self.d,
            K=self.K,
            stream=self.stream,
        )
        return self.resp, self.ll_per_cell.mean()

    def m_step(
        self,
        resp: cp.ndarray,
        weights: cp.ndarray,
        means: cp.ndarray,
        covariances: cp.ndarray,
        reg_covar: float,
    ) -> None:
        self._gc.m_step(
            cp.ascontiguousarray(resp),
            self.X,
            weights,
            means,
            covariances,
            self.N_k,
            self.num,
            n=self.n,
            d=self.d,
            K=self.K,
            reg_covar=float(reg_covar),
            stream=self.stream,
        )


def _resolve_backend(backend: str) -> str:
    if backend == "cupy":
        return backend
    try:
        _get_gmm_cuda()
    except ImportError:
        if backend == "cuda":
            raise
        return "cupy"
    return "cuda"


def _get_gmm_cuda():
    try:
        return importlib.import_module("rapids_singlecell._cuda._gmm_cuda")
    except ImportError as err:
        raise ImportError(
            "The _gmm_cuda extension is not available. Build rapids-singlecell "
            "with CUDA extensions or use backend='cupy'."
        ) from err


def _initialize(
    X: cp.ndarray,
    K: int,
    *,
    random_state: int,
    reg_covar: float,
    init: str,
    kmeans_n_init: int,
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

    km = KMeans(
        n_clusters=K,
        random_state=random_state,
        n_init=int(kmeans_n_init),
        max_iter=100,
    )
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
    *,
    backend: str = "cupy",
) -> tuple[cp.ndarray, cp.ndarray]:
    n, d = X.shape
    if backend == "cuda" and d <= 64:
        _gc = _get_gmm_cuda()

        K = means.shape[0]
        log_prob = cp.empty((n, K), dtype=X.dtype)
        resp = cp.empty((n, K), dtype=X.dtype)
        ll_per_cell = cp.empty(n, dtype=X.dtype)
        _gc.e_step(
            cp.ascontiguousarray(X),
            cp.ascontiguousarray(weights.astype(X.dtype)),
            cp.ascontiguousarray(means),
            cp.ascontiguousarray(prec_chol),
            cp.ascontiguousarray(log_det_half.astype(X.dtype)),
            log_prob,
            resp,
            ll_per_cell,
            n=int(n),
            d=int(d),
            K=int(K),
            stream=cp.cuda.get_current_stream().ptr,
        )
        return resp, ll_per_cell.mean()
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
    resp = cp.exp(log_prob - log_total)
    return resp, log_total.mean()


def _m_step(
    X: cp.ndarray,
    resp: cp.ndarray,
    reg_covar: float,
    *,
    backend: str = "cupy",
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    n, d = X.shape
    K = resp.shape[1]

    if backend == "cuda":
        _gc = _get_gmm_cuda()

        weights = cp.empty(K, dtype=X.dtype)
        means = cp.empty((K, d), dtype=X.dtype)
        covariances = cp.empty((K, d, d), dtype=X.dtype)
        N_k_ws = cp.empty(K, dtype=X.dtype)
        num_ws = cp.empty((K, d), dtype=X.dtype)
        _gc.m_step(
            cp.ascontiguousarray(resp),
            cp.ascontiguousarray(X),
            weights,
            means,
            covariances,
            N_k_ws,
            num_ws,
            n=int(n),
            d=int(d),
            K=int(K),
            reg_covar=float(reg_covar),
            stream=cp.cuda.get_current_stream().ptr,
        )
        return weights, means, covariances

    N_k = resp.sum(axis=0) + 10.0 * cp.finfo(X.dtype).eps  # (K,)
    weights = N_k / n
    means = (resp.T @ X) / N_k[:, None]
    covariances = cp.empty((K, d, d), dtype=X.dtype)
    eye_reg = reg_covar * cp.eye(d, dtype=X.dtype)
    for k in range(K):
        diff = X - means[k]
        covariances[k] = ((resp[:, k : k + 1] * diff).T @ diff) / N_k[k] + eye_reg
    return weights, means, covariances
