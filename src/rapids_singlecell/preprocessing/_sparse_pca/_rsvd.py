"""
GPU-accelerated Randomized SVD.

Based on the algorithm from:
* Halko, Martinsson, Tropp (2009) "Finding structure with randomness"
  https://arxiv.org/abs/0909.4061

Optimized for GPU execution with CuPy.
"""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cpsparse

from ._operators import MeanCenteredOperator


def _matvec(A, v):
    """Matrix-vector product handling sparse, dense, and operator inputs."""
    if hasattr(A, "dot") and not isinstance(A, cp.ndarray) and not cpsparse.issparse(A):
        return A.dot(v)
    return A @ v


def _matvec_T(A, v):
    """Transposed matrix-vector product."""
    if hasattr(A, "T"):
        AT = A.T
        if hasattr(AT, "dot") and not isinstance(AT, cp.ndarray):
            return AT.dot(v)
        return AT @ v
    return A.T @ v


def gpu_randomized_svd(
    A,
    k: int,
    *,
    n_oversamples: int = 10,
    n_iter: int = 4,
    random_state: int | None = 0,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Compute truncated randomized SVD.

    Uses Algorithm 4.4 from Halko et al. (2009) with power iterations.

    Parameters
    ----------
    A : sparse matrix, ndarray, or linear operator
        Matrix to decompose of shape (m, n).
    k : int
        Number of singular values/vectors to compute.
    n_oversamples : int, default=10
        Additional random vectors for better approximation.
    n_iter : int, default=4
        Number of power iterations for accuracy.
    random_state : int or None, default=0
        Random seed for reproducibility.

    Returns
    -------
    U : cupy.ndarray of shape (m, k)
        Left singular vectors.
    s : cupy.ndarray of shape (k,)
        Singular values in descending order.
    Vt : cupy.ndarray of shape (k, n)
        Right singular vectors.
    """
    m, n = A.shape
    dtype = A.dtype
    rng = cp.random.RandomState(random_state if random_state is not None else 0)

    # Number of random vectors
    l = k + n_oversamples

    # Step 1: Random projection
    # Generate random test matrix
    Omega = rng.standard_normal(size=(n, l)).astype(dtype)

    # Step 2: Power iteration for better accuracy
    # Y = (A @ A.T)^n_iter @ A @ Omega
    Y = _matvec(A, Omega)

    for _ in range(n_iter):
        # Orthogonalize for numerical stability
        Y, _ = cp.linalg.qr(Y, mode="reduced")
        Z = _matvec_T(A, Y)
        Z, _ = cp.linalg.qr(Z, mode="reduced")
        Y = _matvec(A, Z)

    # Step 3: Orthonormalize
    Q, _ = cp.linalg.qr(Y, mode="reduced")

    # Step 4: Form B = Q.T @ A and compute its SVD
    B = _matvec_T(A, Q).T  # B = Q.T @ A, but computed as (A.T @ Q).T

    # SVD of small matrix B
    Uhat, s, Vt = cp.linalg.svd(B, full_matrices=False)

    # Step 5: Recover U
    U = Q @ Uhat

    # Truncate to k components
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]

    # Sign correction for deterministic output
    max_abs_cols = cp.argmax(cp.abs(U), axis=0)
    signs = cp.sign(U[max_abs_cols, cp.arange(k)])
    signs[signs == 0] = 1
    U *= signs
    Vt *= signs[:, cp.newaxis]

    return U, s, Vt


class PCA_rsvd:
    """
    PCA using Randomized SVD.

    Fast approximate PCA for sparse matrices. Best when only a small number
    of components are needed relative to matrix dimensions.

    Parameters
    ----------
    n_components : int
        Number of principal components to compute.
    n_oversamples : int, default=10
        Extra random vectors for accuracy. Higher = more accurate but slower.
    n_iter : int, default=4
        Power iterations. Higher = more accurate for matrices with slow
        singular value decay. Use 2-4 for most cases.
    random_state : int or None, default=0
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int,
        *,
        n_oversamples: int = 10,
        n_iter: int = 4,
        random_state: int | None = 0,
    ) -> None:
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X):
        """Fit the PCA model."""
        from rapids_singlecell.preprocessing._utils import _get_mean_var

        self.n_components_ = self.n_components
        self.n_samples_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.dtype = X.dtype

        # Compute mean for centering
        self.mean_, _ = _get_mean_var(X, axis=0)
        self.mean_ = self.mean_.astype(X.dtype)

        # Use mean-centered operator
        X_centered = MeanCenteredOperator(X, self.mean_)

        U, S, Vt = gpu_randomized_svd(
            X_centered,
            k=self.n_components_,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        self.components_ = Vt
        self.explained_variance_ = (S**2) / (self.n_samples_ - 1)

        # Total variance
        _, var_x = _get_mean_var(X, axis=0)
        total_variance = cp.sum(var_x)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X):
        """Apply dimensionality reduction."""
        X_transformed = X.dot(self.components_.T)
        mean_projection = cp.dot(self.mean_, self.components_.T)
        X_transformed -= mean_projection
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform."""
        return self.fit(X).transform(X)
