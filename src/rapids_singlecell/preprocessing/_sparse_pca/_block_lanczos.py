"""
Block Krylov SVD for sparse matrices on GPU.

Based on:
    Tomás, Quintana-Ortí, Anzt (2024)
    "Fast Truncated SVD of Sparse and Dense Matrices on Graphics Processors"
    https://arxiv.org/abs/2403.06218

This implements a block Krylov subspace method with:
- CholeskyQR2 for orthogonalization (Algorithm 4 from the paper)
- Block power iteration for efficient GPU execution
"""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.linalg as cpla
import cupyx.scipy.sparse as cpsparse

from ._operators import MeanCenteredOperator


def _matvec(A, V):
    """Matrix-vector/matrix product handling sparse and operators."""
    if hasattr(A, "dot") and not isinstance(A, cp.ndarray) and not cpsparse.issparse(A):
        return A.dot(V)
    return A @ V


def _matvec_T(A, V):
    """Transposed matrix product."""
    if hasattr(A, "T"):
        AT = A.T
        if hasattr(AT, "dot") and not isinstance(AT, cp.ndarray):
            return AT.dot(V)
        return AT @ V
    return A.T @ V


def _cholesky_qr2(Q: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    CholeskyQR2 orthogonalization (Algorithm 4 from Tomás 2024).

    More numerically stable than single CholeskyQR, and GPU-efficient
    as it uses GEMM operations.

    Parameters
    ----------
    Q : cupy.ndarray of shape (q, b)
        Matrix to orthogonalize.

    Returns
    -------
    Q : cupy.ndarray of shape (q, b)
        Orthonormalized matrix.
    R : cupy.ndarray of shape (b, b)
        Upper triangular factor.
    """
    # First pass
    W = Q.T @ Q  # b x b
    try:
        L = cp.linalg.cholesky(W)  # W = L @ L.T
    except cp.linalg.LinAlgError:
        # Fallback to QR if Cholesky fails
        Q, R = cp.linalg.qr(Q, mode="reduced")
        return Q, R

    Q = cpla.solve_triangular(L, Q.T, lower=True).T  # Q = Q @ L^{-T}

    # Second pass for numerical stability
    W_bar = Q.T @ Q
    try:
        L_bar = cp.linalg.cholesky(W_bar)
    except cp.linalg.LinAlgError:
        Q, R = cp.linalg.qr(Q, mode="reduced")
        return Q, R

    Q = cpla.solve_triangular(L_bar, Q.T, lower=True).T

    # R = L.T @ L_bar.T
    R = L.T @ L_bar.T
    return Q, R


def gpu_block_lanczos_svd(
    A,
    k: int,
    *,
    block_size: int | None = None,
    n_iter: int = 2,
    random_state: int | None = 0,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Compute truncated SVD using block Krylov subspace method.

    Uses block power iteration with CholeskyQR2 orthogonalization,
    following the approach in Tomás et al. (2024) for GPU efficiency.

    Parameters
    ----------
    A : sparse matrix, ndarray, or linear operator
        Matrix to decompose of shape (m, n).
    k : int
        Number of singular values/vectors to compute.
    block_size : int or None, default=None
        Block size. If None, uses k + 10.
    n_iter : int, default=2
        Number of block power iterations. More iterations improve
        accuracy for matrices with slowly decaying singular values.
    random_state : int or None, default=0
        Random seed.

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

    # Set block size
    if block_size is None:
        block_size = k + 10
    block_size = max(block_size, k)

    # Generate random starting matrix
    Omega = rng.standard_normal(size=(n, block_size)).astype(dtype)

    # Build Krylov subspace via block power iteration
    # Y = A @ Omega
    Y = _matvec(A, Omega)
    Y, _ = _cholesky_qr2(Y)

    # Power iterations: Y = (A @ A.T)^q @ Y
    for _ in range(n_iter):
        # Z = A.T @ Y
        Z = _matvec_T(A, Y)
        Z, _ = _cholesky_qr2(Z)
        # Y = A @ Z
        Y = _matvec(A, Z)
        Y, _ = _cholesky_qr2(Y)

    # Y now spans approximate range of A
    # Compute B = Y.T @ A
    Q = Y
    B = _matvec_T(A, Q).T  # B = Q.T @ A = (A.T @ Q).T

    # SVD of the small matrix B
    Uhat, s, Vt = cp.linalg.svd(B, full_matrices=False)

    # Recover U
    U = Q @ Uhat

    # Truncate
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]

    # Sign correction
    max_abs_cols = cp.argmax(cp.abs(U), axis=0)
    signs = cp.sign(U[max_abs_cols, cp.arange(k)])
    signs[signs == 0] = 1
    U *= signs
    Vt *= signs[:, cp.newaxis]

    return U, s, Vt


class PCA_block_lanczos:
    """
    PCA using block Krylov SVD with CholeskyQR2 (Tomás et al. 2024).

    This is optimized for GPU execution using CholeskyQR2
    orthogonalization which leverages efficient GEMM operations.

    Parameters
    ----------
    n_components : int
        Number of principal components.
    block_size : int or None, default=None
        Block size. None = auto (k + 10).
    n_iter : int, default=2
        Power iterations. More = better accuracy.
    random_state : int or None, default=0
        Random seed.
    """

    def __init__(
        self,
        n_components: int,
        *,
        block_size: int | None = None,
        n_iter: int = 2,
        random_state: int | None = 0,
    ) -> None:
        self.n_components = n_components
        self.block_size = block_size
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X):
        """Fit the PCA model."""
        from rapids_singlecell.preprocessing._utils import _get_mean_var

        self.n_components_ = self.n_components
        self.n_samples_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.dtype = X.dtype

        # Compute mean
        self.mean_, _ = _get_mean_var(X, axis=0)
        self.mean_ = self.mean_.astype(X.dtype)

        # Mean-centered operator
        X_centered = MeanCenteredOperator(X, self.mean_)

        U, S, Vt = gpu_block_lanczos_svd(
            X_centered,
            k=self.n_components_,
            block_size=self.block_size,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        self.components_ = Vt
        self.explained_variance_ = (S**2) / (self.n_samples_ - 1)

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
