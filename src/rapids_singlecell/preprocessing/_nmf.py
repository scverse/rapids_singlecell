"""
GPU-accelerated Non-negative Matrix Factorization (NMF).

Two solvers are available:

- **HALS** (Hierarchical ALS, default): coordinate descent with fused CUDA
  kernels and adaptive inner sweeps.  Converges faster than MU because
  multiple essentially-free inner sweeps amortize each expensive sparse
  matrix multiply.
- **MU** (Multiplicative Updates): Lee & Seung 2001.  Simple, no custom
  kernels, but slower convergence.

Both solvers avoid materializing the full (n_cells × n_genes) reconstruction
matrix.  All intermediates are at most (n × K) or (K × K).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import issparse as cpissparse
from scipy.sparse import issparse

from rapids_singlecell.get import _get_obs_rep
from rapids_singlecell.preprocessing._sparse_pca._block_lanczos import randomized_svd

from ._utils import _check_gpu_X

if TYPE_CHECKING:
    from anndata import AnnData

_EPSILON = cp.finfo(cp.float32).eps

# ---------------------------------------------------------------------------
# HALS CUDA kernels (nanobind, templated for float32/float64)
# ---------------------------------------------------------------------------

_DEFAULT_N_INNER = 3  # inner HALS sweeps per sparse matmul


def _get_nmf_cuda():
    """Lazily import the compiled nanobind NMF CUDA module."""
    from rapids_singlecell._cuda import _nmf_cuda

    return _nmf_cuda


# ---------------------------------------------------------------------------
# Public API — AnnData wrapper
# ---------------------------------------------------------------------------


def nmf(
    adata: AnnData,
    n_components: int = 30,
    *,
    layer: str | None = None,
    init: Literal["nndsvd", "nndsvda", "nndsvdar", "random"] = "nndsvd",
    solver: Literal["hals", "mu"] = "hals",
    max_iter: int = 200,
    tol: float = 1e-4,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio: float = 0.0,
    random_state: int = 0,
    n_inner: int = _DEFAULT_N_INNER,
    use_highly_variable: bool | None = None,
) -> None:
    """Compute Non-negative Matrix Factorization (NMF) on GPU.

    Decomposes X ≈ W @ H where W (n_cells × K) and H (K × n_genes) are
    non-negative.  Never materializes the full W @ H matrix.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_components
        Number of components (K).
    layer
        Layer of ``adata`` to use. If ``None``, uses ``adata.X``.
    init
        Initialization method:

        - ``'nndsvd'``: Non-negative Double SVD. Deterministic, good
          for sparse data.
        - ``'nndsvda'``: NNDSVD with zeros filled by the data average.
        - ``'nndsvdar'``: NNDSVD with zeros filled by small random values.
        - ``'random'``: Random non-negative initialization.
    solver
        Optimization method:

        - ``'hals'``: Hierarchical Alternating Least Squares with fused
          CUDA kernels and adaptive inner sweeps.  Default.
        - ``'mu'``: Multiplicative Updates (Lee & Seung 2001).
    max_iter
        Maximum number of iterations.
    tol
        Convergence tolerance on the relative change of the Frobenius norm.
    alpha_W
        Regularization strength on W.
    alpha_H
        Regularization strength on H.
    l1_ratio
        Ratio of L1 vs L2 regularization (0 = pure L2, 1 = pure L1).
    random_state
        Random seed for reproducibility.
    use_highly_variable
        If ``True``, only use highly variable genes.
        If ``None``, uses them if available.

    Returns
    -------
    Adds fields to ``adata``:

    ``adata.obsm['X_nmf']``
        NMF cell scores (W matrix), shape (n_cells, n_components).
    ``adata.varm['NMF']``
        NMF gene loadings (H^T matrix), shape (n_genes, n_components).
    ``adata.uns['nmf']``
        Dictionary with ``'params'`` and ``'reconstruction_error'``.
    """
    X = _get_obs_rep(adata, layer=layer)
    _check_gpu_X(X)

    # Handle highly variable genes
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    if use_highly_variable:
        mask = adata.var["highly_variable"].values
        if issparse(X):
            from cuml.internals.input_utils import sparse_scipy_to_cp

            X = sparse_scipy_to_cp(X, dtype=np.float32)
        if cpissparse(X):
            X = X[:, mask]
        else:
            X = X[:, mask]
    else:
        mask = None
        if issparse(X):
            from cuml.internals.input_utils import sparse_scipy_to_cp

            X = sparse_scipy_to_cp(X, dtype=np.float32)

    if not cpissparse(X) and not isinstance(X, cp.ndarray):
        X = cp.asarray(X)
    X = X.astype(cp.float32)

    W, H, n_iter, error = run_nmf(
        X,
        n_components=n_components,
        init=init,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        random_state=random_state,
        n_inner=n_inner,
    )

    # Store results
    W_out = W.get()
    H_out = H.get()

    adata.obsm["X_nmf"] = W_out
    if mask is not None:
        H_full = np.zeros((n_components, adata.n_vars), dtype=np.float32)
        H_full[:, mask] = H_out
        adata.varm["NMF"] = H_full.T
    else:
        adata.varm["NMF"] = H_out.T

    adata.uns["nmf"] = {
        "params": {
            "n_components": n_components,
            "init": init,
            "solver": solver,
            "max_iter": max_iter,
            "tol": tol,
            "n_iter": n_iter,
        },
        "reconstruction_error": float(error),
    }


# ---------------------------------------------------------------------------
# Public API — standalone (no AnnData)
# ---------------------------------------------------------------------------


def run_nmf(
    X,
    n_components: int = 30,
    *,
    init: Literal["nndsvd", "nndsvda", "nndsvdar", "random"] = "nndsvd",
    solver: Literal["hals", "mu"] = "hals",
    max_iter: int = 200,
    tol: float = 1e-4,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio: float = 0.0,
    random_state: int = 0,
    n_inner: int = _DEFAULT_N_INNER,
) -> tuple[cp.ndarray, cp.ndarray, int, float]:
    """Run NMF on a CuPy array or sparse matrix.

    Parameters
    ----------
    X
        Input matrix (n_samples × n_features), CuPy dense or sparse CSR.
        Must be non-negative and float32.
    n_components
        Number of components (K).
    init
        Initialization method (see :func:`nmf`).
    solver
        ``'hals'`` (default) or ``'mu'``.
    max_iter
        Maximum number of iterations.
    tol
        Convergence tolerance on relative change of Frobenius norm.
    alpha_W
        Regularization strength on W.
    alpha_H
        Regularization strength on H.
    l1_ratio
        L1 vs L2 ratio (0 = pure L2, 1 = pure L1).
    random_state
        Random seed.

    Returns
    -------
    W
        Cell scores, shape (n_samples, n_components).
    H
        Gene loadings, shape (n_components, n_features).
    n_iter
        Number of iterations run.
    error
        Final reconstruction error (Frobenius norm).
    """
    if cpissparse(X):
        if X.data.min() < 0:
            raise ValueError(
                "Input matrix contains negative values. NMF requires "
                "non-negative input."
            )
    elif X.min() < 0:
        raise ValueError(
            "Input matrix contains negative values. NMF requires non-negative input."
        )

    n_samples, n_features = X.shape
    if n_components > min(n_samples, n_features):
        msg = (
            f"n_components ({n_components}) must be <= "
            f"min(n_samples, n_features) = {min(n_samples, n_features)}"
        )
        raise ValueError(msg)

    W, H = _initialize_nmf(X, n_components, init=init, random_state=random_state)

    solver_kwargs = {
        "max_iter": max_iter,
        "tol": tol,
        "alpha_W": alpha_W,
        "alpha_H": alpha_H,
        "l1_ratio": l1_ratio,
    }
    if solver == "hals":
        W, H, n_iter, error = _fit_hals(X, W, H, **solver_kwargs, n_inner=n_inner)
    elif solver == "mu":
        W, H, n_iter, error = _fit_multiplicative_update(X, W, H, **solver_kwargs)
    else:
        msg = f"Unknown solver: {solver!r}. Use 'hals' or 'mu'."
        raise ValueError(msg)

    return W, H, n_iter, error


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _initialize_nmf(
    X,
    n_components: int,
    *,
    init: str = "nndsvd",
    random_state: int = 0,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Initialize W and H for NMF."""
    n_samples, n_features = X.shape

    if init == "random":
        cp.random.seed(random_state)
        avg = cp.sqrt(cp.mean(_safe_sparse_abs_mean(X)) / n_components)
        W = avg * cp.random.uniform(size=(n_samples, n_components)).astype(X.dtype)
        H = avg * cp.random.uniform(size=(n_components, n_features)).astype(X.dtype)
        return W, H

    # NNDSVD variants: compute rank-K SVD
    U, S, Vt = randomized_svd(
        X, k=n_components, n_oversamples=10, n_iter=2, random_state=random_state
    )

    W = cp.zeros((n_samples, n_components), dtype=X.dtype)
    H = cp.zeros((n_components, n_features), dtype=X.dtype)

    # First component: use sqrt of first singular triplet
    W[:, 0] = cp.sqrt(S[0]) * cp.abs(U[:, 0])
    H[0, :] = cp.sqrt(S[0]) * cp.abs(Vt[0, :])

    for j in range(1, n_components):
        u = U[:, j]
        v = Vt[j, :]
        s = S[j]

        u_pos = cp.maximum(u, 0)
        u_neg = cp.maximum(-u, 0)
        v_pos = cp.maximum(v, 0)
        v_neg = cp.maximum(-v, 0)

        n_u_pos = cp.linalg.norm(u_pos)
        n_u_neg = cp.linalg.norm(u_neg)
        n_v_pos = cp.linalg.norm(v_pos)
        n_v_neg = cp.linalg.norm(v_neg)

        pos_term = n_u_pos * n_v_pos
        neg_term = n_u_neg * n_v_neg

        if pos_term >= neg_term:
            u_fac = u_pos / (n_u_pos + _EPSILON)
            v_fac = v_pos / (n_v_pos + _EPSILON)
            scale = pos_term
        else:
            u_fac = u_neg / (n_u_neg + _EPSILON)
            v_fac = v_neg / (n_v_neg + _EPSILON)
            scale = neg_term

        W[:, j] = cp.sqrt(s * scale) * u_fac
        H[j, :] = cp.sqrt(s * scale) * v_fac

    # Handle zeros based on init variant
    if init == "nndsvd":
        # Keep zeros (good for sparse data)
        cp.maximum(W, 0, out=W)
        cp.maximum(H, 0, out=H)
    elif init == "nndsvda":
        # Fill zeros with the average
        avg = _safe_sparse_abs_mean(X)
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        # Fill zeros with small random values
        cp.random.seed(random_state)
        avg = _safe_sparse_abs_mean(X)
        W[W == 0] = (
            avg * cp.random.uniform(size=int((W == 0).sum())).astype(X.dtype) * 0.01
        )
        H[H == 0] = (
            avg * cp.random.uniform(size=int((H == 0).sum())).astype(X.dtype) * 0.01
        )
    else:
        msg = f"Invalid init: {init!r}"
        raise ValueError(msg)

    return W, H


def _safe_sparse_abs_mean(X) -> float:
    """Compute mean of absolute values, handling sparse matrices."""
    if cpissparse(X):
        return cp.abs(X.data).mean()
    return cp.abs(X).mean()


# ---------------------------------------------------------------------------
# HALS solver (Hierarchical ALS with fused CUDA kernels)
# ---------------------------------------------------------------------------


def _fit_hals(
    X,
    W: cp.ndarray,
    H: cp.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio: float = 0.0,
    n_inner: int = _DEFAULT_N_INNER,
) -> tuple[cp.ndarray, cp.ndarray, int, float]:
    """NMF via HALS with fused CUDA kernels.

    Uses coordinate descent updates with ``n_inner`` inner sweeps per
    sparse matrix multiply.  Each sweep is fused into a single CUDA
    kernel launch with the K x K gram matrix in shared memory.

    Each component *k* is updated by the closed-form rule::

        H[k] = max(0, H[k] + (WtX[k] - WtW[k] @ H - reg) / (WtW[k,k] + l2))

    where ``WtX = W^T X`` and ``WtW = W^T W`` are computed once per outer
    iteration (one sparse matmul each), and the *K* component updates are
    fused into a single CUDA kernel with the K × K gram matrix in shared
    memory.
    """
    n, m = X.shape
    k = W.shape[1]
    dtype = W.dtype

    l1_W = dtype.type(alpha_W * l1_ratio)
    l2_W = dtype.type(alpha_W * (1 - l1_ratio))
    l1_H = dtype.type(alpha_H * l1_ratio)
    l2_H = dtype.type(alpha_H * (1 - l1_ratio))

    W = cp.ascontiguousarray(W)
    H = cp.ascontiguousarray(H)

    # Check shared memory requirement (K × K × dtype_size)
    shared = k * k * dtype.itemsize
    device = cp.cuda.Device()
    max_shared = device.attributes["MaxSharedMemoryPerBlock"]
    if shared > max_shared:
        raise ValueError(
            f"n_components={k} requires {shared} bytes of shared memory, "
            f"but device supports at most {max_shared}. Use solver='mu'."
        )

    nmf_cuda = _get_nmf_cuda()
    stream = cp.cuda.get_current_stream().ptr

    # Precompute ||X||_F^2 (constant across iterations)
    if cpissparse(X):
        X_norm_sq = float(cp.sum(X.data**2))
    else:
        X_norm_sq = float(cp.sum(X**2))

    prev_error = float("inf")
    n_iter_done = max_iter

    for n_iter in range(1, max_iter + 1):
        # --- H update: sparse matmuls + fused kernel sweeps ---
        WtX = cp.ascontiguousarray(_safe_sparse_dot_t(W, X))
        WtW = cp.ascontiguousarray(W.T @ W)

        # Convergence check — reuses WtX and WtW, no redundant sparse matmul
        if tol > 0 and n_iter % 10 == 0:
            HHt = H @ H.T
            error = X_norm_sq - 2.0 * float(cp.sum(H * WtX)) + float(cp.sum(WtW * HHt))
            if prev_error > 0:
                relative_change = abs(prev_error - error) / prev_error
                if relative_change < tol:
                    n_iter_done = n_iter
                    break
            prev_error = error

        nmf_cuda.hals_update_H(
            H,
            WtX,
            WtW,
            m=m,
            K=k,
            l1_reg=l1_H,
            l2_reg=l2_H,
            n_sweeps=n_inner,
            stream=stream,
        )

        # --- W update: sparse matmuls + fused kernel sweeps ---
        XHt = cp.ascontiguousarray(_safe_sparse_dot(X, H.T))
        HHt = cp.ascontiguousarray(H @ H.T)

        nmf_cuda.hals_update_W(
            W,
            XHt,
            HHt,
            n=n,
            K=k,
            l1_reg=l1_W,
            l2_reg=l2_W,
            n_sweeps=n_inner,
            stream=stream,
        )

    error = _frobenius_norm_squared(X_norm_sq, W, H, X)
    return W, H, n_iter_done, cp.sqrt(max(error, 0))


# ---------------------------------------------------------------------------
# Multiplicative update solver (Frobenius norm, no full WH materialization)
# ---------------------------------------------------------------------------


def _fit_multiplicative_update(
    X,
    W: cp.ndarray,
    H: cp.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio: float = 0.0,
) -> tuple[cp.ndarray, cp.ndarray, int, float]:
    """NMF via multiplicative updates with Frobenius norm.

    Update rules (Lee & Seung 2001)::

        H ← H * (W^T X) / (W^T W H + reg_H)
        W ← W * (X H^T) / (W H H^T + reg_W)

    Frobenius norm without full reconstruction::

        ||X - WH||_F^2 = ||X||_F^2 - 2 tr(H (X^T W)) + tr((W^T W)(H H^T))
    """
    # Regularization decomposition
    l1_W = alpha_W * l1_ratio
    l2_W = alpha_W * (1 - l1_ratio)
    l1_H = alpha_H * l1_ratio
    l2_H = alpha_H * (1 - l1_ratio)

    # Precompute ||X||_F^2 (constant across iterations)
    if cpissparse(X):
        X_norm_sq = float(cp.sum(X.data**2))
    else:
        X_norm_sq = float(cp.sum(X**2))

    prev_error = _frobenius_norm_squared(X_norm_sq, W, H, X)

    for n_iter_done in range(1, max_iter + 1):
        # --- Update H ---
        WtX = _safe_sparse_dot_t(W, X)
        WtW = W.T @ W
        denom_H = WtW @ H
        if l1_H > 0:
            denom_H += l1_H
        if l2_H > 0:
            denom_H += l2_H * H
        cp.maximum(denom_H, _EPSILON, out=denom_H)
        H *= WtX / denom_H

        # --- Update W ---
        XHt = _safe_sparse_dot(X, H.T)
        HHt = H @ H.T
        denom_W = W @ HHt
        if l1_W > 0:
            denom_W += l1_W
        if l2_W > 0:
            denom_W += l2_W * W
        cp.maximum(denom_W, _EPSILON, out=denom_W)
        W *= XHt / denom_W

        # --- Check convergence ---
        if tol > 0 and n_iter_done % 10 == 0:
            error = _frobenius_norm_squared(X_norm_sq, W, H, X)
            if prev_error > 0:
                relative_change = abs(prev_error - error) / prev_error
                if relative_change < tol:
                    break
            prev_error = error

    error = _frobenius_norm_squared(X_norm_sq, W, H, X)
    return W, H, n_iter_done, cp.sqrt(max(error, 0))


# ---------------------------------------------------------------------------
# Frobenius norm without full matrix materialization
# ---------------------------------------------------------------------------


def _frobenius_norm_squared(
    X_norm_sq: float,
    W: cp.ndarray,
    H: cp.ndarray,
    X,
) -> float:
    """Compute ||X - WH||_F^2 without materializing WH.

    Uses the identity::

        ||X - WH||_F^2 = ||X||_F^2 - 2*tr(H @ X^T @ W) + tr((W^T W)(H H^T))

    All intermediates are at most (K × K) or (K × n_features).
    """
    WtX = _safe_sparse_dot_t(W, X)
    cross_term = float(cp.sum(H * WtX))

    WtW = W.T @ W
    HHt = H @ H.T
    approx_term = float(cp.sum(WtW * HHt))

    return X_norm_sq - 2.0 * cross_term + approx_term


# ---------------------------------------------------------------------------
# Sparse-safe matrix products
# ---------------------------------------------------------------------------


def _safe_sparse_dot(X, Y) -> cp.ndarray:
    """Compute X @ Y handling sparse X."""
    if cpissparse(X):
        return X.dot(Y)
    return X @ Y


def _safe_sparse_dot_t(W: cp.ndarray, X) -> cp.ndarray:
    """Compute W.T @ X handling sparse X.

    Returns (K × n_features) without transposing X.
    """
    if cpissparse(X):
        return X.T.dot(W).T
    return W.T @ X
