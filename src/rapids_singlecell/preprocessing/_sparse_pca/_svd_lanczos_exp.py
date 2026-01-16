"""
Experimental GPU-accelerated Sparse SVD using Lanczos Bidiagonalization.

This module contains experimental optimizations for testing.
The stable version is in _svd_lanczos.py.

Current experiments:
1. Periodic reorthogonalization (reorth every N steps instead of every step)
2. Partial reorthogonalization (only reorth against recent vectors)
3. Optimized memory layout
"""

from __future__ import annotations

import cupy as cp

# Simple fused AXPY kernel - faster than function call overhead
_kernel_axpy = cp.ElementwiseKernel(
    "T alpha, T y", "T x", "x -= alpha * y", "lanczos_axpy_exp"
)


def _cgs2_orth(
    target: cp.ndarray,
    basis_matrix: cp.ndarray,
    n_valid: int,
    coeffs_buf: cp.ndarray,
) -> cp.ndarray:
    """
    Orthogonalize 'target' against the first 'n_valid' columns of 'basis_matrix' using CGS2.
    """
    if n_valid == 0:
        return target

    basis_view = basis_matrix[:, :n_valid]
    coeffs = coeffs_buf[:n_valid]

    # --- Pass 1 ---
    cp.dot(basis_view.T, target, out=coeffs)
    target -= cp.dot(basis_view, coeffs)

    # --- Pass 2 (for stability) ---
    cp.dot(basis_view.T, target, out=coeffs)
    target -= cp.dot(basis_view, coeffs)

    return target


def _cgs1_orth(
    target: cp.ndarray,
    basis_matrix: cp.ndarray,
    n_valid: int,
    coeffs_buf: cp.ndarray,
) -> cp.ndarray:
    """
    Single-pass Classical Gram-Schmidt (CGS1).
    Faster but less stable. Good for V vectors where n << m.
    """
    if n_valid == 0:
        return target

    basis_view = basis_matrix[:, :n_valid]
    coeffs = coeffs_buf[:n_valid]

    cp.dot(basis_view.T, target, out=coeffs)
    target -= cp.dot(basis_view, coeffs)

    return target


def _cgs2_partial_orth(
    target: cp.ndarray,
    basis_matrix: cp.ndarray,
    n_valid: int,
    coeffs_buf: cp.ndarray,
    max_orth: int = 100,
) -> cp.ndarray:
    """
    Partial CGS2: only orthogonalize against the most recent max_orth vectors.

    This reduces O(n_valid) cost to O(max_orth) when n_valid > max_orth.
    Less stable but much faster for large n_valid.
    """
    if n_valid == 0:
        return target

    # Only orthogonalize against most recent vectors
    start_idx = max(0, n_valid - max_orth)
    n_orth = n_valid - start_idx

    basis_view = basis_matrix[:, start_idx:n_valid]
    coeffs = coeffs_buf[:n_orth]

    # --- Pass 1 ---
    cp.dot(basis_view.T, target, out=coeffs)
    target -= cp.dot(basis_view, coeffs)

    # --- Pass 2 (for stability) ---
    cp.dot(basis_view.T, target, out=coeffs)
    target -= cp.dot(basis_view, coeffs)

    return target


def _lanczos_bidiag_exp(
    A,
    *,
    ncv: int,
    v_start: cp.ndarray,
    U_full: cp.ndarray,
    V_full: cp.ndarray,
    n_locked: int,
    rng: cp.random.RandomState,
    ortho_buf: cp.ndarray,
    use_cgs1_for_v: bool = False,
    use_partial_reorth: bool = False,
    partial_reorth_size: int = 150,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Lanczos bidiagonalization step with experimental options.

    Parameters
    ----------
    use_cgs1_for_v : bool
        If True, use single-pass CGS for V vectors (faster but less stable).
    use_partial_reorth : bool
        If True, only orthogonalize against recent vectors (faster but less stable).
    partial_reorth_size : int
        Number of recent vectors to orthogonalize against when use_partial_reorth=True.
    u_work, v_work : cp.ndarray, optional
        Pre-allocated work arrays to avoid allocation in hot loop.
    """
    m, n = A.shape
    dtype = A.dtype

    # Pre-allocate GPU arrays for bidiagonal entries
    alphas = cp.zeros(ncv, dtype=dtype)
    betas = cp.zeros(ncv + 1, dtype=dtype)

    # Choose orthogonalization for V vectors
    v_orth_fn = _cgs1_orth if use_cgs1_for_v else _cgs2_orth

    # Initialize v_start
    v = v_start

    # Ortho against locked V vectors
    v = _cgs2_orth(v, V_full, n_locked, ortho_buf)

    beta_val = cp.linalg.norm(v)
    if beta_val < 1e-9:
        v = rng.standard_normal(n).astype(dtype)
        v = _cgs2_orth(v, V_full, n_locked, ortho_buf)
        beta_val = cp.linalg.norm(v)

    v /= beta_val
    V_full[:, n_locked] = v

    # Choose U orthogonalization function
    if use_partial_reorth:

        def u_orth_fn(u, U, n_valid, buf):
            return _cgs2_partial_orth(u, U, n_valid, buf, partial_reorth_size)
    else:
        u_orth_fn = _cgs2_orth

    # --- LANCZOS LOOP ---
    for i in range(ncv):
        idx_u = n_locked + i
        idx_v = n_locked + i

        # u = A @ v
        u = A.dot(V_full[:, idx_v])

        # Deflation: u = u - beta * u_prev
        if i > 0:
            _kernel_axpy(betas[i], U_full[:, idx_u - 1], u)

        # Ortho U against valid U vectors
        u = u_orth_fn(u, U_full, idx_u, ortho_buf)

        alpha_val = cp.linalg.norm(u)
        if alpha_val < 1e-9:
            u = rng.standard_normal(m).astype(dtype)
            u = _cgs2_orth(u, U_full, idx_u, ortho_buf)  # Full reorth for restart
            alpha_val = cp.linalg.norm(u)
            alphas[i] = 0.0
        else:
            alphas[i] = alpha_val
        u /= alpha_val
        U_full[:, idx_u] = u

        # v = A.T @ u
        if i < ncv:
            v = A.T.dot(u)
            _kernel_axpy(alphas[i], V_full[:, idx_v], v)

            # Ortho V against all valid V (optionally use CGS1)
            v = v_orth_fn(v, V_full, idx_v + 1, ortho_buf)

            beta_val = cp.linalg.norm(v)
            if beta_val < 1e-9:
                v = rng.standard_normal(n).astype(dtype)
                v = _cgs2_orth(v, V_full, idx_v + 1, ortho_buf)
                beta_val = cp.linalg.norm(v)
                betas[i + 1] = 0.0
            else:
                betas[i + 1] = beta_val
            v /= beta_val
            V_full[:, idx_v + 1] = v

    return alphas, betas


def gpu_sparse_svds_exp(
    A,
    k: int = 6,
    *,
    ncv: int | None = None,
    tol: float = 1e-4,
    max_iter: int = 100,
    random_state: int | None = None,
    refine_results: bool = True,
    use_cgs1_for_v: bool = False,
    use_partial_reorth: bool = False,
    partial_reorth_size: int = 150,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Experimental GPU-accelerated sparse SVD.

    Parameters
    ----------
    A
        Sparse matrix (m x n) to decompose.
    k
        Number of singular values and vectors to compute.
    ncv
        Number of Lanczos vectors generated.
    tol
        Tolerance for convergence.
    max_iter
        Maximum number of restart iterations.
    random_state
        Seed for the random number generator.
    refine_results
        If True, refine the singular vectors after convergence.
    use_cgs1_for_v
        If True, use single-pass CGS for V vectors (experimental speedup).
    use_partial_reorth
        If True, only orthogonalize against recent vectors (faster but less stable).
    partial_reorth_size
        Number of recent vectors to orthogonalize against.

    Returns
    -------
    U, S, Vt
    """
    m, n = A.shape
    dtype = A.dtype

    if ncv is None:
        if m > 100000:
            if k < 75:
                ncv = min(int(2.5 * k), min(m, n))
            else:
                ncv = min(int(4 * k), min(m, n))
        else:
            ncv = min(max(3 * k, 50), min(m, n))

    ncv = max(ncv, k + 10)

    rng = cp.random.RandomState(random_state if random_state is not None else 0)

    # Pre-allocate buffers
    total_capacity = k + ncv + 2
    U_full = cp.zeros((m, total_capacity), dtype=dtype, order="F")
    V_full = cp.zeros((n, total_capacity), dtype=dtype, order="F")
    locked_S = cp.zeros(k, dtype=dtype)
    ortho_buf = cp.zeros(total_capacity, dtype=dtype)

    n_locked = 0
    total_iter = 0
    v_start = rng.standard_normal(n).astype(dtype)

    while n_locked < k and total_iter < max_iter:
        alphas, betas = _lanczos_bidiag_exp(
            A,
            ncv=ncv,
            v_start=v_start,
            U_full=U_full,
            V_full=V_full,
            n_locked=n_locked,
            rng=rng,
            ortho_buf=ortho_buf,
            use_cgs1_for_v=use_cgs1_for_v,
            use_partial_reorth=use_partial_reorth,
            partial_reorth_size=partial_reorth_size,
        )

        # SVD of Bidiagonal Matrix
        B = cp.zeros((ncv, ncv), dtype=dtype)
        cp.fill_diagonal(B, alphas)
        cp.fill_diagonal(B[:-1, 1:], betas[1:ncv])

        p, s, qt = cp.linalg.svd(B)

        # Sort descending
        idx = cp.argsort(s)[::-1]
        s = s[idx]
        p = p[:, idx]
        qt = qt[idx, :]

        # Error estimates
        resid = betas[ncv] * cp.abs(p[ncv - 1, :])
        max_s = s[0] if s.shape[0] > 0 else 1.0

        converged_indices = []
        for i in range(min(k - n_locked + 2, ncv)):
            if resid[i] < tol * max_s:
                converged_indices.append(i)

        if converged_indices:
            num_found = len(converged_indices)
            if n_locked + num_found > k:
                num_found = k - n_locked
                converged_indices = converged_indices[:num_found]

            good_idx = cp.array(converged_indices)

            # Compute Ritz vectors
            u_ritz = cp.dot(U_full[:, n_locked : n_locked + ncv], p[:, good_idx])
            v_ritz = cp.dot(qt[good_idx, :], V_full[:, n_locked : n_locked + ncv].T).T

            # Store locked vectors
            start = n_locked
            end = n_locked + num_found
            U_full[:, start:end] = u_ritz
            V_full[:, start:end] = v_ritz
            locked_S[start:end] = s[good_idx]

            n_locked += num_found

            if n_locked >= k:
                break

            # Restart from best non-converged
            best_nc = 0
            for i in range(ncv):
                if i not in converged_indices:
                    best_nc = i
                    break

            v_start = cp.dot(V_full[:, n_locked : n_locked + ncv], qt[best_nc, :].T)
        else:
            # Thick restart
            k_mix = min(k - n_locked, ncv)
            weights = cp.ones(k_mix, dtype=dtype)
            combo_small = cp.dot(qt[:k_mix, :].T, weights)
            v_start = cp.dot(V_full[:, n_locked : n_locked + ncv], combo_small)

        v_start /= cp.linalg.norm(v_start)
        total_iter += 1

    # Final extraction
    final_U = U_full[:, :k].copy()
    final_Vt = V_full[:, :k].T.copy()
    final_S = locked_S.copy()

    # Refinement
    if refine_results:
        final_U = A.dot(final_Vt.T)
        final_S = cp.linalg.norm(final_U, axis=0)
        final_U /= final_S[None, :]

    # Sort by singular value
    sort_idx = cp.argsort(final_S)[::-1]
    final_S = final_S[sort_idx]
    final_U = final_U[:, sort_idx]
    final_Vt = final_Vt[sort_idx, :]

    # Deterministic signs
    max_abs_cols = cp.argmax(cp.abs(final_U), axis=0)
    signs = cp.sign(final_U[max_abs_cols, cp.arange(final_U.shape[1])])
    signs[signs == 0] = 1
    final_U *= signs
    final_Vt *= signs[:, None]

    return final_U, final_S, final_Vt
