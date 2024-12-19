from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cuml import KMeans as CumlKMeans

from ._fuses import _calc_R, _get_factor, _get_pen, _log_div_OE, _R_multi_m
from ._kernels._normalize import _get_normalize_kernel_optimized

if TYPE_CHECKING:
    import pandas as pd


def _normalize_cp_p1(X: cp.ndarray) -> cp.ndarray:
    """
    Normalize rows of a matrix using an optimized kernel with shared memory and warp shuffle.

    Parameters
    ----------
    X
        Input 2D array.

    Returns
    -------
    Row-normalized 2D array.
    """
    assert X.ndim == 2, "Input must be a 2D array."

    rows, cols = X.shape

    # Fixed block size of 32
    block_dim = 32
    grid_dim = rows  # One block per row

    normalize_p1 = _get_normalize_kernel_optimized(X.dtype)
    # Launch the kernel
    normalize_p1((grid_dim,), (block_dim,), (X, rows, cols))
    return X


def _normalize_cp(X: cp.ndarray, p: int = 2) -> cp.ndarray:
    """
    Analogous to `torch.nn.functional.normalize` for `axis = 1`, `p` in numpy is known as `ord`.
    """
    if p == 2:
        return X / cp.linalg.norm(X, ord=2, axis=1, keepdims=True).clip(min=1e-12)

    else:
        return _normalize_cp_p1(X)


def _get_batch_codes(batch_mat: pd.DataFrame, batch_key: str | list[str]) -> pd.Series:
    if type(batch_key) is str:
        batch_vec = batch_mat[batch_key]

    elif len(batch_key) == 1:
        batch_key = batch_key[0]

        batch_vec = batch_mat[batch_key]

    else:
        df = batch_mat[batch_key].astype("str")
        batch_vec = df.apply(lambda row: ",".join(row), axis=1)

    return batch_vec.astype("category")


def _one_hot_tensor_cp(X: pd.Series) -> cp.array:
    """
    One-hot encode a categorical series.

    Parameters
    ----------
    X
        Input categorical series.
    Returns
    -------
    One-hot encoded array.
    """
    ids = cp.array(X.cat.codes.values.copy(), dtype=cp.int32).reshape(-1)
    n_col = X.cat.categories.size
    Phi = cp.eye(n_col)[ids]

    return Phi


def harmonize(
    Z: cp.array,
    batch_mat: pd.DataFrame,
    batch_key: str | list[str],
    *,
    n_clusters: int = None,
    max_iter_harmony: int = 10,
    max_iter_clustering: int = 200,
    tol_harmony: float = 1e-4,
    tol_clustering: float = 1e-5,
    ridge_lambda: float = 1.0,
    sigma: float = 0.1,
    block_proportion: float = 0.05,
    theta: float = 2.0,
    tau: int = 0,
    correction_method: str = "fast",
    random_state: int = 0,
) -> cp.array:
    """
    Integrate data using Harmony algorithm.

    Parameters
    ----------

    X
        The input embedding with rows for cells (N) and columns for embedding coordinates (d).

    batch_mat
        The cell barcode information as data frame, with rows for cells (N) and columns for cell attributes.

    batch_key
        Cell attribute(s) from ``batch_mat`` to identify batches.

    n_clusters
        Number of clusters used in Harmony algorithm. If ``None``, choose the minimum of 100 and N / 30.

    max_iter_harmony
        Maximum iterations on running Harmony if not converged.

    max_iter_clustering
        Within each Harmony iteration, maximum iterations on the clustering step if not converged.

    tol_harmony
        Tolerance on justifying convergence of Harmony over objective function values.

    tol_clustering
        Tolerance on justifying convergence of the clustering step over objective function values within each Harmony iteration.

    ridge_lambda
        Hyperparameter of ridge regression on the correction step.

    sigma
        Weight of the entropy term in objective function.

    block_proportion
        Proportion of block size in one update operation of clustering step.

    theta
        Weight of the diversity penalty term in objective function.

    tau
        Discounting factor on ``theta``. By default, there is no discounting.

    correction_method
        Choose which method for the correction step: ``original`` for original method, ``fast`` for improved method. By default, use improved method.

    random_state
        Random seed for reproducing results.


    Returns
    -------
    The integrated embedding by Harmony, of the same shape as the input embedding.

    Examples
    --------
    >>> adata = anndata.read_h5ad("filename.h5ad")
    >>> X_harmony = harmonize(adata.obsm['X_pca'], adata.obs, 'Channel')

    >>> adata = anndata.read_h5ad("filename.h5ad")
    >>> X_harmony = harmonize(adata.obsm['X_pca'], adata.obs, ['Channel', 'Lab'])
    """

    Z_norm = _normalize_cp(Z)
    n_cells = Z.shape[0]

    batch_codes = _get_batch_codes(batch_mat, batch_key)
    n_batches = batch_codes.cat.categories.size
    N_b = cp.array(batch_codes.value_counts(sort=False).values, dtype=Z.dtype)
    Pr_b = (N_b.reshape(-1, 1) / len(batch_codes)).astype(Z.dtype)

    Phi = _one_hot_tensor_cp(batch_codes).astype(Z.dtype)
    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))
    theta = (cp.ones(n_batches) * theta).astype(Z.dtype)

    if tau > 0:
        theta = theta * (1 - cp.exp(-N_b / (n_clusters * tau)) ** 2)

    theta = theta.reshape(1, -1)
    assert block_proportion > 0 and block_proportion <= 1
    if correction_method not in {"fast", "original"}:
        raise ValueError("correction_method must be either 'fast' or 'original'.")

    cp.random.seed(random_state)

    R, E, O, objectives_harmony = _initialize_centroids(
        Z_norm,
        n_clusters=n_clusters,
        sigma=sigma,
        Pr_b=Pr_b,
        Phi=Phi,
        theta=theta,
        random_state=random_state,
    )
    is_converged = False
    for _ in range(max_iter_harmony):
        _clustering(
            Z_norm,
            Pr_b=Pr_b,
            Phi=Phi,
            R=R,
            E=E,
            O=O,
            theta=theta,
            tol=tol_clustering,
            objectives_harmony=objectives_harmony,
            max_iter=max_iter_clustering,
            sigma=sigma,
            block_proportion=block_proportion,
        )

        Z_hat = _correction(
            Z,
            R=R,
            Phi=Phi,
            O=O,
            ridge_lambda=ridge_lambda,
            correction_method=correction_method,
        )
        Z_norm = _normalize_cp(Z_hat, p=2)
        if _is_convergent_harmony(objectives_harmony, tol=tol_harmony):
            is_converged = True
            break

    if not is_converged:
        warnings.warn(
            "Harmony did not converge. Consider increasing the number of iterations"
        )
    return Z_hat


def _initialize_centroids(
    Z_norm: cp.ndarray,
    *,
    n_clusters: int,
    sigma: float,
    Pr_b: cp.ndarray,
    Phi: cp.ndarray,
    theta: cp.ndarray,
    random_state: int = 0,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, list]:
    kmeans = CumlKMeans(
        n_clusters=n_clusters, init="k-means||", max_iter=25, random_state=random_state
    )
    kmeans.fit(Z_norm)
    Y = kmeans.cluster_centers_.astype(Z_norm.dtype)
    Y_norm = _normalize_cp(Y, p=2)

    # Initialize R
    R = _calc_R(-2 / sigma, cp.dot(Z_norm, Y_norm.T))
    R = _normalize_cp(R, p=1)

    E = cp.dot(Pr_b, cp.sum(R, axis=0, keepdims=True))
    O = cp.dot(Phi.T, R)

    objectives_harmony = []
    _compute_objective(
        Y_norm,
        Z_norm,
        R=R,
        theta=theta,
        sigma=sigma,
        O=O,
        E=E,
        objective_arr=objectives_harmony,
    )

    return R, E, O, objectives_harmony


def _clustering(
    Z_norm: cp.ndarray,
    *,
    Pr_b: cp.ndarray,
    Phi: cp.ndarray,
    R: cp.ndarray,
    E: cp.ndarray,
    O: cp.ndarray,
    theta: cp.ndarray,
    tol: float,
    objectives_harmony: list,
    max_iter: int,
    sigma: float,
    block_proportion: float,
) -> None:
    """
    Perform iterative clustering updates on normalized input data, adjusting
    cluster assignments and associated penalty terms until convergence or
    maximum iterations are reached.

    This function operates in-place to update the cluster assignment matrix (R)
    and penalty-related matrices (O and E).
    """
    n_cells = Z_norm.shape[0]
    objectives_clustering = []
    block_size = int(n_cells * block_proportion)
    term = -2 / sigma
    for _ in range(max_iter):
        # Compute Cluster Centroids
        Y = cp.dot(R.T, Z_norm)  # Compute centroids
        Y_norm = _normalize_cp(Y, p=2)  # Normalize centroids

        idx_list = cp.arange(n_cells)
        cp.random.shuffle(idx_list)
        pos = 0
        while pos < len(idx_list):
            idx_in = idx_list[pos : (pos + block_size)]
            R_in = R[idx_in]  # Slice rows for R
            Phi_in = Phi[idx_in]  # Slice rows for Phi
            # O-=Phi_in.T@R_in
            cp.cublas.gemm("T", "N", Phi_in, R_in, alpha=-1, beta=1, out=O)
            # E-=Pr_b@R_in
            cp.cublas.gemm(
                "N",
                "N",
                Pr_b,
                cp.sum(R_in, axis=0, keepdims=True),
                alpha=-1,
                beta=1,
                out=E,
            )

            # Update and Normalize R
            R_out = _calc_R(term, cp.dot(Z_norm[idx_in], Y_norm.T))

            # Precompute penalty term and apply
            penalty_term = _get_pen(E, O, theta.T)
            omega = cp.dot(Phi_in, penalty_term)
            R_out *= omega

            # Normalize R_out and update R
            R_out = _normalize_cp(R_out, p=1)

            R[idx_in] = R_out

            # Compute O and E with full data using precomputed terms
            # O+=Phi_in.T@R_in
            cp.cublas.gemm("T", "N", Phi_in, R_out, alpha=1, beta=1, out=O)
            # E+=Pr_b@R_in
            cp.cublas.gemm(
                "N",
                "N",
                Pr_b,
                cp.sum(R_out, axis=0, keepdims=True),
                alpha=1,
                beta=1,
                out=E,
            )
            pos += block_size
        _compute_objective(
            Y_norm,
            Z_norm,
            R=R,
            theta=theta,
            sigma=sigma,
            O=O,
            E=E,
            objective_arr=objectives_clustering,
        )

        if _is_convergent_clustering(objectives_clustering, tol):
            objectives_harmony.append(objectives_clustering[-1])
            break


def _correction(
    X: cp.ndarray,
    *,
    R: cp.ndarray,
    Phi: cp.ndarray,
    O: cp.ndarray,
    ridge_lambda: float,
    correction_method: str = "fast",
) -> cp.ndarray:
    if correction_method == "fast":
        return _correction_fast(X, R, Phi, O, ridge_lambda)
    else:
        return _correction_original(X, R, Phi, ridge_lambda)


def _correction_original(
    X: cp.ndarray, R: cp.ndarray, Phi: cp.ndarray, ridge_lambda: float
) -> cp.ndarray:
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = cp.concatenate((cp.ones((n_cells, 1), dtype=X.dtype), Phi), axis=1)

    Z = X.copy()
    id_mat = cp.eye(n_batches + 1, n_batches + 1, dtype=X.dtype)
    id_mat[0, 0] = 0
    Lambda = ridge_lambda * id_mat
    for k in range(n_clusters):
        Phi_t_diag_R = Phi_1.T * R[:, k].reshape(1, -1)
        inv_mat = cp.linalg.inv(cp.dot(Phi_t_diag_R, Phi_1) + Lambda)
        W = cp.dot(inv_mat, cp.dot(Phi_t_diag_R, X))
        W[0, :] = 0
        Z -= cp.dot(Phi_t_diag_R.T, W)

    return Z


def _correction_fast(
    X: cp.ndarray, R: cp.ndarray, Phi: cp.ndarray, O: cp.ndarray, ridge_lambda: float
) -> cp.ndarray:
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = cp.concatenate((cp.ones((n_cells, 1), dtype=X.dtype), Phi), axis=1)
    Z = X.copy()
    P = cp.eye(n_batches + 1, n_batches + 1, dtype=X.dtype)
    for k in range(n_clusters):
        O_k = O[:, k]
        N_k = cp.sum(O_k)

        factor = _get_factor(O_k, ridge_lambda)
        c = N_k + cp.sum(-factor * O_k**2)
        c_inv = 1 / c

        P[0, 1:] = -factor * O_k

        P_t_B_inv = cp.zeros((factor.size + 1, factor.size + 1), dtype=X.dtype)

        # Set diagonal entries
        P_t_B_inv[0, 0] = c_inv
        P_t_B_inv[1:, 1:] = cp.diag(factor)

        # Set off-diagonal entries
        P_t_B_inv[1:, 0] = P[0, 1:] * c_inv
        inv_mat = cp.dot(P_t_B_inv, P)

        Phi_t_diag_R = Phi_1.T * R[:, k].reshape(1, -1)
        W = cp.dot(inv_mat, cp.dot(Phi_t_diag_R, X))
        W[0, :] = 0

        Z -= cp.dot(Phi_t_diag_R.T, W)

    return Z


def _compute_objective(
    Y_norm: cp.ndarray,
    Z_norm: cp.ndarray,
    *,
    R: cp.ndarray,
    theta: cp.ndarray,
    sigma: float,
    O: cp.ndarray,
    E: cp.ndarray,
    objective_arr: list,
) -> None:
    kmeans_error = cp.sum(_R_multi_m(R, cp.dot(Z_norm, Y_norm.T)))
    R = R / R.sum(axis=1, keepdims=True)
    entropy = cp.sum(R * cp.log(R + 1e-12))
    entropy_term = sigma * entropy
    diversity_penalty = sigma * cp.sum(cp.dot(theta, _log_div_OE(O, E)))
    objective = kmeans_error + entropy_term + diversity_penalty
    objective_arr.append(objective)


def _is_convergent_harmony(objectives_harmony: list, tol: float) -> bool:
    """
    Check if the Harmony algorithm has converged based on the objective function values.
    """
    if len(objectives_harmony) < 2:
        return False

    obj_old = objectives_harmony[-2]
    obj_new = objectives_harmony[-1]

    return (obj_old - obj_new) < tol * np.abs(obj_old)


def _is_convergent_clustering(
    objectives_clustering: list, tol: list, window_size: int = 3
) -> bool:
    """
    Check if the clustering step has converged based on the objective function values
    """
    if len(objectives_clustering) < window_size + 1:
        return False
    obj_old = 0.0
    obj_new = 0.0
    for i in range(window_size):
        obj_old += objectives_clustering[-2 - i]
        obj_new += objectives_clustering[-1 - i]

    return (obj_old - obj_new) < tol * np.abs(obj_old)
