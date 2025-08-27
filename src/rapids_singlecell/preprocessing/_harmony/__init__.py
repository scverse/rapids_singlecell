from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
from cuml import KMeans as CumlKMeans

from ._fuses import (
    _calc_R,
    _entropy_kernel,
    _get_factor,
    _get_pen,
    _log_div_OE,
)
from ._helper import (
    _choose_colsum_algo_benchmark,
    _choose_colsum_algo_heuristic,
    _create_category_index_mapping,
    _get_aggregated_matrix,
    _get_batch_codes,
    _get_theta_array,
    _kmeans_error,
    _normalize_cp,
    _one_hot_tensor_cp,
    _outer_cp,
    _penalty_term,
    _scatter_add_cp,
    _scatter_add_cp_bias_csr,
    _Z_correction,
)

if TYPE_CHECKING:
    import pandas as pd

COLSUM_ALGO = Literal["columns", "atomics", "gemm", "cupy", "benchmark"]


def harmonize(
    Z: cp.array,
    batch_mat: pd.DataFrame,
    batch_key: str | list[str],
    *,
    n_clusters: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_clustering: int = 200,
    tol_harmony: float = 1e-4,
    tol_clustering: float = 1e-5,
    ridge_lambda: float = 1.0,
    sigma: float = 0.1,
    block_proportion: float = 0.05,
    theta: float | int | list[float] | np.ndarray | cp.ndarray = 2.0,
    tau: int = 0,
    correction_method: str = "fast",
    use_gemm: bool = False,
    colsum_algo: COLSUM_ALGO | None = None,
    random_state: int = 0,
    verbose: bool = False,
) -> cp.array:
    """
    Integrate data using Harmony algorithm.

    Parameters
    ----------
    Z
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

    use_gemm
        If True, use a One-Hot-Encoding Matrix and GEMM to compute Harmony. If False use a label vector. A label vector is more memory efficient and faster for large datasets with a large number of batches.

    colsum_algo
        Choose which algorithm to use for column sum. If `None`, choose the algorithm based on the number of rows and columns. If `'benchmark'`, benchmark all algorithms and choose the best one.

    random_state
        Random seed for reproducing results.

    verbose
        Whether to print benchmarking results for the column sum algorithm and the number of iterations until convergence.

    Returns
    -------
    The integrated embedding by Harmony, of the same shape as the input embedding.
    """
    Z_norm = _normalize_cp(Z)
    n_cells = Z.shape[0]

    # Process batch information
    batch_codes = _get_batch_codes(batch_mat, batch_key)
    n_batches = batch_codes.cat.categories.size
    N_b = cp.array(batch_codes.value_counts(sort=False).values, dtype=Z.dtype)
    Pr_b = (N_b.reshape(-1, 1) / len(batch_codes)).astype(Z.dtype)

    # Configure matrix representation based on use_gemm flag
    if use_gemm:
        Phi = _one_hot_tensor_cp(batch_codes).astype(Z.dtype)
        cats = None
        cat_offsets = None
        cell_indices = None
    else:
        Phi = None
        cats = cp.array(batch_codes.cat.codes.values, dtype=cp.int32)
        cat_offsets, cell_indices = _create_category_index_mapping(cats, n_batches)

    # Set up parameters
    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))

    # TODO: Allow for multiple colsum algorithms in a list
    assert colsum_algo in ["columns", "atomics", "gemm", "cupy", "benchmark", None]
    colsum_func_big = _choose_colsum_algo_heuristic(n_cells, n_clusters, None)
    if colsum_algo == "benchmark":
        colsum_func_small = _choose_colsum_algo_benchmark(
            int(n_cells * block_proportion), n_clusters, Z.dtype, verbose=verbose
        )
    else:
        colsum_func_small = _choose_colsum_algo_heuristic(
            int(n_cells * block_proportion), n_clusters, colsum_algo
        )
    theta_array = _get_theta_array(theta, n_batches, Z.dtype)
    if tau > 0:
        theta_array = theta_array * (1 - cp.exp(-N_b / (n_clusters * tau)) ** 2)
    theta_array = theta_array.reshape(1, -1)

    # Validate parameters
    assert block_proportion > 0 and block_proportion <= 1
    if correction_method not in {"fast", "original"}:
        raise ValueError("correction_method must be either 'fast' or 'original'.")

    # Set random seed
    cp.random.seed(random_state)

    # Initialize algorithm
    R, E, O, objectives_harmony = _initialize_centroids(
        Z_norm,
        n_clusters=n_clusters,
        sigma=sigma,
        Pr_b=Pr_b,
        Phi=Phi,
        theta=theta_array,
        random_state=random_state,
        cats=cats,
        n_batches=n_batches,
        colsum_func=colsum_func_big,
    )

    # Main harmony iterations
    is_converged = False

    for i in range(max_iter_harmony):
        # Clustering step
        _clustering(
            Z_norm,
            Pr_b=Pr_b,
            Phi=Phi,
            cats=cats,
            R=R,
            E=E,
            O=O,
            theta=theta_array,
            tol=tol_clustering,
            objectives_harmony=objectives_harmony,
            max_iter=max_iter_clustering,
            sigma=sigma,
            block_proportion=block_proportion,
            colsum_func=colsum_func_small,
        )
        # Correction step
        Z_hat = _correction(
            Z,
            R=R,
            Phi=Phi,
            O=O,
            ridge_lambda=ridge_lambda,
            correction_method=correction_method,
            cats=cats,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
        )
        # Normalize corrected data
        Z_norm = _normalize_cp(Z_hat, p=2)
        # Check for convergence
        if _is_convergent_harmony(objectives_harmony, tol=tol_harmony):
            is_converged = True
            if verbose:
                print(f"Harmony converged in {i + 1} iterations")
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
    Phi: cp.ndarray | None,
    theta: cp.ndarray,
    random_state: int = 0,
    cats: cp.ndarray | None = None,
    n_batches: int = None,
    colsum_func: callable = None,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, list]:
    """
    Initialize cluster centroids and related matrices for Harmony algorithm.

    Returns:
        R: Cluster assignment matrix
        E: Expected cluster assignment by batch
        O: Observed cluster assignment by batch
        objectives_harmony: List to store objective function values
    """
    # Run k-means to get initial cluster centers
    kmeans = CumlKMeans(
        n_clusters=n_clusters,
        init="k-means||",
        n_init=1,
        max_iter=25,
        random_state=random_state,
    )
    kmeans.fit(Z_norm)
    Y = kmeans.cluster_centers_.astype(Z_norm.dtype)
    Y_norm = _normalize_cp(Y, p=2)

    # Initialize cluster assignment matrix R
    term = cp.float64(-2 / sigma).astype(Z_norm.dtype)
    R = _calc_R(term, cp.dot(Z_norm, Y_norm.T))
    R = _normalize_cp(R, p=1)

    # Initialize E (expected) and O (observed) matrices
    R_sum = colsum_func(R)
    E = cp.zeros((n_batches, R.shape[1]), dtype=Z_norm.dtype)
    _outer_cp(E, Pr_b, R_sum, 1)

    if Phi is None:
        O = cp.zeros((n_batches, R.shape[1]), dtype=Z_norm.dtype)
        _scatter_add_cp(R, O, cats, 1)
    else:
        O = cp.dot(Phi.T, R)

    # Initialize objectives list
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
    Phi: cp.ndarray | None,
    cats: cp.ndarray | None,
    R: cp.ndarray,
    E: cp.ndarray,
    O: cp.ndarray,
    theta: cp.ndarray,
    tol: float,
    objectives_harmony: list,
    max_iter: int,
    sigma: float,
    block_proportion: float,
    colsum_func: callable = None,
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
    term = cp.float64(-2 / sigma).astype(Z_norm.dtype)

    for _ in range(max_iter):
        # Compute cluster centroids
        Y = cp.cublas.gemm("T", "N", R, Z_norm)
        Y_norm = _normalize_cp(Y, p=2)

        # Randomly shuffle cell indices for block updates
        idx_list = cp.arange(n_cells)
        cp.random.shuffle(idx_list)

        pos = 0
        while pos < len(idx_list):
            # Get current block of cells
            idx_in = idx_list[pos : (pos + block_size)]
            R_in = R[idx_in]

            R_in_sum = colsum_func(R_in)

            # Remove contribution of current block from O
            if Phi is None:
                cats_in = cats[idx_in]
                _scatter_add_cp(R_in, O, cats_in, 0)  # Subtract from O
            else:
                Phi_in = Phi[idx_in]
                cp.cublas.gemm("T", "N", Phi_in, R_in, alpha=-1, beta=1, out=O)

            # Use optimized column sum function
            _outer_cp(E, Pr_b, R_in_sum, 0)

            # Update cluster assignments for current block
            R_out = _calc_R(term, cp.dot(Z_norm[idx_in], Y_norm.T))
            # Apply penalty term to cluster assignments
            penalty_term = _get_pen(E, O, theta.T)
            if Phi is None:
                # R_out *= penalty_term[cats_in]
                R_out = _penalty_term(R_out, penalty_term, cats_in)
            else:
                omega = cp.dot(Phi_in, penalty_term)
                R_out *= omega

            # Normalize updated cluster assignments
            R_out = _normalize_cp(R_out, p=1)
            R[idx_in] = R_out
            # Use optimized column sum function again
            R_out_sum = colsum_func(R_out)

            # Add contribution of updated block back to O
            if Phi is None:
                _scatter_add_cp(R_out, O, cats_in, 1)  # Add to O
            else:
                cp.cublas.gemm("T", "N", Phi_in, R_out, alpha=1, beta=1, out=O)

            # Add contribution of updated block back to E
            _outer_cp(E, Pr_b, R_out_sum, 1)

            # Move to next block
            pos += block_size

        # Compute objective function for current iteration
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

        # Check for convergence
        if _is_convergent_clustering(objectives_clustering, tol):
            objectives_harmony.append(objectives_clustering[-1])
            break


def _correction(
    X: cp.ndarray,
    *,
    R: cp.ndarray,
    Phi: cp.ndarray | None,
    O: cp.ndarray,
    ridge_lambda: float,
    correction_method: str = "fast",
    cats: cp.ndarray | None = None,
    n_batches: int = None,
    cat_offsets: cp.ndarray | None = None,
    cell_indices: cp.ndarray | None = None,
) -> cp.ndarray:
    """
    Apply correction to the embedding based on the specified method.

    Args:
        X: Input embedding
        R: Cluster assignment matrix
        Phi: One-hot encoded batch matrix (if use_gemm=True)
        O: Observed cluster assignment by batch
        ridge_lambda: Ridge regression parameter
        correction_method: Method for correction ("fast" or "original")
        cats: Batch category codes (if use_gemm=False)
        n_batches: Number of batches
        cat_offsets, cell_indices: Category mapping for sparse implementation

    Returns:
        Corrected embedding
    """
    if correction_method == "fast":
        return _correction_fast(
            X,
            R,
            Phi=Phi,
            O=O,
            ridge_lambda=ridge_lambda,
            cats=cats,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
        )
    else:
        return _correction_original(
            X,
            R,
            Phi=Phi,
            ridge_lambda=ridge_lambda,
            cats=cats,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
        )


def _correction_original(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    Phi: cp.ndarray | None,
    ridge_lambda: float,
    cats: cp.ndarray | None = None,
    n_batches: int = None,
    cat_offsets: cp.ndarray | None = None,
    cell_indices: cp.ndarray | None = None,
) -> cp.ndarray:
    """
    Apply the original correction method from the Harmony paper.
    """
    n_cells = X.shape[0]
    n_clusters = R.shape[1]

    if Phi is not None:
        Phi_1 = cp.concatenate((cp.ones((n_cells, 1), dtype=X.dtype), Phi), axis=1)
        n_batches = Phi.shape[1]

    Z = X.copy()
    id_mat = cp.eye(n_batches + 1, n_batches + 1, dtype=X.dtype)
    id_mat[0, 0] = 0
    Lambda = ridge_lambda * id_mat
    for k in range(n_clusters):
        if Phi is not None:
            Phi_t_diag_R = Phi_1.T * R[:, k].reshape(1, -1)
            inv_mat = cp.linalg.inv(cp.dot(Phi_t_diag_R, Phi_1) + Lambda)
            Phi_t_diag_R = Phi_1.T * R[:, k].reshape(1, -1)
            Phi_t_diag_R_X = cp.dot(Phi_t_diag_R, X)
        else:
            R_col = R[:, k].copy()
            scatter_sum = cp.zeros(n_batches, dtype=R.dtype)
            cp.add.at(scatter_sum, cats, R_col)
            aggregated_matrix = cp.zeros((n_batches + 1, n_batches + 1), dtype=X.dtype)
            _get_aggregated_matrix(aggregated_matrix, scatter_sum, n_batches=n_batches)
            inv_mat = cp.linalg.inv(aggregated_matrix + Lambda)
            Phi_t_diag_R_X = cp.zeros((n_batches + 1, X.shape[1]), dtype=X.dtype)
            _scatter_add_cp_bias_csr(
                X,
                Phi_t_diag_R_X,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                bias=R_col,
                n_batches=n_batches,
            )
        W = cp.dot(inv_mat, Phi_t_diag_R_X)
        W[0, :] = 0
        if Phi is not None:
            cp.cublas.gemm("T", "N", Phi_t_diag_R, W, alpha=-1, beta=1, out=Z)
        else:
            _Z_correction(Z, W, cats, R_col)
    return Z


def _correction_fast(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    Phi: cp.ndarray | None,
    O: cp.ndarray,
    ridge_lambda: float,
    cats: cp.ndarray | None = None,
    n_batches: int = None,
    cat_offsets: cp.ndarray | None = None,
    cell_indices: cp.ndarray | None = None,
) -> cp.ndarray:
    """
    Apply the fast correction method (an optimization over the original method).
    """
    n_cells = X.shape[0]
    n_clusters = R.shape[1]

    if Phi is not None:
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
        if Phi is not None:
            Phi_t_diag_R = Phi_1.T * R[:, k].reshape(1, -1)
            Phi_t_diag_R_X = cp.dot(Phi_t_diag_R, X)
        else:
            R_col = R[:, k].copy()
            Phi_t_diag_R_X = cp.zeros((n_batches + 1, X.shape[1]), dtype=X.dtype)
            _scatter_add_cp_bias_csr(
                X,
                Phi_t_diag_R_X,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                bias=R_col,
                n_batches=n_batches,
            )

        W = cp.dot(inv_mat, Phi_t_diag_R_X)
        W[0, :] = 0

        if Phi is not None:
            cp.cublas.gemm("T", "N", Phi_t_diag_R, W, alpha=-1, beta=1, out=Z)
        else:
            _Z_correction(Z, W, cats, R_col)
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
    """
    Compute the objective function value for Harmony.

    The objective function consists of:
    1. K-means error term
    2. Entropy regularization term
    3. Diversity penalty term
    """
    kmeans_error = _kmeans_error(R, cp.dot(Z_norm, Y_norm.T))
    R_normalized = R / R.sum(axis=1, keepdims=True)
    entropy = _entropy_kernel(R_normalized)
    entropy_term = sigma * entropy
    diversity_penalty = sigma * cp.sum(cp.dot(theta, _log_div_OE(O, E)))
    objective = kmeans_error + entropy_term + diversity_penalty
    objective_arr.append(objective)


def _is_convergent_harmony(objectives_harmony: list, tol: float) -> bool:
    """
    Check if the Harmony algorithm has converged based on the objective function values.

    Returns True if the relative improvement in objective is below tolerance.
    """
    if len(objectives_harmony) < 2:
        return False

    obj_old = objectives_harmony[-2]
    obj_new = objectives_harmony[-1]

    return (obj_old - obj_new) < tol * np.abs(obj_old)


def _is_convergent_clustering(
    objectives_clustering: list, tol: float, window_size: int = 3
) -> bool:
    """
    Check if the clustering step has converged based on the objective function values.

    Uses a window of objective values to determine convergence.
    """
    if len(objectives_clustering) < window_size + 1:
        return False

    obj_old = 0.0
    obj_new = 0.0
    for i in range(window_size):
        obj_old += objectives_clustering[-2 - i]
        obj_new += objectives_clustering[-1 - i]

    return (obj_old - obj_new) < tol * np.abs(obj_old)
