from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cuml.internals.logger as logger
import cupy as cp
import cuvs
import numpy as np
from cuml.manifold.simpl_set import fuzzy_simplicial_set
from cupyx.scipy import sparse as cp_sparse
from packaging.version import parse as parse_version
from scipy import sparse as sc_sparse

from rapids_singlecell._utils import _get_logger_level

if TYPE_CHECKING:
    from rapids_singlecell._utils import AnyRandom
    from rapids_singlecell.preprocessing._neighbors import _Algorithms, _Metrics


def _compute_nlist(N):
    base = math.sqrt(N)
    next_pow2 = 2 ** math.ceil(math.log2(base))
    return int(next_pow2 * 2)


def _cuvs_switch():
    return parse_version(cuvs.__version__) > parse_version("24.10")


def _check_neighbors_X(
    X: cp_sparse.spmatrix | sc_sparse.spmatrix | np.ndarray | cp.ndarray,
    algorithm: _Algorithms,
) -> cp_sparse.spmatrix | cp.ndarray | np.ndarray:
    """Check and convert input X to the expected format based on algorithm.

    Parameters
    ----------
    X (array-like or sparse matrix): Input data.
    algorithm (str): The algorithm to be used.
    use_multi_gpu (bool): Whether to use multi-gpu mode.

    Returns
    -------
    X_contiguous (cupy.ndarray or sparse.csr_matrix): Contiguous array or CSR matrix.

    """
    if cp_sparse.issparse(X) or sc_sparse.issparse(X):
        if algorithm != "brute":
            raise ValueError(
                f"Sparse input is not supported for {algorithm} algorithm. Use 'brute' instead."
            )
        X_contiguous = X.tocsr()
    elif algorithm in ["all_neighbors", "mg_ivfflat", "mg_ivfpq"]:
        if isinstance(X, np.ndarray):
            X_contiguous = np.asarray(X, order="C", dtype=np.float32)
        elif isinstance(X, cp.ndarray):
            X_contiguous = cp.ascontiguousarray(X, dtype=cp.float32).get()
        else:
            raise TypeError(
                "Unsupported type for X. Expected ndarray or sparse matrix."
            )
    else:
        if isinstance(X, np.ndarray):
            X_contiguous = cp.asarray(X, order="C", dtype=np.float32)
        elif isinstance(X, cp.ndarray):
            X_contiguous = cp.ascontiguousarray(X, dtype=np.float32)
        else:
            raise TypeError(
                "Unsupported type for X. Expected ndarray or sparse matrix."
            )

    return X_contiguous


def _check_metrics(algorithm: _Algorithms, metric: _Metrics) -> bool:
    """Check if the provided metric is compatible with the chosen algorithm.

    Parameters
    ----------
    algorithm (str): The algorithm to be used.
    metric (str): The metric for distance computation.

    Returns
    -------
    bool: True if the metric is compatible, otherwise ValueError is raised.

    """
    if algorithm == "brute":
        # 'brute' support all metrics, no need to check further.
        return True
    elif algorithm == "cagra":
        if metric not in ["euclidean", "sqeuclidean", "inner_product"]:
            raise ValueError(
                "cagra only supports 'euclidean', 'inner_product' and 'sqeuclidean' metrics."
            )
    elif algorithm in ["ivfpq", "ivfflat", "mg_ivfflat", "mg_ivfpq"]:
        if metric not in ["euclidean", "sqeuclidean", "inner_product", "cosine"]:
            raise ValueError(
                f"{algorithm} only supports 'euclidean', 'sqeuclidean', 'cosine', and 'inner_product' metrics."
            )
    elif algorithm == "nn_descent":
        if metric not in ["euclidean", "sqeuclidean", "cosine", "inner_product"]:
            raise ValueError(
                "nn_descent only supports 'euclidean', 'sqeuclidean', 'inner_product' and 'cosine' metrics."
            )
    elif algorithm == "all_neighbors":
        if metric not in ["euclidean", "sqeuclidean"]:
            raise ValueError(
                "all_neighbors only supports 'euclidean' and 'sqeuclidean' metrics."
            )
    else:
        raise NotImplementedError(f"The {algorithm} algorithm is not implemented yet.")

    return True


def _get_connectivities(
    n_neighbors: int,
    *,
    n_obs: int,
    random_state: AnyRandom,
    metric: _Metrics,
    knn_indices: cp.ndarray,
    knn_dist: cp.ndarray,
) -> cp_sparse.coo_matrix:
    set_op_mix_ratio = 1.0
    local_connectivity = 1.0
    X_conn = cp.empty((n_obs, 1), dtype=np.float32)
    logger_level = _get_logger_level(logger)
    connectivities = fuzzy_simplicial_set(
        X_conn,
        n_neighbors,
        random_state,
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_dist,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )
    logger.set_level(logger_level)
    return connectivities


def _trimming(cnts: cp_sparse.csr_matrix, trim: int) -> cp_sparse.csr_matrix:
    from ._kernels._bbknn import cut_smaller_func, find_top_k_per_row_kernel

    n_rows = cnts.shape[0]
    vals_gpu = cp.zeros(n_rows, dtype=cp.float32)

    threads_per_block = 64
    blocks_per_grid = (n_rows + threads_per_block - 1) // threads_per_block

    shared_mem_per_thread = trim * cp.dtype(cp.float32).itemsize
    shared_mem_size = threads_per_block * shared_mem_per_thread

    find_top_k_per_row_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (cnts.data, cnts.indptr, cnts.shape[0], trim, vals_gpu),
        shared_mem=shared_mem_size,
    )
    cut_smaller_func(
        (cnts.shape[0],),
        (64,),
        (cnts.indptr, cnts.indices, cnts.data, vals_gpu, cnts.shape[0]),
    )
    cnts.eliminate_zeros()
    return cnts
