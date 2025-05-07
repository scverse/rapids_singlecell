from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from ._kernels._kmeans import _get_kmeans_err_kernel
from ._kernels._normalize import _get_normalize_kernel_optimized
from ._kernels._outer import _get_harmony_correction_kernel, _get_outer_kernel
from ._kernels._scatter_add import (
    _get_aggregated_matrix_kernel,
    _get_scatter_add_kernel_optimized,
    _get_scatter_add_kernel_with_bias_block,
)

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


def _scatter_add_cp(
    X: cp.ndarray,
    out: cp.ndarray,
    cats: cp.ndarray,
    switcher: int,
) -> None:
    """
    Scatter add operation for Harmony algorithm.
    """
    n_cells = X.shape[0]
    n_pcs = X.shape[1]
    N = n_cells * n_pcs
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block

    scatter_add_kernel = _get_scatter_add_kernel_optimized(X.dtype)
    scatter_add_kernel((blocks,), (256,), (X, cats, n_cells, n_pcs, switcher, out))


def _Z_correction(
    Z: cp.ndarray,
    W: cp.ndarray,
    cats: cp.ndarray,
    R: cp.ndarray,
) -> None:
    """
    Scatter add operation for Harmony algorithm.
    """
    n_cells = Z.shape[0]
    n_pcs = Z.shape[1]
    N = n_cells * n_pcs
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block

    scatter_add_kernel = _get_harmony_correction_kernel(Z.dtype)
    scatter_add_kernel((blocks,), (256,), (Z, W, cats, R, n_cells, n_pcs))


def _outer_cp(
    E: cp.ndarray, Pr_b: cp.ndarray, R_sum: cp.ndarray, switcher: int
) -> None:
    n_cats, n_pcs = E.shape

    # Determine the total number of elements to process and configure the grid.
    N = n_cats * n_pcs
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block
    outer_kernel = _get_outer_kernel(E.dtype)
    outer_kernel(
        (blocks,), (threads_per_block,), (E, Pr_b, R_sum, n_cats, n_pcs, switcher)
    )


def _normalize_cp(X: cp.ndarray, p: int = 2) -> cp.ndarray:
    """
    Analogous to `torch.nn.functional.normalize` for `axis = 1`, `p` in numpy is known as `ord`.
    """
    if p == 2:
        return X / cp.linalg.norm(X, ord=2, axis=1, keepdims=True).clip(min=1e-12)

    else:
        return _normalize_cp_p1(X)


def _get_aggregated_matrix(
    aggregated_matrix: cp.ndarray, sum: cp.ndarray, n_batches: int
) -> None:
    """
    Get the aggregated matrix for the correction step.
    """
    aggregated_matrix_kernel = _get_aggregated_matrix_kernel(aggregated_matrix.dtype)

    threads_per_block = 32
    blocks = (n_batches + 1 + threads_per_block - 1) // threads_per_block
    aggregated_matrix_kernel(
        (blocks,), (threads_per_block,), (aggregated_matrix, sum, sum.sum(), n_batches)
    )


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


def _create_category_index_mapping(cats, n_batches):
    """
    Create a CSR-like data structure mapping categories to cell indices using lexicographical sort.
    """
    cat_counts = cp.zeros(n_batches, dtype=cp.int32)
    cp.add.at(cat_counts, cats, 1)
    cat_offsets = cp.zeros(n_batches + 1, dtype=cp.int32)
    cp.cumsum(cat_counts, out=cat_offsets[1:])

    n_cells = cats.shape[0]
    indices = cp.arange(n_cells, dtype=cp.int32)

    cell_indices = cp.lexsort(cp.stack((indices, cats))).astype(cp.int32)
    return cat_offsets, cell_indices


def _scatter_add_cp_bias_csr(
    X: cp.ndarray,
    out: cp.ndarray,
    *,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    bias: cp.ndarray,
    n_batches: int,
) -> None:
    n_cells = X.shape[0]
    n_pcs = X.shape[1]

    blocks = int((n_batches + 1) * (n_pcs + 1) / 2)

    threads_per_block = 1024

    scatter_kernel = _get_scatter_add_kernel_with_bias_block(X.dtype)
    scatter_kernel(
        (blocks,),
        (threads_per_block,),
        (X, cat_offsets, cell_indices, n_cells, n_pcs, n_batches, out, bias),
    )


def _kmeans_error(R, dot):
    """Optimized raw CUDA implementation of kmeans error calculation"""
    assert R.size == dot.size and R.dtype == dot.dtype

    out = cp.zeros(1, dtype=R.dtype)
    threads = 256
    blocks = min(
        (R.size + threads - 1) // threads,
        cp.cuda.Device().attributes["MultiProcessorCount"] * 8,
    )
    kernel = _get_kmeans_err_kernel(R.dtype.name)
    kernel((blocks,), (threads,), (R, dot, R.size, out))

    return out[0]


def _get_theta_array(
    theta: float | int | list[float | int] | np.ndarray | cp.ndarray,
    n_batches: int,
    dtype: cp.dtype,
) -> cp.ndarray:
    """
    Convert theta parameter to a CuPy array of appropriate shape.
    """
    # Handle scalar inputs (float, int)
    if isinstance(theta, float | int):
        return cp.ones(n_batches, dtype=dtype) * float(theta)

    # Handle array-like inputs (list, numpy array, cupy array)
    if isinstance(theta, list):
        theta_array = cp.array(theta, dtype=dtype)
    elif isinstance(theta, np.ndarray):
        theta_array = cp.array(theta, dtype=dtype)
    elif isinstance(theta, cp.ndarray):
        theta_array = theta.astype(dtype)
    else:
        raise ValueError(
            f"Theta must be float, int, list, numpy array, or cupy array, got {type(theta)}"
        )

    # Verify dimensions
    if theta_array.size != n_batches:
        raise ValueError(
            f"Theta array size ({theta_array.size}) must match number of batches ({n_batches})"
        )

    return theta_array.ravel()
