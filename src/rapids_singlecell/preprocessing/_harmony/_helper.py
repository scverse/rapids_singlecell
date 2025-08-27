from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from ._kernels._kmeans import _get_kmeans_err_kernel
from ._kernels._normalize import _get_normalize_kernel_optimized
from ._kernels._outer import (
    _get_colsum_atomic_kernel,
    _get_colsum_kernel,
    _get_harmony_correction_kernel,
    _get_outer_kernel,
)
from ._kernels._pen import _get_pen_kernel
from ._kernels._scatter_add import (
    _get_aggregated_matrix_kernel,
    _get_scatter_add_kernel_optimized,
    _get_scatter_add_kernel_with_bias_block,
    _get_scatter_add_kernel_with_bias_cat0,
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


def _create_category_index_mapping(
    cats: cp.ndarray, n_batches: int
) -> tuple[cp.ndarray, cp.ndarray]:
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

    threads_per_block = 1024
    if n_cells < 300_000:
        blocks = int((n_pcs + 1) / 2)
        scatter_kernel0 = _get_scatter_add_kernel_with_bias_cat0(X.dtype)
        scatter_kernel0(
            (blocks, 8), (threads_per_block,), (X, n_cells, n_pcs, out, bias)
        )
    else:
        out[0] = X.T @ bias
    blocks = int((n_batches) * (n_pcs + 1) / 2)
    scatter_kernel = _get_scatter_add_kernel_with_bias_block(X.dtype)
    scatter_kernel(
        (blocks,),
        (threads_per_block,),
        (X, cat_offsets, cell_indices, n_cells, n_pcs, n_batches, out, bias),
    )


def _kmeans_error(R: cp.ndarray, dot: cp.ndarray) -> float:
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


def _column_sum(X: cp.ndarray) -> cp.ndarray:
    """
    Sum each column of the 2D, C-contiguous float32 array A.
    Returns a 1D float32 cupy array of length A.shape[1].
    """
    rows, cols = X.shape
    if not X.flags.c_contiguous:
        return X.sum(axis=0)

    out = cp.zeros(cols, dtype=X.dtype)

    dev = cp.cuda.Device()
    nSM = dev.attributes["MultiProcessorCount"]
    max_blocks = nSM * 8
    threads = max(int(round(1 / 32) * 32), 32)
    blocks = min(cols, max_blocks)
    _colsum = _get_colsum_kernel(X.dtype)
    _colsum((blocks,), (threads,), (X, out, rows, cols))
    return out


def _column_sum_atomic(X: cp.ndarray) -> cp.ndarray:
    """
    Sum each column of the 2D, C-contiguous array A
    using 32×32 tiles + one atomic per tile-column.
    """
    assert X.ndim == 2
    rows, cols = X.shape
    if not X.flags.c_contiguous:
        return X.sum(axis=0)

    out = cp.zeros(cols, dtype=X.dtype)
    tile_rows = (rows + 31) // 32
    tile_cols = (cols + 31) // 32
    blocks = tile_rows * tile_cols
    threads = (32, 32)

    kernel = _get_colsum_atomic_kernel(X.dtype)
    kernel((blocks,), threads, (X, out, rows, cols))
    return out


def _gemm_colsum(X: cp.ndarray) -> cp.ndarray:
    """
    Sum each column with cuBLAS GEMM
    """
    return X.T @ cp.ones(X.shape[0], dtype=X.dtype)


def _choose_colsum_algo_heuristic(rows: int, cols: int, algo: str | None) -> callable:
    """
    Returns one of:
    - _colsum_columns
    - _colsum_atomics
    - _gemm_colsum
    - cp.sum (with axis=0)
    """
    # first pick the strategy string
    if algo is None:
        cc = cp.cuda.Device().compute_capability
        algo = _colsum_heuristic(rows, cols, cc)
    if algo == "cupy":
        return lambda X: X.sum(axis=0)
    if algo == "columns":
        return _column_sum
    if algo == "atomics":
        return _column_sum_atomic
    if algo == "gemm":
        return _gemm_colsum
    # fallback: global CuPy reduction
    return lambda X: X.sum(axis=0)


# TODO: Make this more robust
def _colsum_heuristic(rows: int, cols: int, compute_capability: str) -> str:
    is_data_center = compute_capability in ["100", "90"]
    if cols < 200 and rows < 20000:
        return "columns"
    if cols < 200 and rows < 100000 and is_data_center:
        return "columns"
    if cols < 800 and rows < 10000:
        return "atomics"
    if cols < 1024 and rows < 5000:
        return "atomics"
    if cols < 800 and rows < 20000 and is_data_center:
        return "atomics"
    if cols < 2000 and rows < 10000 and is_data_center:
        return "atomics"
    if rows >= 5000:
        return "gemm"
    return "cupy"


# TODO: Make this more robust
def _benchmark_colsum_algorithms(
    shape: tuple[int, int],
    dtype: cp.dtype = cp.float32,
    n_warmup: int = 1,
    n_trials: int = 3,
) -> callable:
    """
    Benchmark all column sum algorithms and return the fastest one.
    Parameters
    ----------
    shape
        Shape of the test matrix (rows, cols)
    dtype
        Data type for the test matrix
    n_warmup
        Number of warmup iterations
    n_trials
        Number of benchmark trials
    Returns
    -------
    Name of the fastest algorithm: 'cupy', 'columns', 'atomics', or 'gemm'
    """
    rows, cols = shape

    # Create test data
    X = cp.random.random(shape, dtype=dtype)

    # Ensure it's C-contiguous for fair comparison
    if not X.flags.c_contiguous:
        X = cp.ascontiguousarray(X)

    algorithms = {
        "cupy": lambda x: x.sum(axis=0),
        "columns": _column_sum,
        "atomics": _column_sum_atomic,
        "gemm": _gemm_colsum,
    }

    results = {}

    for name, func in algorithms.items():
        # Warmup
        for _ in range(n_warmup):
            try:
                _ = func(X)
                cp.cuda.Stream.null.synchronize()
            except Exception:  # noqa: BLE001
                # If algorithm fails, skip it
                results[name] = float("inf")
                break
        else:
            # Benchmark
            times = []
            for _ in range(n_trials):
                try:
                    start_event = cp.cuda.Event()
                    end_event = cp.cuda.Event()

                    start_event.record()
                    _ = func(X)
                    end_event.record()
                    end_event.synchronize()
                    elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
                    times.append(elapsed_ms)
                except Exception:  # noqa: BLE001
                    results[name] = float("inf")
                    break
            else:
                # Use median time for robustness
                results[name] = cp.median(cp.array(times))

    # Return the algorithm with minimum time
    fastest_algo = min(results.items(), key=lambda x: x[1])[0]

    return algorithms[fastest_algo], fastest_algo


def _choose_colsum_algo_benchmark(
    rows: int,
    cols: int,
    dtype: cp.dtype = cp.float32,
    *,
    verbose: bool = True,
) -> callable:
    """
    Automatically choose the best column sum algorithm by benchmarking.

    Parameters
    ----------
    rows
        Number of rows
    cols
        Number of columns
    dtype
        Data type
    verbose
        Whether to print the chosen algorithm
    Returns
    -------
    Function of the fastest algorithm
    """
    func, algo = _benchmark_colsum_algorithms((rows, cols), dtype)
    if verbose:
        print(f"Using {algo} for column sum")
    return func


def _penalty_term(R: cp.ndarray, penalty: cp.ndarray, cats: cp.ndarray) -> cp.ndarray:
    """
    Calculate the penalty term for the Harmony algorithm.
    """
    n_cats, n_pcs = R.shape
    N = n_cats * n_pcs
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block
    pen_kernel = _get_pen_kernel(R.dtype)
    pen_kernel((blocks,), (threads_per_block,), (R, penalty, cats, n_cats, n_pcs))
    return R
