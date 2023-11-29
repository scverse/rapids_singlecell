"""
This is still under active development


"""
# nF811
import cupy as cp
import numpy as np
import pandas as pd
from scipy import sparse

from ._kernels._aggr_kernels import _div_kernel, _div_kernel2, _get_aggr_kernel


def _combine_categories(label_df: pd.DataFrame, cols: list[str]) -> pd.Categorical:
    from itertools import product

    if isinstance(cols, str):
        cols = [cols]

    df = pd.DataFrame(
        {c: pd.Categorical(label_df[c]).remove_unused_categories() for c in cols},
    )
    result_categories = [
        "_".join(map(str, x)) for x in product(*[df[c].cat.categories for c in cols])
    ]
    n_categories = [len(df[c].cat.categories) for c in cols]

    factors = np.ones(len(cols) + 1, dtype=np.int32)  # First factor needs to be 1
    np.cumsum(n_categories[::-1], out=factors[1:])
    factors = factors[:-1][::-1]

    # TODO: pick a more optimal bit width
    final_codes = np.zeros(df.shape[0], dtype=np.int32)
    for factor, c in zip(factors, cols):
        final_codes += df[c].cat.codes * factor

    return pd.Categorical.from_codes(
        final_codes, categories=result_categories
    ).remove_unused_categories()


def sparse_indicator(
    categorical, weights: None | np.ndarray = None
) -> sparse.coo_matrix:
    if weights is None:
        weights = np.broadcast_to(1.0, len(categorical))
    A = sparse.coo_matrix(
        (weights, (categorical.codes, np.arange(len(categorical)))),
        shape=(len(categorical.categories), len(categorical)),
    )
    return A


# cats = _combine_categories(adata.obs, "CellType")
# groupby_mat = sparse_cp.csr_matrix(sparse_indicator(cats))
def count_mean_var_sparse(by, data):
    sums = by @ data
    counts = by @ data._with_data(cp.ones(len(data.data), dtype=data.data.dtype))
    means = sums.copy()
    n_cells = by.sum(axis=1).astype(data.dtype)
    block = (128,)
    grid = (by.shape[0],)
    divide_sparse = _div_kernel(data.dtype)
    divide_sparse2 = _div_kernel2(data.dtype)
    divide_sparse(block, grid, (means.indptr, means.data, by.shape[0], n_cells))
    sq_mean = by @ data.multiply(data)
    divide_sparse(block, grid, (sq_mean.indptr, sq_mean.data, by.shape[0], n_cells))
    var = sq_mean - means.multiply(means)
    divide_sparse2(block, grid, (var.indptr, var.data, by.shape[0], n_cells))
    return sums, counts, means, var


def count_mean_var_sparse(gmat, X):
    means = cp.zeros((gmat.shape[0], X.shape[1]), dtype=cp.float64)
    var = cp.zeros((gmat.shape[0], X.shape[1]), dtype=cp.float64)
    sums = cp.zeros((gmat.shape[0], X.shape[1]), dtype=cp.float64)
    count = cp.zeros((gmat.shape[0], X.shape[1]), dtype=cp.int32)
    block = (512,)
    grid = (gmat.shape[0],)
    get_mean_var_minor = _get_aggr_kernel(X.data.dtype)
    get_mean_var_minor(
        grid,
        block,
        (
            gmat.indptr,
            gmat.indices,
            X.indptr,
            X.indices,
            X.data,
            count,
            sums,
            means,
            var,
            gmat.shape[0],
            X.shape[1],
        ),
    )
    n_cells = gmat.sum(axis=1).astype(cp.float64)
    var = var - cp.power(means, 2)
    var *= n_cells / (n_cells - 1)
    return sums, count, means, var


def count_mean_var_dense(by, data):
    # todo add custom kernels
    n_cells = by.sum(axis=1).astype(data.dtype)
    by = by.toarray()

    sums = by @ data

    set_non_zero_to_one = cp.ElementwiseKernel(
        "T x",
        "T y",
        """
        if (x != 0) { y = 1; }
        else { y = 0; }
        """,
        "set_non_zero_to_one",
    )
    nnz = cp.empty_like(data)
    set_non_zero_to_one(data, nnz)
    counts = by @ nnz

    means = sums / n_cells
    sq_mean = by @ data**2 / n_cells
    var = sq_mean - means**2
    var *= n_cells / (n_cells - 1)
    return sums, counts, means, var
