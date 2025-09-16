from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from cupyx.scipy.sparse import spmatrix
try:
    from rapids_singlecell._cuda import _spca_cuda as _spca
except ImportError:
    _spca = None


def _copy_gram(gram_matrix: cp.ndarray, n_cols: int) -> cp.ndarray:
    _spca.copy_upper_to_lower(
        gram_matrix.data.ptr, int(n_cols), int(cp.dtype(gram_matrix.dtype).itemsize)
    )
    return gram_matrix


def _compute_cov(
    cov_result: cp.ndarray, gram_matrix: cp.ndarray, mean_x: cp.ndarray
) -> cp.ndarray:
    _spca.cov_from_gram(
        cov_result.data.ptr,
        gram_matrix.data.ptr,
        mean_x.data.ptr,
        mean_x.data.ptr,
        int(gram_matrix.shape[0]),
        int(cp.dtype(gram_matrix.dtype).itemsize),
    )
    return cov_result


def _check_matrix_for_zero_genes(X: spmatrix) -> None:
    gene_ex = cp.zeros(X.shape[1], dtype=cp.int32)
    _spca.check_zero_genes(
        X.indices.data.ptr,
        gene_ex.data.ptr,
        int(X.nnz),
    )
    if cp.any(gene_ex == 0):
        raise ValueError(
            "There are genes with zero expression. Please remove them before running PCA."
        )
