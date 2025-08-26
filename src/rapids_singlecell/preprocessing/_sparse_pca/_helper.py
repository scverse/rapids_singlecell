from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp

from ._kernels._pca_sparse_kernel import _copy_kernel, _cov_kernel

if TYPE_CHECKING:
    from cupyx.scipy.sparse import spmatrix


def _copy_gram(gram_matrix, n_cols):
    """
    Flips the upper triangle of the gram matrix to the lower triangle. This is necessary because the kernel only computes the upper triangle.
    """
    copy_gram = _copy_kernel(gram_matrix.dtype)
    block = (32, 32)
    grid = (math.ceil(n_cols / block[0]), math.ceil(n_cols / block[1]))
    copy_gram(
        grid,
        block,
        (gram_matrix, n_cols),
    )
    return gram_matrix


def _compute_cov(cov_result, gram_matrix, mean_x):
    compute_cov = _cov_kernel(gram_matrix.dtype)

    block_size = (32, 32)
    grid_size = (math.ceil(gram_matrix.shape[0] / 8),) * 2
    compute_cov(
        grid_size,
        block_size,
        (cov_result, gram_matrix, mean_x, mean_x, gram_matrix.shape[0]),
    )
    return cov_result


def _check_matrix_for_zero_genes(X: spmatrix) -> None:
    gene_ex = cp.zeros(X.shape[1], dtype=cp.int32)

    from ._kernels._pca_sparse_kernel import _zero_genes_kernel

    block = (32,)
    grid = (int(math.ceil(X.nnz / block[0])),)
    _zero_genes_kernel(
        grid,
        block,
        (
            X.indices,
            gene_ex,
            X.nnz,
        ),
    )
    if cp.any(gene_ex == 0):
        raise ValueError(
            "There are genes with zero expression. "
            "Please remove them before running PCA."
        )
