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
        out=gram_matrix.data.ptr,
        ncols=n_cols,
        itemsize=cp.dtype(gram_matrix.dtype).itemsize,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return gram_matrix


def _compute_cov(
    cov_result: cp.ndarray, gram_matrix: cp.ndarray, mean_x: cp.ndarray
) -> cp.ndarray:
    _spca.cov_from_gram(
        gram_matrix.data.ptr,
        mean_x.data.ptr,
        mean_x.data.ptr,
        cov=cov_result.data.ptr,
        ncols=gram_matrix.shape[0],
        itemsize=cp.dtype(gram_matrix.dtype).itemsize,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return cov_result


def _check_matrix_for_zero_genes(X: spmatrix) -> None:
    gene_ex = cp.zeros(X.shape[1], dtype=cp.int32)
    if X.nnz > 0:
        _spca.check_zero_genes(
            X.indices.data.ptr,
            out=gene_ex.data.ptr,
            nnz=X.nnz,
            num_genes=X.shape[1],
            stream=cp.cuda.get_current_stream().ptr,
        )
    if cp.any(gene_ex == 0):
        raise ValueError(
            "There are genes with zero expression. Please remove them before running PCA."
        )
