from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cupyx.scipy import sparse

from .kernels._autocorr import (
    get_morans_I_num_dense_kernel,
    get_morans_I_num_sparse_kernel,
    get_pre_den_sparse_kernel,
)

if TYPE_CHECKING:
    from cupyx.scipy.sparse import csr_matrix


def _morans_I_cupy_dense(
    data: cp.ndarray,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    n_samples, n_features = data.shape
    dtype = data.dtype
    data_centered_cupy = data - data.mean(axis=0)

    # Calculate the numerator and denominator for Moran's I
    num = cp.zeros(n_features, dtype=dtype)
    block_size = 8
    fg = int(math.ceil(n_features / block_size))
    sg = int(math.ceil(n_samples / block_size))
    grid_size = (fg, sg, 1)

    num_kernel = get_morans_I_num_dense_kernel(np.dtype(dtype))
    num_kernel(
        grid_size,
        (block_size, block_size, 1),
        (
            data_centered_cupy,
            adj_matrix_cupy.indptr,
            adj_matrix_cupy.indices,
            adj_matrix_cupy.data,
            num,
            n_samples,
            n_features,
        ),
    )

    # Calculate the denominator for Moran's I
    den = cp.sum(data_centered_cupy**2, axis=0)

    # Calculate Moran's I
    morans_I = num / den

    # Calculate p-values using permutation tests
    if n_permutations:
        morans_I_permutations = cp.zeros((n_permutations, n_features), dtype=dtype)
        num_permuted = cp.zeros(n_features, dtype=dtype)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_kernel(
                grid_size,
                (block_size, block_size, 1),
                (
                    data_centered_cupy,
                    adj_matrix_permuted.indptr,
                    adj_matrix_permuted.indices,
                    adj_matrix_permuted.data,
                    num_permuted,
                    n_samples,
                    n_features,
                ),
            )
            morans_I_permutations[p, :] = num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _morans_I_cupy_sparse(
    data: csr_matrix,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    n_samples, n_features = data.shape
    dtype = data.dtype

    # Calculate the numerator for Moran's I
    num = cp.zeros(n_features, dtype=dtype)
    num_kernel = get_morans_I_num_sparse_kernel(np.dtype(dtype))
    means = data.mean(axis=0).ravel()

    sg = n_samples
    # Launch the kernel
    num_kernel(
        (sg,),
        (1024,),
        (
            adj_matrix_cupy.indptr,
            adj_matrix_cupy.indices,
            adj_matrix_cupy.data,
            data.indptr,
            data.indices,
            data.data,
            n_samples,
            n_features,
            means,
            num,
        ),
    )

    # Calculate the denominator for Moran's I
    den = cp.zeros(n_features, dtype=dtype)
    counter = cp.zeros(n_features, dtype=cp.int32)
    block_den = math.ceil(data.nnz / 32)
    pre_den_kernel = get_pre_den_sparse_kernel(np.dtype(dtype))

    pre_den_kernel(
        (block_den,), (32,), (data.indices, data.data, data.nnz, means, den, counter)
    )
    counter = n_samples - counter
    den += counter * means**2

    # Calculate Moran's I
    morans_I = num / den

    if n_permutations:
        morans_I_permutations = cp.zeros((n_permutations, n_features), dtype=dtype)
        num_permuted = cp.zeros(n_features, dtype=dtype)

        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_permuted = cp.zeros(n_features, dtype=dtype)
            num_kernel(
                (sg,),
                (1024,),
                (
                    adj_matrix_permuted.indptr,
                    adj_matrix_permuted.indices,
                    adj_matrix_permuted.data,
                    data.indptr,
                    data.indices,
                    data.data,
                    n_samples,
                    n_features,
                    means,
                    num_permuted,
                ),
            )

            morans_I_permutations[p, :] = num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _morans_I_cupy(
    data: cp.ndarray | csr_matrix,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    if sparse.isspmatrix_csr(data):
        return _morans_I_cupy_sparse(data, adj_matrix_cupy, n_permutations)
    elif isinstance(data, cp.ndarray):
        return _morans_I_cupy_dense(data, adj_matrix_cupy, n_permutations)
    else:
        raise ValueError("Datatype not supported")
