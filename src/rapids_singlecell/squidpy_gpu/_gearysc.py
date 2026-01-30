from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cupyx.scipy import sparse

from ._utils import _check_precision_issues
from .kernels._autocorr import (
    get_gearys_C_num_dense_kernel,
    get_gearys_C_num_sparse_kernel,
    get_pre_den_sparse_kernel,
)

if TYPE_CHECKING:
    from cupyx.scipy.sparse import csr_matrix


def _gearys_C_cupy_dense(
    data: cp.ndarray,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    n_samples, n_features = data.shape
    dtype = data.dtype

    # Calculate the numerator for Geary's C
    num = cp.zeros(n_features, dtype=dtype)
    num_kernel = get_gearys_C_num_dense_kernel(np.dtype(dtype))

    block_size = 8
    fg = int(math.ceil(n_features / block_size))
    sg = int(math.ceil(n_samples / block_size))
    grid_size = (fg, sg, 1)
    num_kernel(
        grid_size,
        (block_size, block_size, 1),
        (
            data,
            adj_matrix_cupy.indptr,
            adj_matrix_cupy.indices,
            adj_matrix_cupy.data,
            num,
            n_samples,
            n_features,
        ),
    )

    # Calculate the denominator for Geary's C
    gene_mean = data.mean(axis=0).ravel()
    preden = cp.sum((data - gene_mean) ** 2, axis=0)
    den = 2 * adj_matrix_cupy.sum() * preden

    # Calculate Geary's C
    gearys_C = (n_samples - 1) * num / den

    # Check for numerical issues before expensive permutations
    _check_precision_issues(gearys_C, dtype)

    # Calculate p-values using permutation tests
    if n_permutations:
        gearys_C_permutations = cp.zeros((n_permutations, n_features), dtype=dtype)
        num_permuted = cp.zeros(n_features, dtype=dtype)

        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_kernel(
                grid_size,
                (block_size, block_size, 1),
                (
                    data,
                    adj_matrix_permuted.indptr,
                    adj_matrix_permuted.indices,
                    adj_matrix_permuted.data,
                    num_permuted,
                    n_samples,
                    n_features,
                ),
            )
            gearys_C_permutations[p, :] = (n_samples - 1) * num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        gearys_C_permutations = None
    return gearys_C, gearys_C_permutations


def _gearys_C_cupy_sparse(
    data: csr_matrix,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    n_samples, n_features = data.shape
    dtype = data.dtype

    # Calculate the numerator for Geary's C
    num = cp.zeros(n_features, dtype=dtype)
    num_kernel = get_gearys_C_num_sparse_kernel(np.dtype(dtype))

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
            num,
        ),
    )

    # Calculate the denominator for Geary's C
    means = data.mean(axis=0).ravel()
    den = cp.zeros(n_features, dtype=dtype)
    counter = cp.zeros(n_features, dtype=cp.int32)
    block_den = math.ceil(data.nnz / 32)
    pre_den_kernel = get_pre_den_sparse_kernel(np.dtype(dtype))

    pre_den_kernel(
        (block_den,), (32,), (data.indices, data.data, data.nnz, means, den, counter)
    )
    counter = n_samples - counter
    den += counter * means**2
    den *= 2 * adj_matrix_cupy.sum()

    # Calculate Geary's C
    gearys_C = (n_samples - 1) * num / den

    # Check for numerical issues before expensive permutations
    _check_precision_issues(gearys_C, dtype)

    # Calculate p-values using permutation tests
    if n_permutations:
        gearys_C_permutations = cp.zeros((n_permutations, n_features), dtype=dtype)
        num_permuted = cp.zeros(n_features, dtype=dtype)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
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
                    num_permuted,
                ),
            )
            gearys_C_permutations[p, :] = (n_samples - 1) * num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        gearys_C_permutations = None
    return gearys_C, gearys_C_permutations


def _gearys_C_cupy(
    data: cp.ndarray | csr_matrix,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    if sparse.isspmatrix_csr(data):
        return _gearys_C_cupy_sparse(data, adj_matrix_cupy, n_permutations)
    elif isinstance(data, cp.ndarray):
        return _gearys_C_cupy_dense(data, adj_matrix_cupy, n_permutations)
    else:
        raise ValueError("Datatype not supported")
