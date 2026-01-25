from __future__ import annotations

import cupy as cp
from cupyx.scipy import sparse

try:
    from rapids_singlecell._cuda import _autocorr_cuda as _ac
except ImportError:
    _ac = None


def _gearys_C_cupy_dense(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    # Calculate the numerator for Geary's C
    num = cp.zeros(n_features, dtype=cp.float32)
    _ac.gearys_dense(
        data,
        adj_row_ptr=adj_matrix_cupy.indptr,
        adj_col_ind=adj_matrix_cupy.indices,
        adj_data=adj_matrix_cupy.data,
        num=num,
        n_samples=n_samples,
        n_features=n_features,
        stream=cp.cuda.get_current_stream().ptr,
    )
    # Calculate the denominator for Geary's C
    gene_mean = data.mean(axis=0).ravel()
    preden = cp.sum((data - gene_mean) ** 2, axis=0)
    den = 2 * adj_matrix_cupy.sum() * preden

    # Calculate Geary's C
    gearys_C = (n_samples - 1) * num / den

    # Calculate p-values using permutation tests
    if n_permutations:
        gearys_C_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        num_permuted = cp.zeros(n_features, dtype=data.dtype)

        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            _ac.gearys_dense(
                data,
                adj_row_ptr=adj_matrix_permuted.indptr,
                adj_col_ind=adj_matrix_permuted.indices,
                adj_data=adj_matrix_permuted.data,
                num=num_permuted,
                n_samples=n_samples,
                n_features=n_features,
                stream=cp.cuda.get_current_stream().ptr,
            )
            gearys_C_permutations[p, :] = (n_samples - 1) * num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        gearys_C_permutations = None
    return gearys_C, gearys_C_permutations


def _gearys_C_cupy_sparse(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    # Calculate the numerator for Geary's C
    num = cp.zeros(n_features, dtype=cp.float32)

    n_samples, n_features = data.shape
    _ac.gearys_sparse(
        adj_matrix_cupy.indptr,
        adj_matrix_cupy.indices,
        adj_matrix_cupy.data,
        data_row_ptr=data.indptr,
        data_col_ind=data.indices,
        data_values=data.data,
        n_samples=n_samples,
        n_features=n_features,
        num=num,
        stream=cp.cuda.get_current_stream().ptr,
    )
    # Calculate the denominator for Geary's C
    means = data.mean(axis=0).ravel()
    den = cp.zeros(n_features, dtype=cp.float32)
    counter = cp.zeros(n_features, dtype=cp.int32)
    _ac.pre_den_sparse(
        data.indices,
        data.data,
        nnz=data.nnz,
        mean_array=means,
        den=den,
        counter=counter,
        stream=cp.cuda.get_current_stream().ptr,
    )
    counter = n_samples - counter
    den += counter * means**2
    den *= 2 * adj_matrix_cupy.sum()

    # Calculate Geary's C
    gearys_C = (n_samples - 1) * num / den

    # Calculate p-values using permutation tests
    if n_permutations:
        gearys_C_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        num_permuted = cp.zeros(n_features, dtype=data.dtype)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            _ac.gearys_sparse(
                adj_matrix_permuted.indptr,
                adj_matrix_permuted.indices,
                adj_matrix_permuted.data,
                data_row_ptr=data.indptr,
                data_col_ind=data.indices,
                data_values=data.data,
                n_samples=n_samples,
                n_features=n_features,
                num=num_permuted,
                stream=cp.cuda.get_current_stream().ptr,
            )
            gearys_C_permutations[p, :] = (n_samples - 1) * num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        gearys_C_permutations = None
    return gearys_C, gearys_C_permutations


def _gearys_C_cupy(data, adj_matrix_cupy, n_permutations=100):
    if sparse.isspmatrix_csr(data):
        return _gearys_C_cupy_sparse(data, adj_matrix_cupy, n_permutations)
    elif isinstance(data, cp.ndarray):
        return _gearys_C_cupy_dense(data, adj_matrix_cupy, n_permutations)
    else:
        raise ValueError("Datatype not supported")
