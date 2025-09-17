from __future__ import annotations

import cupy as cp
from cupyx.scipy import sparse

try:
    from rapids_singlecell._cuda import _autocorr_cuda as _ac
except ImportError:
    _ac = None


def _morans_I_cupy_dense(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    data_centered_cupy = data - data.mean(axis=0)

    # Calculate the numerator and denominator for Moran's I
    num = cp.zeros(n_features, dtype=cp.float32)

    _ac.morans_dense(
        data_centered_cupy.data.ptr,
        adj_matrix_cupy.indptr.data.ptr,
        adj_matrix_cupy.indices.data.ptr,
        adj_matrix_cupy.data.data.ptr,
        num.data.ptr,
        int(n_samples),
        int(n_features),
        int(cp.cuda.get_current_stream().ptr),
    )

    # Calculate the denominator for Moarn's I
    den = cp.sum(data_centered_cupy**2, axis=0)

    # Calculate Moarn's I
    morans_I = num / den
    # Calculate p-values using permutation tests
    if n_permutations:
        morans_I_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        num_permuted = cp.zeros(n_features, dtype=cp.float32)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            _ac.morans_dense(
                data_centered_cupy.data.ptr,
                adj_matrix_permuted.indptr.data.ptr,
                adj_matrix_permuted.indices.data.ptr,
                adj_matrix_permuted.data.data.ptr,
                num_permuted.data.ptr,
                int(n_samples),
                int(n_features),
                int(cp.cuda.get_current_stream().ptr),
            )
            morans_I_permutations[p, :] = num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _morans_I_cupy_sparse(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    # Calculate the numerator for Moarn's I
    num = cp.zeros(n_features, dtype=cp.float32)
    means = data.mean(axis=0).ravel()

    n_samples, n_features = data.shape
    # Launch the kernel
    _ac.morans_sparse(
        adj_matrix_cupy.indptr.data.ptr,
        adj_matrix_cupy.indices.data.ptr,
        adj_matrix_cupy.data.data.ptr,
        data.indptr.data.ptr,
        data.indices.data.ptr,
        data.data.data.ptr,
        int(n_samples),
        int(n_features),
        means.data.ptr,
        num.data.ptr,
        int(cp.cuda.get_current_stream().ptr),
    )

    # Calculate the denominator for Moarn's I
    den = cp.zeros(n_features, dtype=cp.float32)
    counter = cp.zeros(n_features, dtype=cp.int32)
    _ac.pre_den_sparse(
        data.indices.data.ptr,
        data.data.data.ptr,
        int(data.nnz),
        means.data.ptr,
        den.data.ptr,
        counter.data.ptr,
        int(cp.cuda.get_current_stream().ptr),
    )
    counter = n_samples - counter
    den += counter * means**2

    # Calculate Moarn's I
    morans_I = num / den

    if n_permutations:
        morans_I_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        num_permuted = cp.zeros(n_features, dtype=cp.float32)

        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_permuted = cp.zeros(n_features, dtype=data.dtype)
            _ac.morans_sparse(
                adj_matrix_permuted.indptr.data.ptr,
                adj_matrix_permuted.indices.data.ptr,
                adj_matrix_permuted.data.data.ptr,
                data.indptr.data.ptr,
                data.indices.data.ptr,
                data.data.data.ptr,
                int(n_samples),
                int(n_features),
                means.data.ptr,
                num_permuted.data.ptr,
                int(cp.cuda.get_current_stream().ptr),
            )

            morans_I_permutations[p, :] = num_permuted / den
            num_permuted[:] = 0
            cp.cuda.Stream.null.synchronize()
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _morans_I_cupy(data, adj_matrix_cupy, n_permutations=100):
    if sparse.isspmatrix_csr(data):
        return _morans_I_cupy_sparse(data, adj_matrix_cupy, n_permutations)
    elif isinstance(data, cp.ndarray):
        return _morans_I_cupy_dense(data, adj_matrix_cupy, n_permutations)
    else:
        raise ValueError("Datatype not supported")
