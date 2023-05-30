import cupy as cp
import cupyx as cpx
import math

kernel_morans_I_num_dense = r"""
extern "C" __global__
void morans_I_num_dense(const float* data_centered, const int* adj_matrix_row_ptr, const int* adj_matrix_col_ind,
const float* adj_matrix_data, float* num, int n_samples, int n_features) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_samples || f >= n_features) {
        return;
    }

    int k_start = adj_matrix_row_ptr[i];
    int k_end = adj_matrix_row_ptr[i + 1];

    for (int k = k_start; k < k_end; ++k) {
        int j = adj_matrix_col_ind[k];
        float edge_weight = (adj_matrix_data[k]);
        float product = data_centered[i * n_features + f] * data_centered[j * n_features + f];
        atomicAdd(&num[f], edge_weight * product);
    }
}
"""


def _morans_I_cupy_dense(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    data_centered_cupy = data - data.mean(axis=0)

    # Calculate the numerator and denominator for Moran's I
    num = cp.zeros(n_features, dtype=cp.float32)
    block_size = 8
    fg = int(math.ceil(n_features / block_size))
    sg = int(math.ceil(n_samples / block_size))
    grid_size = (fg, sg, 1)

    num_kernel = cp.RawKernel(kernel_morans_I_num_dense, "morans_I_num_dense")
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
    den = cp.sum(data_centered_cupy**2, axis=0)
    morans_I = num / den
    # Calculate p-values using permutation tests
    if n_permutations:
        morans_I_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_permuted = cp.zeros(n_features, dtype=cp.float32)
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
            cp.cuda.Stream.null.synchronize()
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _morans_I_cupy_sparse(data, adj_matrix_cupy, n_permutations=100):
    n_samples, n_features = data.shape
    data_mean = data.mean(axis=0).ravel()

    # Calculate den
    den = cp.sum((data - data_mean) ** 2, axis=0)

    # Calculate the numerator and denominator for Moran's I
    data = data.tocsc()
    num = cp.zeros(n_features, dtype=data.dtype)
    block_size = 8

    sg = int(math.ceil(n_samples / block_size))

    num_kernel = cp.RawKernel(kernel_morans_I_num_dense, "morans_I_num_dense")
    batchsize = 1000
    n_batches = math.ceil(n_features / batchsize)
    for batch in range(n_batches):
        start_idx = batch * batchsize
        stop_idx = min(batch * batchsize + batchsize, n_features)
        data_centered_cupy = data[:, start_idx:stop_idx].toarray()
        data_centered_cupy = data_centered_cupy - data_mean[start_idx:stop_idx]
        num_block = cp.zeros(data_centered_cupy.shape[1], dtype=data.dtype)
        fg = int(math.ceil(data_centered_cupy.shape[1] / block_size))
        grid_size = (fg, sg, 1)

        num_kernel(
            grid_size,
            (block_size, block_size, 1),
            (
                data_centered_cupy,
                adj_matrix_cupy.indptr,
                adj_matrix_cupy.indices,
                adj_matrix_cupy.data,
                num_block,
                n_samples,
                data_centered_cupy.shape[1],
            ),
        )
        num[start_idx:stop_idx] = num_block
        cp.cuda.Stream.null.synchronize()

    morans_I = num / den
    # Calculate p-values using permutation tests
    if n_permutations:
        morans_I_permutations = cp.zeros((n_permutations, n_features), dtype=cp.float32)
        for p in range(n_permutations):
            idx_shuffle = cp.random.permutation(adj_matrix_cupy.shape[0])
            adj_matrix_permuted = adj_matrix_cupy[idx_shuffle, :]
            num_permuted = cp.zeros(n_features, dtype=data.dtype)
            for batch in range(n_batches):
                start_idx = batch * batchsize
                stop_idx = min(batch * batchsize + batchsize, n_features)
                data_centered_cupy = data[:, start_idx:stop_idx].toarray()
                data_centered_cupy = data_centered_cupy - data_mean[start_idx:stop_idx]
                num_block = cp.zeros(data_centered_cupy.shape[1], dtype=data.dtype)
                fg = int(math.ceil(data_centered_cupy.shape[1] / block_size))
                grid_size = (fg, sg, 1)
                num_kernel(
                    grid_size,
                    (block_size, block_size, 1),
                    (
                        data_centered_cupy,
                        adj_matrix_permuted.indptr,
                        adj_matrix_permuted.indices,
                        adj_matrix_permuted.data,
                        num_block,
                        n_samples,
                        data_centered_cupy.shape[1],
                    ),
                )
                num_permuted[start_idx:stop_idx] = num_block
                cp.cuda.Stream.null.synchronize()
            morans_I_permutations[p, :] = num_permuted / den
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _morans_I_cupy(data, adj_matrix_cupy, n_permutations=100):
    if cpx.scipy.sparse.isspmatrix_csr(data):
        return _morans_I_cupy_sparse(data, adj_matrix_cupy, n_permutations)
    elif isinstance(data, cp.ndarray):
        return _morans_I_cupy_dense(data, adj_matrix_cupy, n_permutations)
    else:
        raise ValueError("Datatype not supported")
