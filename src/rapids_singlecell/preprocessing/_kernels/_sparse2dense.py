from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_sparse2dense = r"""
(const int* indptr,const int *index,const {0} *data,
            {0}* out, long long int major, long long int minor, int c_switch) {
    long long int row = blockIdx.x * blockDim.x + threadIdx.x;
    long long int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= major) {
        return;
    }
    long long int start = (long long int)indptr[row];
    long long int stop = (long long int)indptr[row + 1];
    if (col >= (stop - start)) {
        return;
    }
    long long int idx = (long long int)index[start + col];
    if (idx >= minor) {
        return;
    }
    long long int res_index;
    if (c_switch == 1) {
        res_index = row * minor + idx;
    } else {
        res_index = row + idx * major;
    }

    atomicAdd(&out[res_index], data[start + col]);
}
"""


def _sparse2densekernel(dtype):
    return cuda_kernel_factory(_sparse2dense, (dtype,), "_sparse2dense")
