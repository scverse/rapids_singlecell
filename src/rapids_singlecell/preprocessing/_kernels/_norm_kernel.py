from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_mul_kernel_csr = r"""
(const int *indptr, {0} *data,
                    int nrows, int tsum) {
        int row = blockDim.x * blockIdx.x + threadIdx.x;

        if(row >= nrows)
            return;

        {0} scale = 0.0;
        int start_idx = indptr[row];
        int stop_idx = indptr[row+1];

        for(int i = start_idx; i < stop_idx; i++)
            scale += data[i];

        if(scale > 0.0) {
            scale = tsum / scale;
            for(int i = start_idx; i < stop_idx; i++)
                data[i] *= scale;
        }
    }
"""

_mul_kernel_dense = r"""
({0} *data, int nrows, int ncols, int tsum) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= nrows)
        return;

    {0} scale = 0.0;
    for(int i = 0; i < ncols; i++)
        scale += data[row * ncols + i];

    if(scale > 0.0) {
        scale = tsum / scale;
        for(int i = 0; i < ncols; i++)
            data[row * ncols + i] *= scale;
    }
}
"""

_get_sparse_sum_major_kernel = r"""
        (const int *indptr,const {0} *data,
            {0}* sums, int major) {
        int major_idx = blockIdx.x;
        if(major_idx >= major){
            return;
        }
        int start_idx = indptr[major_idx];
        int stop_idx = indptr[major_idx+1];

        __shared__ {0} sum_place[64];

        sum_place[threadIdx.x] = 0.0;
        __syncthreads();

        for(int minor_idx = start_idx+threadIdx.x; minor_idx < stop_idx; minor_idx+= blockDim.x){
               sum_place[threadIdx.x] += data[minor_idx];
        }
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sum_place[threadIdx.x] += sum_place[threadIdx.x + s];
            }
            __syncthreads(); // Synchronize at each step of the reduction
        }
        if (threadIdx.x == 0) {
            sums[major_idx] = sum_place[threadIdx.x];
        }

        }
"""


def _mul_csr(dtype):
    return cuda_kernel_factory(_mul_kernel_csr, (dtype,), "_mul_kernel_csr")


def _mul_dense(dtype):
    return cuda_kernel_factory(_mul_kernel_dense, (dtype,), "_mul_kernel_dense")


def _get_sparse_sum_major(dtype):
    return cuda_kernel_factory(
        _get_sparse_sum_major_kernel, (dtype,), "_get_sparse_sum_major_kernel"
    )
