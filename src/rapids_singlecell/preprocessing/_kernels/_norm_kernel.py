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


def _mul_csr(dtype):
    return cuda_kernel_factory(_mul_kernel_csr, (dtype,), "_mul_kernel_csr")


def _mul_dense(dtype):
    return cuda_kernel_factory(_mul_kernel_dense, (dtype,), "_mul_kernel_dense")
