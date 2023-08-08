from cuml.common.kernel_utils import cuda_kernel_factory

_mul_kernel = r"""
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


def _mul_kernel_csr(dtype):
    return cuda_kernel_factory(_mul_kernel, (dtype,), "_mul_kernel")
