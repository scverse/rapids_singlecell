from cuml.common.kernel_utils import cuda_kernel_factory

_csc_scale_diff_kernel = r"""
(const int *indptr, {0} *data, const double * std, int ncols) {
        int col = blockIdx.x;

        if(col >= ncols){
            return;
        }
        int start_idx = indptr[col];
        int stop_idx = indptr[col+1];
        double diver = 1/std[col];
        for(int i = start_idx+ threadIdx.x; i < stop_idx; i+=blockDim.x){
            data[i] *= diver;
        }

    }
"""


def _csc_scale_diff(dtype):
    return cuda_kernel_factory(
        _csc_scale_diff_kernel, (dtype,), "_csc_scale_diff_kernel"
    )
