from __future__ import annotations

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

_csr_update_kernel = r"""
(const int *indptr, {0} *data, const int * sub_indptr, const {0} *sub_data, const int * subset_mask, int sub_nrows) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row >= sub_nrows){
        return;
    }
    int sub_start_idx = sub_indptr[row];
    int sub_stop_idx = sub_indptr[row+1];
    int sub_idx = subset_mask[row];

    int start_idx = indptr[sub_idx];
    int stop_idx = indptr[sub_idx+1];

    if(sub_stop_idx -sub_start_idx == stop_idx-start_idx){
        for(int i = 0; i < sub_stop_idx -sub_start_idx ; i++){
            data[start_idx+i] = sub_data[sub_start_idx+i];
        }
    }
}
"""


def _csc_scale_diff(dtype):
    return cuda_kernel_factory(
        _csc_scale_diff_kernel, (dtype,), "_csc_scale_diff_kernel"
    )


def _csr_update(dtype):
    return cuda_kernel_factory(_csr_update_kernel, (dtype,), "_csr_update_kernel")
