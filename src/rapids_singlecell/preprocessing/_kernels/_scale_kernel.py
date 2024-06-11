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

_csr_scale_diff_kernel = r"""
(const int *indptr, const int *indices, {0} *data, const double * std, const int *mask, {0} clipper,int nrows) {
        int row = blockIdx.x;

        if(row >= nrows){
            return;
        }
        if(mask[row]){
            int start_idx = indptr[row];
            int stop_idx = indptr[row+1];
            for(int i = start_idx+ threadIdx.x; i < stop_idx; i+=blockDim.x){
                int idx = indices[i];
                {0} res = data[i]/std[idx];
                data[i] = min(clipper,res);
        }
    }
}
"""

_dense_scale_center_diff_kernel = r"""
({0} *data, const {0}  *mean, const {0}  *std, const int *mask, {0} clipper,long long int nrows,long long int ncols)
{
    long long int row = blockIdx.x * blockDim.x + threadIdx.x;
    long long int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < nrows && col < ncols) {
        if (mask[row]){
            {0} res = data[row * ncols + col] - mean[col];
            res /= std[col];
            data[row * ncols + col] = max(-clipper,min(clipper,res));
        }
    }
}
"""

_dense_scale_diff_kernel = r"""
({0} *data, const {0} *std,const int *mask,const {0} clipper,long long int nrows,long long int ncols){
    long long int row = blockIdx.x * blockDim.x + threadIdx.x;
    long long int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < nrows && col < ncols) {
        if (mask[row]){
            {0} res = data[row * ncols + col] / std[col];
            data[row * ncols + col] = min(clipper,res);
        }
    }
}
"""


def _csc_scale_diff(dtype):
    return cuda_kernel_factory(
        _csc_scale_diff_kernel, (dtype,), "_csc_scale_diff_kernel"
    )


def _csr_scale_kernel(dtype):
    return cuda_kernel_factory(
        _csr_scale_diff_kernel, (dtype,), "_csr_scale_diff_kernel"
    )


def _dense_center_scale_kernel(dtype):
    return cuda_kernel_factory(
        _dense_scale_center_diff_kernel, (dtype,), "_dense_scale_center_diff_kernel"
    )


def _dense_scale_kernel(dtype):
    return cuda_kernel_factory(
        _dense_scale_diff_kernel, (dtype,), "_dense_scale_diff_kernel"
    )
