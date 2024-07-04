from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_sparse2dense = r"""
(const int* indptr,const int *index,const {0} *data,
            {0}* out, int major, int minor, int c_switch) {
        int row = blockIdx.x*blockDim.x+threadIdx.x ;
        int col = blockIdx.y*blockDim.y+threadIdx.y ;
        if(row >= major){
            return;
        }
        int start = indptr[row];
        int stop = indptr[row+1];
        if (col>= (stop - start)){
            return;
        }
        int idx = index[start + col];
        long long int res_index ;
        if (c_switch== 1){
            res_index = static_cast<long long int>(row)*minor+idx;
        }
        else{
            res_index = static_cast<long long int>(row)+idx*major;
        }
        atomicAdd(&out[res_index], data[start + col]);
}
"""


def _sparse2densekernel(dtype):
    return cuda_kernel_factory(_sparse2dense, (dtype,), "_sparse2dense")
