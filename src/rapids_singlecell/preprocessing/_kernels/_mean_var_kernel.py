from __future__ import annotations

import cupy as cp
from cuml.common.kernel_utils import cuda_kernel_factory

_get_mean_var_major_kernel = r"""
        (const int *indptr,const int *index,const {0} *data,
            double* means,double* vars,
            int major, int minor) {
        int major_idx = blockIdx.x;
        if(major_idx >= major){
            return;
        }
        int start_idx = indptr[major_idx];
        int stop_idx = indptr[major_idx+1];

        __shared__ double mean_place[64];
        __shared__ double var_place[64];

        mean_place[threadIdx.x] = 0.0;
        var_place[threadIdx.x] = 0.0;
        __syncthreads();

        for(int minor_idx = start_idx+threadIdx.x; minor_idx < stop_idx; minor_idx+= blockDim.x){
               double value = (double)data[minor_idx];
               mean_place[threadIdx.x] += value;
               var_place[threadIdx.x] += value*value;
        }
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                mean_place[threadIdx.x] += mean_place[threadIdx.x + s];
                var_place[threadIdx.x] += var_place[threadIdx.x + s];
            }
            __syncthreads(); // Synchronize at each step of the reduction
        }
        if (threadIdx.x == 0) {
            means[major_idx] = mean_place[threadIdx.x];
            vars[major_idx] = var_place[threadIdx.x];
        }

        }
"""

_get_mean_var_minor_kernel = r"""
        (const int *index,const {0} *data,
            double* means, double* vars,
            int major, int nnz) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx >= nnz){
            return;
        }
       double value = (double) data[idx];
       int minor_pos = index[idx];
       atomicAdd(&means[minor_pos], value/major);
       atomicAdd(&vars[minor_pos], value*value/major);
        }
    """


sq_sum = cp.ReductionKernel(
    "T x",  # input params
    "float64 y",  # output params
    "x * x",  # map
    "a + b",  # reduce
    "y = a",  # post-reduction map
    "0",  # identity value
    "sqsum64",  # kernel name
)

mean_sum = cp.ReductionKernel(
    "T x",  # input params
    "float64 y",  # output params
    "x",  # map
    "a + b",  # reduce
    "y = a",  # post-reduction map
    "0",  # identity value
    "sum64",  # kernel name
)


def _get_mean_var_major(dtype):
    return cuda_kernel_factory(
        _get_mean_var_major_kernel, (dtype,), "_get_mean_var_major_kernel"
    )


def _get_mean_var_minor(dtype):
    return cuda_kernel_factory(
        _get_mean_var_minor_kernel, (dtype,), "_get_mean_var_minor_kernel"
    )
