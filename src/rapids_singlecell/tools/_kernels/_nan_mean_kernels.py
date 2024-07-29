from __future__ import annotations

from cuml.common.kernel_utils import cuda_kernel_factory

_get_nan_mean_major_kernel = r"""
        (const int *indptr,const int *index,const {0} *data,
            double* means,int* nans, bool* mask,
            int major, int minor) {
        int major_idx = blockIdx.x;
        if(major_idx >= major){
            return;
        }
        int start_idx = indptr[major_idx];
        int stop_idx = indptr[major_idx+1];

        __shared__ double mean_place[64];
        __shared__ int nan_place[64];

        mean_place[threadIdx.x] = 0.0;
        nan_place[threadIdx.x] = 0;
        __syncthreads();

        for(int minor_idx = start_idx+threadIdx.x; minor_idx < stop_idx; minor_idx+= blockDim.x){
            int gene_number = index[minor_idx];
            if (mask[gene_number]==true){
                if(isnan(data[minor_idx])){
                    nan_place[threadIdx.x] += 1;
                }
                else{
                    double value = (double) data[minor_idx];
                    mean_place[threadIdx.x] += value;
                }
            }
        }
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                mean_place[threadIdx.x] += mean_place[threadIdx.x + s];
                nan_place[threadIdx.x] += nan_place[threadIdx.x + s];
            }
            __syncthreads(); // Synchronize at each step of the reduction
        }
        if (threadIdx.x == 0) {
            means[major_idx] = mean_place[threadIdx.x];
            nans[major_idx] = nan_place[threadIdx.x];
        }

        }
"""

_get_nan_mean_minor_kernel = r"""
        (const int *index,const {0} *data,
            double* means, int* nans, bool* mask, int nnz) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nnz) {
            return;
        }
        int minor_pos = index[idx];
        if (mask[minor_pos] == false) {
            return;
        }
        if(isnan(data[idx])){
            atomicAdd(&nans[minor_pos], 1);
            }
        else{
            double value = (double) data[idx];
            atomicAdd(&means[minor_pos], value);
        }
        }
    """


def _get_nan_mean_major(dtype):
    return cuda_kernel_factory(
        _get_nan_mean_major_kernel, (dtype,), "_get_nan_mean_major_kernel"
    )


def _get_nan_mean_minor(dtype):
    return cuda_kernel_factory(
        _get_nan_mean_minor_kernel, (dtype,), "_get_nan_mean_minor_kernel"
    )
