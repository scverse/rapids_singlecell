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

_get_mean_var_minor_fast_kernel = r"""
(const long long nnz,
const int* __restrict__ indices,
const {0}* __restrict__ data,
double* __restrict__ g_sum,
double* __restrict__ g_sumsq)
{
    extern __shared__ unsigned shmem[];
    unsigned HASH_SIZE = 1024;
    // layout in shared:
    // keys[HASH_SIZE] (uint32, 0xFFFFFFFF = empty)
    // sum[HASH_SIZE]  (double)
    // sq[HASH_SIZE]   (double)
    unsigned* keys = shmem;
    double*   sum  = (double*)(keys + HASH_SIZE);
    double*   sq   = (double*)(sum + HASH_SIZE);

    // init table
    for (int i = threadIdx.x; i < HASH_SIZE; i += blockDim.x) {
        keys[i] = 0xFFFFFFFFu;
        sum[i]  = 0.0;
        sq[i]   = 0.0;
    }
    __syncthreads();

    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nnz; i += stride)
    {
        unsigned col = (unsigned)__ldg(indices + i);
        double dv = (double)__ldg(data + i);
        double d2 = dv * dv;

        unsigned h = (col * 2654435761u) & (HASH_SIZE - 1);
        bool done = false;

        #pragma unroll 8
        for (int probe = 0; probe < 8; ++probe) {
            unsigned pos = (h + probe) & (HASH_SIZE - 1);
            unsigned key = atomicCAS(&keys[pos], 0xFFFFFFFFu, col);
            if (key == 0xFFFFFFFFu || key == col) {
                atomicAdd(&sum[pos], dv);
                atomicAdd(&sq[pos],  d2);
                done = true;
                break;
            }
        }
        if (!done) {
            atomicAdd(&g_sum[col],   dv);
            atomicAdd(&g_sumsq[col], d2);
        }
    }
    __syncthreads();

    // flush
    for (int i = threadIdx.x; i < HASH_SIZE; i += blockDim.x) {
        unsigned key = keys[i];
        if (key != 0xFFFFFFFFu) {
            atomicAdd(&g_sum[key],   sum[i]);
            atomicAdd(&g_sumsq[key], sq[i]);
        }
    }
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


def _get_mean_var_minor_fast(dtype):
    return cuda_kernel_factory(
        _get_mean_var_minor_fast_kernel, (dtype,), "_get_mean_var_minor_fast_kernel"
    )
