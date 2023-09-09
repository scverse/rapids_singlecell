from cuml.common.kernel_utils import cuda_kernel_factory

_get_mean_var_major_kernel = r"""
        (const int *indptr,const int *index,const {0} *data,
            double* means,double* vars,
            int major, int minor) {
        int major_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(major_idx >= major){
            return;
        }
        int start_idx = indptr[major_idx];
        int stop_idx = indptr[major_idx+1];

        for(int minor_idx = start_idx; minor_idx < stop_idx; minor_idx++){
               double value = (double)data[minor_idx];
               means[major_idx]+= value;
               vars[major_idx]+= value*value;
        }
        means[major_idx]/=minor;
        vars[major_idx]/=minor;
        vars[major_idx]-=(means[major_idx]*means[major_idx]);
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


def _get_mean_var_major(dtype):
    return cuda_kernel_factory(
        _get_mean_var_major_kernel, (dtype,), "_get_mean_var_major_kernel"
    )


def _get_mean_var_minor(dtype):
    return cuda_kernel_factory(
        _get_mean_var_minor_kernel, (dtype,), "_get_mean_var_minor_kernel"
    )
