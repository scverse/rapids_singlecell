import cupy as cp
import cupyx as cpx
from cupyx.scipy.sparse import issparse
import math

_get_mean_var_major = cp.RawKernel(
    r"""
    extern "C" __global__
    void caluclate_meanvar_major(const int *indptr,const int *index,const float *data,
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
    """,
    "caluclate_meanvar_major",
)


_get_mean_var_minor = cp.RawKernel(
    r"""
    extern "C" __global__
    void caluclate_mean_minor(const int *index,const float *data,
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
    """,
    "caluclate_mean_minor",
)


def _mean_var_major(X, major, minor):
    mean = cp.zeros(major, dtype=cp.float64)
    var = cp.zeros(major, dtype=cp.float64)
    block = (32,)
    grid = (int(math.ceil(major / block[0])),)
    _get_mean_var_major(
        grid, block, (X.indptr, X.indices, X.data, mean, var, major, minor)
    )
    var *= minor / (minor - 1)
    return mean, var


def _mean_var_minor(X, major, minor):
    mean = cp.zeros(minor, dtype=cp.float64)
    var = cp.zeros(minor, dtype=cp.float64)
    block = (32,)
    grid = (int(math.ceil(X.nnz / block[0])),)
    _get_mean_var_minor(grid, block, (X.indices, X.data, mean, var, major, X.nnz))

    var = (var - mean**2) * (major / (major - 1))
    return mean, var


def _get_mean_var(X, axis=0):
    if axis == 0:
        if cpx.scipy.sparse.isspmatrix_csr(X):
            major = X.shape[0]
            minor = X.shape[1]
            mean, var = _mean_var_major(X, major, minor)
        elif cpx.scipy.sparse.isspmatrix_csc(X):
            major = X.shape[1]
            minor = X.shape[0]
            mean, var = _mean_var_minor(X, major, minor)
    elif axis == 1:
        if cpx.scipy.sparse.isspmatrix_csr(X):
            major = X.shape[0]
            minor = X.shape[1]
            mean, var = _mean_var_minor(X, major, minor)
        elif cpx.scipy.sparse.isspmatrix_csc(X):
            major = X.shape[1]
            minor = X.shape[0]
            mean, var = _mean_var_major(X, major, minor)
    return mean, var


def _check_nonnegative_integers(X):
    if issparse(X):
        data = X.data
    else:
        data = X
    """Checks values of data to ensure it is count data"""
    # Check no negatives
    if cp.signbit(data).any():
        return False
    elif cp.any(~cp.equal(cp.mod(data, 1), 0)):
        return False
    else:
        return True
