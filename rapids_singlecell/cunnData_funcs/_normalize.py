import cupy as cp
import cupyx as cpx
import numpy as np
import math
import warnings
from typing import Optional

from ..cunnData import cunnData
from ._utils import _check_nonnegative_integers

def normalize_total(cudata: cunnData, target_sum):
    """
    Normalizes rows in matrix so they sum to `target_sum`

    Parameters
    ----------

    target_sum : int
        Each row will be normalized to sum to this value
    
    
    Returns
    -------
    
    a normalized sparse Matrix to a specified target sum
    
    """
    csr_arr = cudata.X
    mul_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void mul_kernel(const int *indptr, float *data, 
                        int nrows, int tsum) {
            int row = blockDim.x * blockIdx.x + threadIdx.x;

            if(row >= nrows)
                return;

            float scale = 0.0;
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
        ''', 'mul_kernel')

    mul_kernel((math.ceil(csr_arr.shape[0] / 32.0),), (32,),
                    (csr_arr.indptr,
                    csr_arr.data,
                    csr_arr.shape[0],
                    int(target_sum)))

    cudata.X = csr_arr

def log1p(cudata: cunnData):
    """
    Calculated the natural logarithm of one plus the sparse marttix, element-wise inlpace in cunnData object.
    """
    cudata.X = cudata.X.log1p()
    cudata.uns["log1p"] = {"base": None}

def normalize_pearson_residuals(cudata: cunnData,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    layer: Optional[str] = None,
    inplace = True):

    X = cudata.layers[layer] if layer is not None else cudata.X
    X = X.copy()
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
    )
    if theta <= 0:
        raise ValueError('Pearson residuals require theta > 0')
    if clip is None:
        n = X.shape[0]
        clip = cp.sqrt(n)
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")

    if cpx.scipy.sparse.isspmatrix_csc(X):
        sums_cells = X.sum(axis=1)
        X =X.tocsr()
        sums_genes = X.sum(axis=0)
    elif cpx.scipy.sparse.isspmatrix_csr(X):
        sums_genes = X.sum(axis=0)
        X =X.tocsc()
        sums_cells = X.sum(axis=1)
    
    sum_total = sums_genes.sum().squeeze()
    mu = sums_cells @ sums_genes / sum_total
    X = X - mu
    X = X / cp.sqrt( mu + mu**2 / theta)
    X = cp.clip(X, a_min=-clip, a_max=clip)
    X = cp.array(X, dtype= cp.float32)
    if inplace == True:
        if layer:
            cudata.layers[layer]= X
        else:
            cudata.X= X
    else:
        return X