import cupy as cp
import cupyx as cpx
import math
import warnings
from typing import Optional

from ..cunnData import cunnData
from ._utils import _check_nonnegative_integers

def normalize_total(cudata: cunnData, 
                    target_sum:int,
                    layer: Optional[str] = None,
                    inplace = True):
    """
    Normalizes rows in matrix so they sum to `target_sum`

    Parameters
    ----------
    cudata: cunnData object

    target_sum : int
        Each row will be normalized to sum to this value
    
    layer
        Layer to normalize instead of `X`. If `None`, `X` is normalized.

    inplace: bool
        Whether to update `cudata` or return the normalized matrix.
    
    
    Returns
    -------
    Returns a normalized copy or  updates `cudata` with a normalized version of
    the original `cudata.X` and `cudata.layers['layer']`, depending on `inplace`.
    
    """
    csr_arr = cudata.layers[layer] if layer is not None else cudata.X

    if not inplace:
        csr_arr = csr_arr.copy()

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

    if inplace:
        if layer:
            cudata.layers[layer] = csr_arr
        else:
            cudata.X = csr_arr
    else:
        return csr_arr

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
    """
    Applies analytic Pearson residual normalization, based on Lause21.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to `sqrt(n_obs)`
    and overdispersion `theta=100` is used.

    Parameters
    ----------
    cudata:
        cunnData object
    theta : float (default: 100)
        The negative binomial overdispersion parameter theta for Pearson residuals. 
        Higher values correspond to less overdispersion (var = mean + mean^2/theta), and theta=np.Inf corresponds to a Poisson model.
    clip : Optional[float] (default: None)
        Determines if and how residuals are clipped:
        If None, residuals are clipped to the interval [-sqrt(n_obs), sqrt(n_obs)], where n_obs is the number of cells in the dataset (default behavior).
        If any scalar c, residuals are clipped to the interval [-c, c]. Set clip=np.Inf for no clipping.
    check_values : bool (default: True)
        If True, checks if counts in selected layer are integers as expected by this function, 
        and return a warning if non-integers are found. Otherwise, proceed without checking. Setting this to False can speed up code for large datasets.
    layer : Optional[str] (default: None)
        Layer to use as input instead of X. If None, X is used.
    inplace : bool (default: True)
        If True, update cunnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    ----------
    If `inplace=True`, `cudata.X` or the selected layer in `cudata.layers` is updated with the normalized values.
    If `inplace=False` the normalized matrix is returned.

    """
    
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
