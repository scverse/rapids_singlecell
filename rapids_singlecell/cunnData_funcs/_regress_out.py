import cupy as cp
import cupyx as cpx
from cuml.linear_model import LinearRegression
from rapids_singlecell.cunnData import cunnData
from typing import Literal, Union
from ..cunnData import cunnData
import math

def regress_out(cudata:cunnData,
                keys,
                batchsize: Union[int,Literal["all"],None] = 100,
                verbose=False):

    """
    Use linear regression to adjust for the effects of unwanted noise
    and variation. 
    Parameters
    ----------
    adata
        The annotated data matrix.
    keys
        Keys for numerical observation annotation on which to regress on.
    
    batchsize: Union[int,Literal["all"],None] (default: 100)
        Number of genes that should be processed together. 
        If `'all'` all genes will be processed together if `.n_obs` <100000. 
        If `None` each gene will be analysed seperatly.
        Will be ignored if cuML version < 22.12
        
    verbose : bool
        Print debugging information
    Returns
    -------
    updates cunndata object with the corrected data matrix
    """
    
    if batchsize != "all" and type(batchsize) not in [int, type(None)]:
        raise ValueError("batchsize must be `int`, `None` or `'all'`")
    
    if cpx.scipy.sparse.issparse(cudata.X) and not cpx.scipy.sparse.isspmatrix_csc(cudata.X):
        cudata.X = cudata.X.tocsc()

    dim_regressor= 2
    if type(keys)is list:
        dim_regressor = len(keys)+1

    regressors = cp.ones((cudata.X.shape[0]*dim_regressor)).reshape((cudata.X.shape[0], dim_regressor), order="F")
    if dim_regressor==2:
        regressors[:, 1] = cp.array(cudata.obs[keys]).ravel()
    else:
        for i in range(dim_regressor-1):
            regressors[:, i+1] = cp.array(cudata.obs[keys[i]]).ravel()

    outputs = cp.empty(cudata.X.shape, dtype=cudata.X.dtype, order="F")

    cuml_supports_multi_target = LinearRegression._get_tags()['multioutput']

    if cuml_supports_multi_target and batchsize:
        if batchsize == "all" and cudata.X.shape[0] < 100000:
            if cpx.scipy.sparse.issparse(cudata.X): 
                cudata.X = cudata.X.todense()
            X = regressors
            lr = LinearRegression(fit_intercept=False, output_type="cupy", algorithm='svd')
            lr.fit(X, cudata.X, convert_dtype=True)
            outputs[:] = cudata.X - lr.predict(X)
        else:
            if batchsize == "all":
                batchsize = 100
            n_batches = math.ceil(cudata.X.shape[1] / batchsize)
            for batch in range(n_batches):
                start_idx = batch * batchsize
                stop_idx = min(batch * batchsize + batchsize, cudata.X.shape[1])
                if cpx.scipy.sparse.issparse(cudata.X):
                    arr_batch = cudata.X[:,start_idx:stop_idx].todense()
                else:
                    arr_batch = cudata.X[:,start_idx:stop_idx].copy()
                X = regressors
                lr = LinearRegression(fit_intercept=False, output_type="cupy", algorithm='svd')
                lr.fit(X, arr_batch, convert_dtype=True)
                outputs[:,start_idx:stop_idx] =arr_batch - lr.predict(X)
    else:
        if cudata.X.shape[0] < 100000 and cpx.scipy.sparse.issparse(cudata.X):
            cudata.X = cudata.X.todense()
        for i in range(cudata.X.shape[1]):
            if verbose and i % 500 == 0:
                print("Regressed %s out of %s" %(i, cudata.X.shape[1]))
            X = regressors
            y = cudata.X[:,i]
            outputs[:, i] = _regress_out_chunk(X, y)
            
    cudata.X= outputs

def _regress_out_chunk(X, y):
    """
    Performs a data_cunk.shape[1] number of local linear regressions,
    replacing the data in the original chunk w/ the regressed result.
    Parameters
    ----------
    X : cupy.ndarray of shape (n_cells, 3)
        Matrix of regressors
    y : cupy.sparse.spmatrix of shape (n_cells,)
        Sparse matrix containing a single column of the cellxgene matrix
    Returns
    -------
    dense_mat : cupy.ndarray of shape (n_cells,)
        Adjusted column
    """
    if cp.sparse.issparse(y):
        y = y.todense()

    lr = LinearRegression(fit_intercept=False, output_type="cupy")
    lr.fit(X, y, convert_dtype=True)
    return y.reshape(y.shape[0],) - lr.predict(X).reshape(y.shape[0])