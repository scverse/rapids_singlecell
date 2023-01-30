import cupy as cp
import cupyx as cpx
from cuml.linear_model import LinearRegression
from rapids_singlecell.cunnData import cunnData
from typing import Literal, Union,Optional
from ..cunnData import cunnData
import math

def regress_out(cudata:cunnData,
                keys,
                layer: Optional[str] = None,
                inplace = True,
                batchsize: Union[int,Literal["all"],None] = 100,
                verbose=False):

    """
    Use linear regression to adjust for the effects of unwanted noise
    and variation. 
    Parameters
    ----------
    cudata: cunnData object

    keys
        Keys for numerical observation annotation on which to regress on.
    
    layer
        Layer to regress instead of `X`. If `None`, `X` is regressed.

    inplace: bool
        Whether to update `cudata` or return the corrected matrix of
        `cudata.X` and `cudata.layers`.

    batchsize: Union[int,Literal["all"],None] (default: 100)
        Number of genes that should be processed together. 
        If `'all'` all genes will be processed together if `.n_obs` <100000. 
        If `None` each gene will be analysed seperatly.
        Will be ignored if cuML version < 22.12
        
    verbose : bool
        Print debugging information
    Returns
    -------
    Returns a corrected copy or  updates `cudata` with a corrected version of the 
    original `cudata.X` and `cudata.layers['layer']`, depending on `inplace`.
    """
    
    if batchsize != "all" and type(batchsize) not in [int, type(None)]:
        raise ValueError("batchsize must be `int`, `None` or `'all'`")
    
    X= cudata.layers[layer] if layer is not None else cudata.X

    if cpx.scipy.sparse.issparse(X) and not cpx.scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    dim_regressor= 2
    if type(keys)is list:
        dim_regressor = len(keys)+1

    regressors = cp.ones((X.shape[0]*dim_regressor)).reshape((X.shape[0], dim_regressor), order="F")
    if dim_regressor==2:
        regressors[:, 1] = cp.array(cudata.obs[keys]).ravel()
    else:
        for i in range(dim_regressor-1):
            regressors[:, i+1] = cp.array(cudata.obs[keys[i]]).ravel()

    outputs = cp.empty(X.shape, dtype=X.dtype, order="F")

    cuml_supports_multi_target = LinearRegression._get_tags()['multioutput']

    if cuml_supports_multi_target and batchsize:
        if batchsize == "all" and X.shape[0] < 100000:
            if cpx.scipy.sparse.issparse(X): 
                X = X.todense()
            lr = LinearRegression(fit_intercept=False, output_type="cupy", algorithm='svd')
            lr.fit(regressors, X, convert_dtype=True)
            outputs[:] = X - lr.predict(regressors)
        else:
            if batchsize == "all":
                batchsize = 100
            n_batches = math.ceil(X.shape[1] / batchsize)
            for batch in range(n_batches):
                start_idx = batch * batchsize
                stop_idx = min(batch * batchsize + batchsize, X.shape[1])
                if cpx.scipy.sparse.issparse(X):
                    arr_batch = X[:,start_idx:stop_idx].todense()
                else:
                    arr_batch = X[:,start_idx:stop_idx].copy()
                lr = LinearRegression(fit_intercept=False, output_type="cupy", algorithm='svd')
                lr.fit(regressors, arr_batch, convert_dtype=True)
                outputs[:,start_idx:stop_idx] =arr_batch - lr.predict(regressors)
    else:
        if X.shape[0] < 100000 and cpx.scipy.sparse.issparse(X):
            X = X.todense()
        for i in range(X.shape[1]):
            if verbose and i % 500 == 0:
                print("Regressed %s out of %s" %(i, X.shape[1]))
            
            y = X[:,i]
            outputs[:, i] = _regress_out_chunk(regressors, y)
    
    if inplace:
        if layer:
            cudata.layers[layer] = outputs
        else:
            cudata.X = outputs
    else:
        return outputs


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
