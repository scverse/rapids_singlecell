import cupy as cp
import cupyx as cpx
from cuml.linear_model import LinearRegression
from ..cunnData import cunnData

def regress_out(cudata:cunnData, keys, verbose=False):

    """
    Use linear regression to adjust for the effects of unwanted noise
    and variation. 
    Parameters
    ----------

    adata
        The annotated data matrix.
    keys
        Keys for numerical observation annotation on which to regress on.

    verbose : bool
        Print debugging information

    Returns
    -------
    updates cunndata object with the corrected data matrix


    """
    
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

    if cudata.X.shape[0] < 100000 and cpx.scipy.sparse.issparse(cudata.X):
        cudata.X = cudata.X.todense()
    
    for i in range(cudata.X.shape[1]):
        if verbose and i % 500 == 0:
            print("Regressed %s out of %s" %(i, cudata.X.shape[1]))
        X = regressors
        y = cudata.X[:,i]
        outputs[:, i] = _regress_out_chunk(X, y)
    cudata.X = outputs

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