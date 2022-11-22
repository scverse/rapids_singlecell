import cupy as cp
from ..cunnData import cunnData

def scale(cudata:cunnData, max_value=10):
    """
    Scales matrix to unit variance and clips values
    Parameters
    ----------
    max_value : int
                After scaling matrix to unit variance,
                values will be clipped to this number
                of std deviations.
    Return
    ------
    updates cunndata object with a scaled cunndata.X
    """
    if type(cudata.X) is not cp._core.core.ndarray:
        print("densifying _.X")
        X = cudata.X.toarray()
    else:
        X =cudata.X
    mean = X.mean(axis=0)
    X -= mean
    del mean
    stddev = cp.sqrt(X.var(axis=0))
    X /= stddev
    del stddev
    cudata.X = cp.clip(X,a_max=max_value)