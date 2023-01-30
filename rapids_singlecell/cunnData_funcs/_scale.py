import cupy as cp
from ..cunnData import cunnData
from typing import Optional


def scale(cudata:cunnData, 
        max_value=10,
        layer: Optional[str] = None,
        inplace = True):
    """
    Scales matrix to unit variance and clips values
    Parameters
    ----------
    cudata:
        cunnData object

    max_value : int
        After scaling matrix to unit variance,
        values will be clipped to this number
        of std deviations.

    layer : Optional[str] (default: None)
        Layer to use as input instead of X. If None, X is used.

    inplace : bool (default: True)
        If True, update cunnData with results. Otherwise, return results. See below for details of what is returned.
    Return
    ------
    Returns a sacled copy or  updates `cudata` with a scaled version of the 
    original `cudata.X` and `cudata.layers['layer']`, depending on `inplace`.
    """
    X = cudata.layers[layer] if layer is not None else cudata.X

    if type(X) is not cp._core.core.ndarray:
        print("densifying _.X")
        X = X.toarray()
    else:
        X =X.copy()
    mean = X.sum(axis=0).flatten() / X.shape[0]
    X -= mean
    del mean
    stddev = cp.sqrt(X.var(axis=0))
    X /= stddev
    del stddev
    X= cp.clip(X,a_max=max_value)
    if inplace:
        if layer:
            cudata.layers[layer] = X
        else:
            cudata.X = X
    else:
        return X
