from typing import Optional

import cupy as cp
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell.cunnData import cunnData

from ._utils import _check_gpu_X


def scale(
    cudata: cunnData,
    max_value: Optional[int] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[cp.ndarray]:
    """
    Scales matrix to unit variance and clips values

    Parameters
    ----------
        cudata
            cunnData object

        max_value
            After scaling matrix to unit variance, values will be clipped to this number of std deviations.

        layer
            Layer to use as input instead of X. If None, X is used.

        inplace
            If True, update cunnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    -------
    Returns a sacled copy or updates `cudata` with a scaled version of the original `cudata.X` and `cudata.layers['layer']`, \
    depending on `inplace`.

    """
    X = _get_obs_rep(cudata, layer=layer)
    _check_gpu_X(X)

    if not isinstance(X, cp.ndarray):
        print("densifying _.X")
        X = X.toarray()

    if not inplace:
        X = X.copy()

    mean = X.sum(axis=0).flatten() / X.shape[0]
    X -= mean
    del mean
    stddev = cp.sqrt(X.var(axis=0))
    X /= stddev
    del stddev
    if max_value:
        X = cp.clip(X, a_min=-max_value, a_max=max_value)
    if inplace:
        _set_obs_rep(cudata, X, layer=layer)
    else:
        return X
