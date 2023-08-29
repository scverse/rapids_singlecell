from typing import Optional, Union

import cupy as cp
from anndata import AnnData
from scanpy._utils import view_to_actual
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell.cunnData import cunnData

from ._utils import _check_gpu_X, _get_mean_var


def scale(
    adata: Union[AnnData, cunnData],
    max_value: Optional[int] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[cp.ndarray]:
    """
    Scales matrix to unit variance and clips values

    Parameters
    ----------
        adata
            AnnData/ cunnData object

        max_value
            After scaling matrix to unit variance, values will be clipped to this number of std deviations.

        layer
            Layer to use as input instead of X. If None, X is used.

        inplace
            If True, update cunnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    -------
    Returns a sacled copy or updates `adata` with a scaled version of the original `adata.X` and `adata.layers['layer']`, \
    depending on `inplace`.

    """
    if isinstance(adata, AnnData):
        view_to_actual(adata)

    X = _get_obs_rep(adata, layer=layer)
    _check_gpu_X(X)

    if not isinstance(X, cp.ndarray):
        print("densifying _.X")
        X = X.toarray()

    if not inplace:
        X = X.copy()

    if not X.flags.c_contiguous:
        X = cp.asarray(X, order="C")

    mean, var = _get_mean_var(X)
    std = cp.sqrt(var)
    std[std == 0] = 1
    X -= mean
    X /= std

    if max_value:
        X = cp.clip(X, a_min=-max_value, a_max=max_value)
    if inplace:
        _set_obs_rep(adata, X, layer=layer)
    else:
        return X
