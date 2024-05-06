from __future__ import annotations

import math

import cupy as cp
import numpy as np
from anndata import AnnData
from cupyx.scipy import sparse
from scanpy._utils import view_to_actual

from rapids_singlecell.get import _check_mask, _get_obs_rep, _set_obs_rep
from rapids_singlecell.preprocessing._utils import _check_gpu_X, _get_mean_var


def scale(
    adata: AnnData,
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    copy: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    mask_obs: np.ndarray | str | None = None,
    inplace: bool = True,
) -> None | cp.ndarray:
    """
    Scales matrix to unit variance and clips values

    Parameters
    ----------
        adata
            AnnData object

        zero_center
            If `False`, omit zero-centering variables, which allows to handle sparse
            input efficiently.

        max_value
            Clip (truncate) to this value after scaling. If `None`, do not clip.

        copy
            Whether this function should be performed inplace. If an AnnData object
            is passed, this also determines if a copy is returned.

        layer
            If provided, which element of layers to scale.

        obsm
            If provided, which element of obsm to scale.

        mask_obs
            Restrict both the derivation of scaling parameters and the scaling itself
            to a certain set of observations. The mask is specified as a boolean array
            or a string referring to an array in :attr:`~anndata.AnnData.obs`. If the matrix is in csc format and a mask is provided, the matrix will be transformed to csr format.

        inplace
            If True, update AnnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    -------
    Returns a scaled copy or updates `adata` with a scaled version of the original `adata.X` and `adata.layers['layer']`, \
    depending on `inplace`.

    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()

    if isinstance(adata, AnnData):
        view_to_actual(adata)

    X = _get_obs_rep(adata, layer=layer, obsm=obsm)
    _check_gpu_X(X)

    str_mean_std = ("mean", "std")
    if mask_obs is not None:
        if isinstance(mask_obs, str):
            str_mean_std = (f"mean of {mask_obs}", f"std of {mask_obs}")
        else:
            str_mean_std = ("mean with mask", "std with mask")
        mask_obs = _check_mask(adata, mask_obs, "obs")

    if isinstance(X, cp.ndarray):
        X, means, std = _scale_array(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=inplace,
            max_value=max_value,
        )
    elif isinstance(X, sparse.csr_matrix):
        X, means, std = _scale_sparse_csr(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=inplace,
            max_value=max_value,
        )
    elif isinstance(X, sparse.csc_matrix):
        X, means, std = _scale_sparse_csc(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=inplace,
            max_value=max_value,
        )

    if inplace:
        _set_obs_rep(adata, X, layer=layer, obsm=obsm)
        adata.var[str_mean_std[0]] = means.get()
        adata.var[str_mean_std[1]] = std.get()

    if copy:
        return adata
    elif not inplace:
        return X


def _scale_array(X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None):
    if not inplace:
        X = X.copy()
    if mask_obs is None:
        mean, var = _get_mean_var(X)
        mask_array = cp.ones(X.shape[0], dtype=cp.int32)

    else:
        mean, var = _get_mean_var(X[mask_obs, :])
        mask_array = cp.array(mask_obs).astype(cp.int32)
    X = cp.ascontiguousarray(X)

    std = cp.sqrt(var)
    std[std == 0] = 1
    max_value = _get_max_value(max_value, X.dtype)
    if zero_center:
        from ._kernels._scale_kernel import _dense_center_scale_kernel

        scale_kernel_center = _dense_center_scale_kernel(X.dtype)

        scale_kernel_center(
            (math.ceil(X.shape[0] / 32), math.ceil(X.shape[1] / 32)),
            (32, 32),
            (
                X,
                mean.astype(X.dtype),
                std.astype(X.dtype),
                mask_array,
                max_value,
                X.shape[0],
                X.shape[1],
            ),
        )
    else:
        from ._kernels._scale_kernel import _dense_scale_kernel

        scale_kernel = _dense_scale_kernel(X.dtype)

        scale_kernel(
            (math.ceil(X.shape[0] / 32), math.ceil(X.shape[1] / 32)),
            (32, 32),
            (X, std.astype(X.dtype), mask_array, max_value, X.shape[0], X.shape[1]),
        )

    return X, mean, std


def _scale_sparse_csc(
    X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None
):
    if zero_center:
        X = X.toarray()
        # inplace is True because we copied with `toarray`
        return _scale_array(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=True,
            max_value=max_value,
        )
    elif mask_obs is not None:
        X = X.tocsr()
        return _scale_sparse_csr(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=True,
            max_value=max_value,
        )
    else:
        if not inplace:
            X = X.copy()

        mean, var = _get_mean_var(X)
        std = cp.sqrt(var)
        std[std == 0] = 1
        from ._kernels._scale_kernel import _csc_scale_diff

        scale_csc = _csc_scale_diff(X.dtype)
        scale_csc(
            (X.shape[1],),
            (64,),
            (X.indptr, X.data, std, X.shape[1]),
        )
        if max_value:
            X.data = cp.clip(X.data, a_min=None, a_max=max_value)

        return X, mean, std


def _scale_sparse_csr(
    X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None
):
    if zero_center:
        X = X.toarray()
        # inplace is True because we copied with `toarray`
        return _scale_array(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=True,
            max_value=max_value,
        )
    else:
        if not inplace:
            X = X.copy()
        if mask_obs is None:
            mean, var = _get_mean_var(X)
            mask_array = cp.ones(X.shape[0], dtype=cp.int32)

        else:
            mean, var = _get_mean_var(X[mask_obs, :])
            mask_array = cp.array(mask_obs).astype(cp.int32)
        std = cp.sqrt(var)
        std[std == 0] = 1

        max_value = _get_max_value(max_value, X.dtype)
        from ._kernels._scale_kernel import _csr_scale_kernel

        scale_csr = _csr_scale_kernel(X.dtype)
        scale_csr(
            (X.shape[0],),
            (64,),
            (X.indptr, X.indices, X.data, std, mask_array, max_value, X.shape[0]),
        )

        return X, mean, std


def _get_max_value(val, dtype):
    if val is None:
        val = np.inf
    if dtype == cp.float32:
        val = cp.float32(val)
    else:
        val = cp.float64(val)
    return val
