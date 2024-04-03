from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
from anndata import AnnData
from cupyx.scipy import sparse
from scanpy._utils import view_to_actual

from rapids_singlecell.get import _check_mask, _get_obs_rep, _set_obs_rep

from ._utils import _check_gpu_X, _get_mean_var

if TYPE_CHECKING:
    import numpy as np


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
        X, means, std = scale_array(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=inplace,
            max_value=max_value,
        )
    else:
        X, means, std = scale_sparse(
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


def scale_array(X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None):
    if not inplace:
        X = X.copy()
    if mask_obs is not None:
        scale_rv = scale_array(X[mask_obs, :], zero_center=zero_center, inplace=True)
        X[mask_obs, :], mean, std = scale_rv
        return X, mean, std

    X = cp.ascontiguousarray(X)
    mean, var = _get_mean_var(X)
    std = cp.sqrt(var)
    std[std == 0] = 1
    if zero_center:
        X -= mean
    X /= std
    if max_value:
        if zero_center:
            X = cp.clip(X, a_min=-max_value, a_max=max_value)
        else:
            X[X > max_value] = max_value

    return X, mean, std


def scale_sparse(X, *, mask_obs=None, zero_center=True, inplace=True, max_value=None):
    if zero_center:
        X = X.toarray()
        # inplace is True because we copied with `toarray`
        return scale_array(
            X,
            mask_obs=mask_obs,
            zero_center=zero_center,
            inplace=True,
            max_value=max_value,
        )
    else:
        if mask_obs is not None:
            # checking inplace because we are going to update the matrix
            # `tocsr` copies the matrix
            if sparse.isspmatrix_csc(X):
                X = X.tocsr()
            elif not inplace:
                X = X.copy()

            scale_rv = scale_sparse(
                X[mask_obs, :],
                zero_center=zero_center,
                inplace=True,
                max_value=max_value,
            )
            X_sub, mean, std = scale_rv
            mask_array = cp.where(cp.array(mask_obs))[0].astype(cp.int32)

            from ._kernels._scale_kernel import _csr_update

            update_inplace = _csr_update(X.dtype)
            update_inplace(
                (math.ceil(X.shape[0] / 64),),
                (64,),
                (
                    X.indptr,
                    X.data,
                    X_sub.indptr,
                    X_sub.data,
                    mask_array,
                    X_sub.shape[0],
                ),
            )
            return X, mean, std

        mean, var = _get_mean_var(X)
        std = cp.sqrt(var)
        std[std == 0] = 1
        if not inplace:
            X = X.copy()

        if sparse.isspmatrix_csr(X):
            X.data /= std[X.indices]
        elif sparse.isspmatrix_csc(X):
            from ._kernels._scale_kernel import _csc_scale_diff

            scale_csc = _csc_scale_diff(X.dtype)
            scale_csc(
                (X.shape[1],),
                (64,),
                (X.indptr, X.data, std, X.shape[1]),
            )
        else:
            raise ValueError("The sparse matrix must be a CSR or CSC matrix")

        if max_value:
            X.data[X.data > max_value] = max_value

        return X, mean, std
