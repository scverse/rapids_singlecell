import warnings
from typing import Optional

import cupy as cp
import numpy as np
from anndata import AnnData
from cupyx.scipy.sparse import csc_matrix as csc_matrix_gpu
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from cupyx.scipy.sparse import isspmatrix_csc as isspmatrix_csc_gpu
from cupyx.scipy.sparse import isspmatrix_csr as isspmatrix_csr_gpu
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import isspmatrix_csc as isspmatrix_csc_cpu
from scipy.sparse import isspmatrix_csr as isspmatrix_csr_cpu


def anndata_to_GPU(
    adata: AnnData,
    layer: Optional[str] = None,
    convert_all: bool = False,
) -> None:
    """
    Transfers matrices and arrays to the GPU

    Parameters
    ----------
    adata
        AnnData object

    layer
        Layer to use as input instead of `X`. If `None`, `X` is used.

    convert_all
        If True, move all supported arrays and matrices on the GPU

    Returns
    -------
    Updates `adata` inplace
    """
    if convert_all:
        anndata_to_GPU(adata)
        if adata.layers:
            for key in adata.layers.keys():
                anndata_to_GPU(adata, layer=key)
    else:
        X = _get_obs_rep(adata, layer=layer)
        if isspmatrix_csr_cpu(X):
            X = csr_matrix_gpu(X)
        elif isspmatrix_csc_cpu(X):
            X = csc_matrix_gpu(X)
        elif isinstance(X, np.ndarray):
            X = cp.array(X)
        else:
            error = layer if layer else "X"
            warnings.warn(f"{error} not supported for GPU conversion", Warning)

        _set_obs_rep(adata, X, layer=layer)


def anndata_to_CPU(
    adata: AnnData,
    layer: Optional[str] = None,
    convert_all: bool = False,
) -> None:
    """
    Transfers matrices and arrays from the GPU

    Parameters
    ----------
    adata
        AnnData object

    layer
        Layer to use as input instead of `X`. If `None`, `X` is used.

    convert_all
        If True, move all GPU based arrays and matrices to the host memory

    Returns
    -------
    Updates `adata` inplace
    """
    if convert_all:
        anndata_to_CPU(adata)
        if adata.layers:
            for key in adata.layers.keys():
                anndata_to_CPU(adata, layer=key)
    else:
        X = _get_obs_rep(adata, layer=layer)
        if isspmatrix_csr_gpu(X):
            X = X.get()
        elif isspmatrix_csc_gpu(X):
            X = X.get()
        elif isinstance(X, cp.ndarray):
            X = X.get()
        else:
            pass

        _set_obs_rep(adata, X, layer=layer)
