from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csc_matrix as csc_matrix_gpu
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import csc_matrix as csc_matrix_cpu
from scipy.sparse import csr_matrix as csr_matrix_cpu
from scipy.sparse import isspmatrix_csc as isspmatrix_csc_cpu
from scipy.sparse import isspmatrix_csr as isspmatrix_csr_cpu

if TYPE_CHECKING:
    from anndata import AnnData

GPU_ARRAY_TYPE = Union[cp.ndarray, csr_matrix_gpu, csc_matrix_gpu]
CPU_ARRAY_TYPE = Union[np.ndarray, csr_matrix_cpu, csc_matrix_cpu]


def anndata_to_GPU(
    adata: AnnData,
    layer: str | None = None,
    convert_all: bool = False,
    copy: bool = False,
) -> AnnData | None:
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

    copy
        Whether to return a copy or update `adata`.

    Returns
    -------
    Updates `adata` inplace or returns an updated copy
    """

    if copy:
        adata = adata.copy()

    if convert_all:
        anndata_to_GPU(adata)
        if adata.layers:
            for key in adata.layers.keys():
                anndata_to_GPU(adata, layer=key)
    else:
        X = _get_obs_rep(adata, layer=layer)
        error = layer if layer else "X"
        X = X_to_GPU(X, warning=error)
        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata


def X_to_GPU(X: CPU_ARRAY_TYPE, warning: str = "X") -> GPU_ARRAY_TYPE:
    """
    Transfers matrices and arrays to the GPU

    Parameters
    ----------
    X
        Matrix or array to transfer to the GPU
    warning
        Warning message to display if the input is not supported
    """
    if isinstance(X, GPU_ARRAY_TYPE):
        pass
    elif isspmatrix_csr_cpu(X):
        X = csr_matrix_gpu(X)
    elif isspmatrix_csc_cpu(X):
        X = csc_matrix_gpu(X)
    elif isinstance(X, np.ndarray):
        X = cp.array(X)
    else:
        warnings.warn(
            f"{warning} not supported for GPU conversion returning {warning}", Warning
        )
    return X


def anndata_to_CPU(
    adata: AnnData,
    layer: str | None = None,
    convert_all: bool = False,
    copy: bool = False,
) -> AnnData | None:
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

    copy
        Whether to return a copy or update `adata`.

    Returns
    -------
    Updates `adata` inplace or returns an updated copy
    """

    if copy:
        adata = adata.copy()

    if convert_all:
        anndata_to_CPU(adata)
        if adata.layers:
            for key in adata.layers.keys():
                anndata_to_CPU(adata, layer=key)
    else:
        X = _get_obs_rep(adata, layer=layer)
        X = X_to_CPU(X)
        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata


def X_to_CPU(X: GPU_ARRAY_TYPE) -> CPU_ARRAY_TYPE:
    """
    Transfers matrices and arrays from the GPU

    Parameters
    ----------
    X
        Matrix or array to transfer to the host memory
    """
    if isinstance(X, GPU_ARRAY_TYPE):
        X = X.get()
    else:
        pass
    return X
