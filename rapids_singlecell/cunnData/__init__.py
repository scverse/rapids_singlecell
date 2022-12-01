#
# created by Severin Dicks (IBSM, Freiburg)
#
#

import cupy as cp
import cupyx as cpx
import anndata

import numpy as np
import pandas as pd
import scipy
import math
from scipy import sparse
from typing import Any, Union, Optional, Mapping

import warnings

from scipy.sparse import issparse as issparse_cpu
from cupyx.scipy.sparse import issparse as issparse_gpu

from cuml.linear_model import LinearRegression


class Layer_Mapping(dict):
    """
    Dictonary subclass for layers handeling in cunnData
    """
    def __init__(self, shape):
        super().__init__({})
        self.shape = shape
    
    def update_shape(self,shape):
        self.shape = shape

    def __setitem__(self, key, item):
        if self.shape == item.shape:
            super().__setitem__(key, item)
        else:
            raise ValueError(f"Shape of {key} does not match `.X`")

class obsm_Mapping(dict):
    """
    Dictonary subclass for layers handeling in cunnData
    """
    def __init__(self, shape):
        super().__init__({})
        self.shape = shape
    
    def update_shape(self,shape):
        self.shape = shape

    def __setitem__(self, key, item):
        if self.shape == item.shape[0]:
            super().__setitem__(key, item)
        else:
            raise ValueError(f"Shape of {key} does not match `.X`")

class cunnData:
    """
    The cunnData objects can be used as an AnnData replacement for the inital preprocessing of single cell Datasets. It replaces some of the most common preprocessing steps within scanpy for annData objects.
    It can be initalized with a preexisting annData object or with a countmatrix and seperate Dataframes for var and obs. Index of var will be used as gene_names. Initalization with an AnnData object is advised.
    """
    uns = {}
    def __init__(
        self,
        X: Optional[Union[np.ndarray,sparse.spmatrix, cp.array, cp.sparse.csr_matrix]] = None,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Mapping[str, Any]] = None,
        layers: Optional[Mapping[str, Any]] = None,
        obsm: Optional[Mapping[str, Any]] = None,
        adata: Optional[anndata.AnnData] = None):
            if adata:
                if not issparse_cpu(adata.X):
                    inter = scipy.sparse.csr_matrix(adata.X)
                    self.X = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                    del inter
                else:
                    self.X = cp.sparse.csr_matrix(adata.X, dtype=cp.float32)
                self.obs = adata.obs.copy()
                self.var = adata.var.copy()
                self.uns = adata.uns.copy()
                self.layers = Layer_Mapping(self.shape)
                self.obsm = obsm_Mapping(self.shape[0])
                if adata.layers:
                    for key, matrix in adata.layers.items():
                        if not issparse_cpu(matrix):
                            inter = scipy.sparse.csr_matrix(matrix)
                            inter = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                            
                        else:
                            inter = cp.sparse.csr_matrix(matrix, dtype=cp.float32)
                    self.layers[key] = inter.copy()
                    del inter
                if adata.obsm:
                    for key, matrix in adata.obsm.items():
                        self.obsm[key] = matrix
                
            else:
                if issparse_gpu(X):
                    self.X = X                
                elif not issparse_cpu(X):
                    inter = scipy.sparse.csr_matrix(X)
                    self.X = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                    del inter
                else:
                    self.X = cp.sparse.csr_matrix(X, dtype=cp.float32)

                self.obs = obs
                self.var = var
                self.uns = uns
                self.layers = Layer_Mapping(self.shape)
                self.obsm = obsm_Mapping(self.shape[0])
                if layers:
                    for key, matrix in layers.items():
                        if issparse_gpu(matrix):
                            inter = matrix.copy()               
                        elif not issparse_cpu(X):
                            inter = scipy.sparse.csr_matrix(matrix)
                            inter = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                        else:
                            inter = cp.sparse.csr_matrix(matrix, dtype=cp.float32)
                        self.layers[key] = inter.copy()
                        del inter
                if obsm:
                    for key, matrix in obsm.items():
                        self.obsm[key] = matrix.copy()
    
    @property
    def shape(self):
        return self.X.shape
    @property
    def nnz(self):
        return self.X.nnz
    
    @property
    def obs_names(self):
        return  self.obs.index

    @property
    def var_names(self):
        return self.var.index
    
    @property
    def n_obs(self):
        return self.shape[0]

    @property
    def n_vars(self):
        return self.shape[1]

    def __getitem__(self, index):
        """
        Currently only works for `obs`
        """
        if type(index) == tuple:
            obs_dx, var_dx = index
        else:
            obs_dx = index
            var_dx = slice(None,None,None)
        
        if isinstance(obs_dx,pd.Series):
            obs_dx = obs_dx.values
        
        if isinstance(var_dx,pd.Series):
            var_dx = var_dx.values

        self.X = self.X[obs_dx,var_dx]
        self.layers.update_shape(self.shape)
        self.obsm.update_shape(self.shape[0])
        if self.layers:
            for key, matrix in self.layers.items():
                self.layers[key] = matrix[obs_dx, var_dx]
        if self.obsm:
            for key, matrix in self.obsm.items():
                if isinstance(matrix, pd.DataFrame):
                    self.obsm[key] = matrix.iloc[obs_dx, :]
                else:
                    self.obsm[key] = matrix[obs_dx, :]
        return(cunnData(X = self.X,obs = self.obs.iloc[obs_dx,:],var = self.var.iloc[var_dx,:],uns=self.uns,layers= self.layers,obsm= self.obsm))


    def to_AnnData(self):
        """
        Takes the cunnData object and creates an AnnData object
        
        Returns
        -------
            annData object
        
        """
        adata = anndata.AnnData(self.X.get())
        adata.obs = self.obs.copy()
        adata.var = self.var.copy()
        adata.uns = self.uns.copy()
        if self.layers:
            for key, matrix in self.layers.items():
                adata.layers[key] = matrix.get()
        if self.obsm:
            for key, matrix in self.obsm.items():
                adata.obsm[key] = matrix.copy()
        return adata