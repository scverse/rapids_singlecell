import cupy as cp
import anndata
from anndata._core.index import _normalize_indices

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from typing import Any, Union, Optional, Mapping
from pandas.api.types import infer_dtype

from natsort import natsorted

from scipy.sparse import issparse as issparse_cpu
from cupyx.scipy.sparse import issparse as issparse_gpu


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
    Dictonary subclass for obsm handeling in cunnData
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
            raise ValueError(f"Shape of {key} does not match `.n_obs`")
            
class varm_Mapping(dict):
    """
    Dictonary subclass for obsm handeling in cunnData
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
            raise ValueError(f"Shape of {key} does not match `.n_vars`")




class cunnData:
    """
    The cunnData objects can be used as an AnnData replacement for the inital preprocessing of single cell Datasets. It replaces some of the most common preprocessing steps within scanpy for annData objects.
    It can be initalized with a preexisting annData object or with a countmatrix and seperate Dataframes for var and obs. Index of var will be used as gene_names. Initalization with an AnnData object is advised.
    """
    uns = {}
    def __init__(
        self,
        adata: Optional[anndata.AnnData] = None,
        X: Optional[Union[np.ndarray,sparse.spmatrix, cp.array, cp.sparse.csr_matrix]] = None,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Mapping[str, Any]] = None,
        layers: Optional[Mapping[str, Any]] = None,
        obsm: Optional[Mapping[str, Any]] = None,
        varm: Optional[Mapping[str, Any]] = None):
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
                self.varm = varm_Mapping(self.shape[1])
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
                if adata.varm:
                    for key, matrix in adata.varm.items():
                        self.varm[key] = matrix
                
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
                self.varm = varm_Mapping(self.shape[1])
                self.raw = None

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
                if varm:
                    for key, matrix in adata.varm.items():
                        self.varm[key] = matrix
    
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
    
    def _update_shape(self):
        self.layers.update_shape(self.shape)
        self.obsm.update_shape(self.shape[0])
        self.varm.update_shape(self.shape[1])
        
    def _sanitize(self):
        dfs = [self.obs, self.var]
        for df in dfs:
            string_cols = [
                key for key in df.columns if infer_dtype(df[key]) == "string"
            ]
            for key in string_cols:
                c = pd.Categorical(df[key])
                if len(c.categories) >= len(c):
                    continue
                # Ideally this could be done inplace
                sorted_categories = natsorted(c.categories)
                if not np.array_equal(c.categories, sorted_categories):
                    c = c.reorder_categories(sorted_categories)
                df[key] = c

    def __getitem__(self, index):
        obs_dx,var_dx = _normalize_indices(index, self.obs_names, self.var_names)
        self.X = self.X[obs_dx,var_dx]
        self._update_shape()
        if self.layers:
            for key, matrix in self.layers.items():
                self.layers[key] = matrix[obs_dx, var_dx]
        if self.obsm:
            for key, matrix in self.obsm.items():
                if isinstance(matrix, pd.DataFrame):
                    self.obsm[key] = matrix.iloc[obs_dx, :]
                else:
                    self.obsm[key] = matrix[obs_dx, :]
        if self.varm:
            for key, matrix in self.varm.items():
                if isinstance(matrix, pd.DataFrame):
                    self.varm[key] = matrix.iloc[var_dx, :]
                else:
                    self.varm[key] = matrix[var_dx, :]
        return(cunnData(X = self.X,
                        obs = self.obs.iloc[obs_dx,:],
                        var = self.var.iloc[var_dx,:],
                        uns=self.uns,
                        layers= self.layers,
                        obsm= self.obsm,
                        varm= self.varm))
    
    def _gen_repr(self, n_obs, n_vars) -> str:
        descr = f"cunnData object with n_obs × n_vars = {n_obs} × {n_vars}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
            return self._gen_repr(self.n_obs, self.n_vars)


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
        if self.varm:
            for key, matrix in self.varm.items():
                adata.varm[key] = matrix.copy()
        return adata
