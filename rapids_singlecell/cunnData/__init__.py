import cupy as cp
import cupyx as cpx
from anndata import AnnData
from anndata._core.index import _normalize_indices
import anndata

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from collections import OrderedDict
from typing import Any, Union, Optional, Mapping, MutableMapping
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
            raise ValueError(f"Shape of {key} does not match :attr:`.X`")

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
            raise ValueError(f"Shape of {key} does not match :attr:`.n_obs`")
            
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
            raise ValueError(f"Shape of {key} does not match :attr:`.n_vars`")




class cunnData:
    """
    The cunnData objects can be used as an AnnData replacement for the inital preprocessing 
    of single cell Datasets. It replaces some of the most common preprocessing steps within 
    scanpy for annData objects.
    It can be initalized with a preexisting annData object or with a countmatrix and seperate 
    Dataframes for var and obs. Index of var will be used as gene_names. Initalization with an 
    AnnData object is advised.
    """
    def __init__(
        self,
        adata: Optional[AnnData] = None,
        X: Optional[Union[np.ndarray,sparse.spmatrix, cp.ndarray, cpx.scipy.sparse.csr_matrix]] = None,
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
                    self._X = cp.sparse.csr_matrix(adata.X, dtype=cp.float32)
                self._obs = adata.obs.copy()
                self._var = adata.var.copy()
                self._uns = adata.uns.copy()
                self._layers = Layer_Mapping(self.shape)
                self._obsm = obsm_Mapping(self.shape[0])
                self._varm = varm_Mapping(self.shape[1])
                self.raw = None
                if adata.layers:
                    for key, matrix in adata.layers.items():
                        if not issparse_cpu(matrix):
                            inter = scipy.sparse.csr_matrix(matrix)
                            inter = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                            
                        else:
                            inter = cp.sparse.csr_matrix(matrix, dtype=cp.float32)
                        self._layers[key] = inter.copy()
                        del inter
                if adata.obsm:
                    for key, matrix in adata.obsm.items():
                        self._obsm[key] = matrix.copy()
                if adata.varm:
                    for key, matrix in adata.varm.items():
                        self._varm[key] = matrix.copy()
                
            else:
                if issparse_gpu(X):
                    self._X = X      
                elif isinstance(X,cp.ndarray):
                    self._X  = X            
                elif not issparse_cpu(X):
                    inter = scipy.sparse.csr_matrix(X)
                    self._X = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                    del inter
                else:
                    self._X = cp.sparse.csr_matrix(X, dtype=cp.float32)

                self._obs = obs
                self._var = var
                if uns:
                    self._uns = uns
                else:
                    self._uns = OrderedDict()
                self._layers = Layer_Mapping(self.shape)
                self._obsm = obsm_Mapping(self.shape[0])
                self._varm = varm_Mapping(self.shape[1])
                self.raw = None

                if layers:
                    for key, matrix in layers.items():
                        if issparse_gpu(matrix):
                            inter = matrix
                        elif isinstance(matrix,cp.ndarray):
                            inter = matrix               
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
    def X(self):
        """Data matrix of shape :attr:`.n_obs` × :attr:`.n_vars`."""
        return  self._X

    @X.setter
    def X(self, value: Optional[Union[cp.ndarray, cpx.scipy.sparse.spmatrix]]):
        self._X = value
        self._update_shape()


    @property
    def obs(self):
        """One-dimensional annotation of observations (`pd.DataFrame`)."""     
        return  self._obs

    @obs.setter
    def obs(self, value: pd.DataFrame):
        if value.shape[0] == self.shape[0]:
            self._obs = value.copy()
        else:
            raise ValueError("dimension mismatch")
    @property
    def var(self):
        """One-dimensional annotation of variables/ features (`pd.DataFrame`)."""
        return self._var

    @var.setter
    def var(self,value: pd.DataFrame):
        if value.shape[0] == self.shape[1]:
            self._var = value
        else:
            raise ValueError("dimension mismatch")

    @property
    def uns(self):
        """Unstructured annotation (ordered dictionary)."""
        return self._uns

    @uns.setter
    def uns(self, value: MutableMapping):
        if not isinstance(value, MutableMapping):
            raise ValueError(
                "Only mutable mapping types (e.g. dict) are allowed for `.uns`."
            )
        if isinstance(value, (anndata.compact.OverloadedDict, anndata._core.views.DictView)):
            value = value.copy()
        self._uns = value

    @uns.deleter
    def uns(self):
        self.uns = OrderedDict()

    @property
    def layers(self):
        """\
        Dictionary-like object with values of the same dimensions as :attr:`.X`.
        Layers in cunnData are inspired by AnnData.

        Return the layer named `"unspliced"`::
            adata.layers["unspliced"]
        Create or replace the `"spliced"` layer::
            adata.layers["spliced"] = ...
        Assign the 10th column of layer `"spliced"` to the variable a::
            a = adata.layers["spliced"][:, 10]
        Delete the `"spliced"` layer::
            del adata.layers["spliced"]
        Return layers’ names::
            adata.layers.keys()
        """  
        return  self._layers

    @property
    def obsm(self):
        """\
        Multi-dimensional annotation of observations
        (mutable structured :class:`~numpy.ndarray`).
        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        of length :attr:`n_obs`.
        Is sliced with `data` and `obs` but behaves otherwise like a :term:`mapping`.
        """      
        return  self._obsm


    @property
    def varm(self):
        """\
        Multi-dimensional annotation of variables/features
        (mutable structured :class:`~numpy.ndarray`).
        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        of length :attr:`n_vars`.
        Is sliced with `data` and `var` but behaves otherwise like a :term:`mapping`.
        """
        return  self._varm
    

    @property
    def obs_names(self):
        """Names of observations (alias for `.obs.index`)."""        
        return  self.obs.index

    @property
    def var_names(self):
        """Names of variables (alias for `.var.index`)."""
        return self.var.index
    
    @property
    def n_obs(self)->int:
        """Number of observations."""
        return self.shape[0]

    @property
    def n_vars(self)->int:
        """Number of variables/features."""
        return self.shape[1]

    @property
    def shape(self):
        """Shape of data matrix (:attr:`.n_obs`, :attr:`.n_vars`)."""
        return self.X.shape

    @property
    def nnz(self):
        """Get the count of explicitly-stored values (nonzeros) in :attr:`.X`"""
        if issparse_gpu(self.X):
            return self.X.nnz
        else:
            return None
    
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
        :class:`~anndata.AnnData`
            Annotated data matrix.
        
        """
        adata = AnnData(self.X.get())
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
