import cupy as cp
import cupyx as cpx
from anndata import AnnData
from anndata._core.index import _normalize_indices
import anndata

import numpy as np
import pandas as pd
from scipy import sparse
from collections import OrderedDict
from typing import Any, Union, Optional, Mapping, MutableMapping, List
from pandas.api.types import infer_dtype, is_string_dtype
from itertools import repeat
import warnings

from natsort import natsorted

from scipy.sparse import issparse as issparse_cpu
from cupyx.scipy.sparse import issparse as issparse_gpu


class Layer_Mapping(dict):
    """
    Dictonary subclass for layers handeling in cunnData
    """

    def __init__(self, shape=None):
        super().__init__({})
        self.shape = shape

    def update_shape(self, shape):
        self.shape = shape

    def __setitem__(self, key, item):
        if self.shape == item.shape:
            if issparse_gpu(item):
                item = item.astype(cp.float32)
            elif isinstance(item, cp.ndarray):
                item = item.astype(cp.float32)
            elif not issparse_cpu(item):
                inter = sparse.csr_matrix(item)
                item = cpx.scipy.sparse.csr_matrix(inter, dtype=cp.float32)
            else:
                item = cpx.scipy.sparse.csr_matrix(item, dtype=cp.float32)
            if issparse_gpu(item):
                if item.nnz > 2**31 - 1:
                    raise ValueError(
                        "Cupy only supports Sparse Matrices with `.nnz`"
                        "with less than 2**31-1 for the int32 indptr"
                    )
            super().__setitem__(key, item)
        else:
            raise ValueError(
                f"Shape of {key} {item.shape} does not match :attr:`.X` {self.shape}"
            )


class obsm_Mapping(dict):
    """
    Dictonary subclass for obsm handeling in cunnData
    """

    def __init__(self, shape=None):
        super().__init__({})
        self.shape = shape

    def update_shape(self, shape):
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

    def __init__(self, shape=None):
        super().__init__({})
        self.shape = shape

    def update_shape(self, shape):
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
        X: Optional[
            Union[np.ndarray, sparse.spmatrix, cp.ndarray, cpx.scipy.sparse.csr_matrix]
        ] = None,
        obs: Optional[Union[pd.DataFrame, Mapping, None]] = None,
        var: Optional[Union[pd.DataFrame, Mapping, None]] = None,
        uns: Optional[Mapping[str, Any]] = None,
        layers: Optional[Mapping[str, Any]] = None,
        obsm: Optional[Mapping[str, Any]] = None,
        varm: Optional[Mapping[str, Any]] = None,
    ):
        # Initialize from adata
        if adata:
            if not issparse_cpu(adata.X):
                inter = sparse.csr_matrix(adata.X)
                self._X = cpx.scipy.sparse.csr_matrix(inter, dtype=cp.float32)
                del inter
            else:
                self._X = cpx.scipy.sparse.csr_matrix(adata.X, dtype=cp.float32)
            self._obs = adata.obs.copy()
            self._var = adata.var.copy()
            self._uns = adata.uns.copy()
            self._layers = Layer_Mapping()
            self._obsm = obsm_Mapping()
            self._varm = varm_Mapping()
            self.raw = None
            self._update_shape()
            if adata.layers:
                for key, matrix in adata.layers.items():
                    self._layers[key] = matrix
            if adata.obsm:
                for key, matrix in adata.obsm.items():
                    self._obsm[key] = matrix.copy()
            if adata.varm:
                for key, matrix in adata.varm.items():
                    self._varm[key] = matrix.copy()

        # Initialize from items
        else:
            if issparse_gpu(X):
                self._X = X.astype(cp.float32)
            elif isinstance(X, cp.ndarray):
                self._X = X.astype(cp.float32)
            elif not issparse_cpu(X):
                inter = sparse.csr_matrix(X)
                self._X = cpx.scipy.sparse.csr_matrix(inter, dtype=cp.float32)
                del inter
            else:
                self._X = cpx.scipy.sparse.csr_matrix(X, dtype=cp.float32)
            if obs is not None:
                if isinstance(obs, Mapping):
                    self._obs = pd.DataFrame(obs)
                    if self.obs.empty:
                        self._obs = pd.DataFrame(index=range(self.shape[0]))
                elif isinstance(obs, pd.DataFrame):
                    self._obs = obs.copy()
                elif isinstance(obs, pd.Series):
                    obs_df = pd.DataFrame(obs)
                    if obs_df.shape[0] == self.shape[0]:
                        self._obs = obs_df
                    else:
                        self._obs = obs_df.T
                else:
                    raise ValueError(".obs has to be a Mapping or Dataframe")
            else:
                self._obs = pd.DataFrame(index=range(self.shape[0]))
            if self.obs.shape[0] != self.shape[0]:
                raise ValueError("Shape mismatch: 'obs' rows must equal '.X' rows.")
            if not is_string_dtype(self.obs):
                self.obs.index = self.obs.index.astype(str)
            if var is not None:
                if isinstance(var, Mapping):
                    self._var = pd.DataFrame(var)
                    if self.var.empty:
                        self._var = pd.DataFrame(index=range(self.shape[1]))
                elif isinstance(var, pd.DataFrame):
                    self._var = var.copy()
                elif isinstance(var, pd.Series):
                    var_df = pd.DataFrame(var)
                    if var_df.shape[0] == self.shape[1]:
                        self._var = var_df
                    else:
                        self._var = var_df.T
                else:
                    raise ValueError(".var has to be a Mapping or Dataframe")
            else:
                self._var = pd.DataFrame(index=range(self.shape[1]))
            if self.var.shape[0] != self.shape[1]:
                raise ValueError("Shape mismatch: 'var' rows must equal '.X' columns.")
            if not is_string_dtype(self.var):
                self.var.index = self.var.index.astype(str)
            if uns:
                self._uns = uns
            else:
                self._uns = OrderedDict()
            self._layers = Layer_Mapping()
            self._obsm = obsm_Mapping()
            self._varm = varm_Mapping()
            self.raw = None
            self._update_shape()
            if layers:
                for key, matrix in layers.items():
                    self.layers[key] = matrix
            if obsm:
                for key, matrix in obsm.items():
                    self.obsm[key] = matrix.copy()
            if varm:
                for key, matrix in adata.varm.items():
                    self.varm[key] = matrix

        if issparse_gpu(self._X):
            if self._X.nnz > 2**31 - 1:
                raise ValueError(
                    "Cupy only supports Sparse Matrices with `.nnz` "
                    "less than 2**31-1 due to the int32 limit on `indptr`."
                )

    @property
    def X(self):
        """Data matrix of shape :attr:`.n_obs` × :attr:`.n_vars`."""
        return self._X

    @X.setter
    def X(self, value: Optional[Union[cp.ndarray, cpx.scipy.sparse.spmatrix]]):
        if value.shape != self.shape:
            raise ValueError(
                f"Dimension mismatch: value has shape {value.shape}, but expected shape is {self.shape}"
            )
        if _check_X(value):
            self._X = value.astype(cp.float32)
            self._update_shape()
        else:
            raise TypeError("Input must be a CuPy ndarray, CSR matrix, or CSC matrix.")

    @property
    def obs(self):
        """One-dimensional annotation of observations (`pd.DataFrame`)."""
        return self._obs

    @obs.setter
    def obs(self, value: Union[pd.DataFrame, Mapping]):
        if not isinstance(value, Mapping) and not isinstance(value, pd.DataFrame):
            raise ValueError(".obs has to be a Mapping or Dataframe")
        if isinstance(value, Mapping):
            value = pd.DataFrame(value)
        if value.shape[0] == self.shape[0]:
            self._obs = value.copy()
        else:
            raise ValueError(
                f"Dimension mismatch: value has shape {value.shape[0]}, but expected shape is {self.shape[0]}"
            )
        if not is_string_dtype(self._obs):
            self._obs.index = self._obs.index.astype(str)

    @obs.deleter
    def obs(self):
        self._obs = pd.DataFrame(index=range(self.shape[0]))
        self._obs.index = self._obs.index.astype(str)

    @property
    def var(self):
        """One-dimensional annotation of variables/ features (`pd.DataFrame`)."""
        return self._var

    @var.setter
    def var(self, value: Union[pd.DataFrame, Mapping]):
        if not isinstance(value, Mapping) and not isinstance(value, pd.DataFrame):
            raise ValueError(".var has to be a Mapping or Dataframe")
        if isinstance(value, Mapping):
            value = pd.DataFrame(value)
        if value.shape[0] == self.shape[1]:
            self._var = value
        else:
            raise ValueError(
                f"Dimension mismatch: value has shape {value.shape[0]}, but expected shape is {self.shape[1]}"
            )
        if not is_string_dtype(self._var):
            self._var.index = self._var.index.astype(str)

    @var.deleter
    def var(self):
        self._var = pd.DataFrame(index=range(self.shape[1]))
        self._var.index = self._var.index.astype(str)

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
        if isinstance(value, (anndata._core.views.DictView)):
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
        return self._layers

    @property
    def obsm(self):
        """\
        Multi-dimensional annotation of observations
        (mutable structured :class:`~numpy.ndarray`).
        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        of length :attr:`n_obs`.
        Is sliced with `data` and `obs` but behaves otherwise like a :term:`mapping`.
        """
        return self._obsm

    @property
    def varm(self):
        """\
        Multi-dimensional annotation of variables/features
        (mutable structured :class:`~numpy.ndarray`).
        Stores for each key a two or higher-dimensional :class:`~numpy.ndarray`
        of length :attr:`n_vars`.
        Is sliced with `data` and `var` but behaves otherwise like a :term:`mapping`.
        """
        return self._varm

    @property
    def obs_names(self):
        """Names of observations (alias for `.obs.index`)."""
        return self.obs.index

    @property
    def var_names(self):
        """Names of variables (alias for `.var.index`)."""
        return self.var.index

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.shape[0]

    @property
    def n_vars(self) -> int:
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
        obs_dx, var_dx = _normalize_indices(index, self.obs_names, self.var_names)
        x_dim = _get_dimensions(obs_dx, self.shape[0])
        y_dim = _get_dimensions(var_dx, self.shape[1])
        cudata = cunnData(
            X=self.X[obs_dx, var_dx].reshape(x_dim, y_dim).copy(),
            obs=self.obs.iloc[obs_dx, :],
            var=self.var.iloc[var_dx, :],
            uns=self.uns,
        )
        if self.layers:
            for key, matrix in self.layers.items():
                cudata.layers[key] = matrix[obs_dx, var_dx].reshape(x_dim, y_dim).copy()
        if self.obsm:
            for key, matrix in self.obsm.items():
                if isinstance(matrix, pd.DataFrame):
                    sliced = matrix.iloc[obs_dx, :].copy()
                    if isinstance(sliced, pd.Series):
                        sliced = pd.DataFrame(sliced)
                        if sliced.shape[0] != cudata.shape[0]:
                            sliced = sliced.T
                    cudata.obsm[key] = sliced
                else:
                    org_dim = matrix.shape[1]
                    cudata.obsm[key] = matrix[obs_dx, :].reshape(x_dim, org_dim).copy()
        if self.varm:
            for key, matrix in self.varm.items():
                if isinstance(matrix, pd.DataFrame):
                    sliced = matrix.iloc[var_dx, :].copy()
                    if isinstance(sliced, pd.Series):
                        sliced = pd.DataFrame(sliced)
                        if sliced.shape[0] != cudata.shape[1]:
                            sliced = sliced.T
                    cudata.varm[key] = sliced
                else:
                    org_dim = matrix.shape[1]
                    cudata.varm[key] = matrix[var_dx, :].reshape(y_dim, org_dim).copy()
        return cudata

    def _inplace_subset_var(self, index):
        """\
        Inplace subsetting along variables dimension.

        Same as `adata = adata[:, index]`, but inplace.
        """
        _, var_dx = _normalize_indices(
            (slice(None, None, None), index), self.obs_names, self.var_names
        )
        y_dim = _get_dimensions(var_dx, self.shape[1])
        self._X = self.X[:, var_dx].reshape(self.n_obs, y_dim)
        self._update_shape()
        self.var = pd.DataFrame(self.var.iloc[var_dx, :])
        if self.layers:
            for key, matrix in self.layers.items():
                self.layers[key] = matrix[:, var_dx].reshape(self.n_obs, y_dim)
        if self.varm:
            for key, matrix in self.varm.items():
                if isinstance(matrix, pd.DataFrame):
                    sliced = matrix.iloc[var_dx, :].copy()
                    if isinstance(sliced, pd.Series):
                        sliced = pd.DataFrame(sliced)
                        if sliced.shape[0] != self.shape[1]:
                            sliced = sliced.T
                    self.varm[key] = sliced
                else:
                    org_dim = matrix.shape[1]
                    self.varm[key] = matrix[var_dx, :].reshape(y_dim, org_dim).copy()

    def _inplace_subset_obs(self, index):
        """\
        Inplace subsetting along observations dimension.

        Same as `adata = adata[index, :]`, but inplace.
        """
        obs_dx, _ = _normalize_indices(
            (index, slice(None, None, None)), self.obs_names, self.var_names
        )
        x_dim = _get_dimensions(obs_dx, self.shape[0])
        self._X = self.X[obs_dx, :].reshape(x_dim, self.n_vars)
        self._update_shape()
        self.obs = pd.DataFrame(self.obs.iloc[obs_dx, :])
        if self.layers:
            for key, matrix in self.layers.items():
                self.layers[key] = matrix[obs_dx, :].reshape(x_dim, self.n_vars)
        if self.obsm:
            for key, matrix in self.obsm.items():
                if isinstance(matrix, pd.DataFrame):
                    sliced = matrix.iloc[obs_dx, :].copy()
                    if isinstance(sliced, pd.Series):
                        sliced = pd.DataFrame(sliced)
                        if sliced.shape[0] != self.shape[0]:
                            sliced = sliced.T
                    self.obsm[key] = sliced
                else:
                    org_dim = matrix.shape[1]
                    self.obsm[key] = matrix[obs_dx, :].reshape(x_dim, org_dim).copy()

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

    def _normalize_indices(self, index):
        return _normalize_indices(index, self.obs_names, self.var_names)

    def _get_X(self, layer=None):
        """\
        Convenience method for getting expression values
        with common arguments and error handling.
        """
        is_layer = layer is not None
        if is_layer:
            return self.layers[layer]
        else:
            return self.X

    def obs_vector(self, k: str, *, layer: Optional[str] = None):
        """\
        Convenience function for returning a 1 dimensional ndarray of values
        from :attr:`X`, :attr:`layers`\\ `[k]`, or :attr:`obs`.

        Made for convenience, not performance.
        Intentionally permissive about arguments, for easy iterative use.

        Params
        ------
        k
            Key to use. Should be in :attr:`var_names` or :attr:`obs`\\ `.columns`.
        layer
            What layer values should be returned from. If `None`, :attr:`X` is used.

        Returns
        -------
        A one dimensional nd array, with values for each obs in the same order
        as :attr:`obs_names`.
        """
        if layer == "X":
            if "X" in self.layers:
                pass
            else:
                warnings.warn(
                    "In a future version of AnnData, access to `.X` by passing"
                    " `layer='X'` will be removed. Instead pass `layer=None`.",
                    FutureWarning,
                )
                layer = None
        return _get_vector(self, k, "obs", "var", layer=layer)

    def var_vector(self, k, *, layer: Optional[str] = None):
        """\
        Convenience function for returning a 1 dimensional ndarray of values
        from :attr:`X`, :attr:`layers`\\ `[k]`, or :attr:`var`.

        Made for convenience, not performance. Intentionally permissive about
        arguments, for easy iterative use.

        Params
        ------
        k
            Key to use. Should be in :attr:`obs_names` or :attr:`var`\\ `.columns`.
        layer
            What layer values should be returned from. If `None`, :attr:`X` is used.

        Returns
        -------
        A one dimensional nd array, with values for each var in the same order
        as :attr:`var_names`.
        """
        if layer == "X":
            if "X" in self.layers:
                pass
            else:
                warnings.warn(
                    "In a future version of AnnData, access to `.X` by passing "
                    "`layer='X'` will be removed. Instead pass `layer=None`.",
                    FutureWarning,
                )
                layer = None
        return _get_vector(self, k, "var", "obs", layer=layer)

    def __repr__(self) -> str:
        return self._gen_repr(self.n_obs, self.n_vars)

    def obs_keys(self) -> List[str]:
        """List keys of observation annotation :attr:`obs`."""
        return self._obs.keys().tolist()

    def var_keys(self) -> List[str]:
        """List keys of variable annotation :attr:`var`."""
        return self._var.keys().tolist()

    def obsm_keys(self) -> List[str]:
        """List keys of observation annotation :attr:`obsm`."""
        return list(self._obsm.keys())

    def varm_keys(self) -> List[str]:
        """List keys of variable annotation :attr:`varm`."""
        return list(self._varm.keys())

    def uns_keys(self) -> List[str]:
        """List keys of unstructured annotation."""
        return sorted(list(self._uns.keys()))

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


def _slice_length(slice_obj, iterable_length):
    start, stop, step = slice_obj.indices(iterable_length)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def _get_dimensions(s_object, shape):
    if isinstance(s_object, slice):
        return _slice_length(s_object, shape)
    if isinstance(s_object, int):
        return 1
    else:
        return len(s_object)


def _check_X(X):
    return isinstance(
        X, (cp.ndarray, cpx.scipy.sparse.csr_matrix, cpx.scipy.sparse.csc_matrix)
    )


def _get_vector(cudata, k, coldim, idxdim, layer=None):
    # cudata could be self if Raw and AnnData shared a parent
    dims = ("obs", "var")
    col = getattr(cudata, coldim).columns
    idx = getattr(cudata, f"{idxdim}_names")

    in_col = k in col
    in_idx = k in idx

    if (in_col + in_idx) == 2:
        raise ValueError(
            f"Key {k} could be found in both .{idxdim}_names and .{coldim}.columns"
        )
    elif (in_col + in_idx) == 0:
        raise KeyError(
            f"Could not find key {k} in .{idxdim}_names or .{coldim}.columns."
        )
    elif in_col:
        return getattr(cudata, coldim)[k].values
    elif in_idx:
        selected_dim = dims.index(idxdim)
        idx = cudata._normalize_indices(_make_slice(k, selected_dim))
        a = cudata._get_X(layer=layer)[idx]
    if issparse_gpu(a):
        a = a.toarray()
    return cp.asnumpy(a)


def _make_slice(idx, dimidx, n=2):
    mut = list(repeat(slice(None), n))
    mut[dimidx] = idx
    return tuple(mut)
