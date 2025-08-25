from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as csps
import numpy as np
import scipy.sparse as sps

import rapids_singlecell.decoupler_gpu as dc


def test_extract(
    adata,
):
    data = [adata.X, adata.obs_names, adata.var_names]
    X, obs, var = dc._helper._data.extract(data)
    assert X.shape[0] == obs.size
    assert X.shape[1] == var.size
    X, obs, var = dc._helper._data.extract(adata.to_df())
    assert X.shape[0] == obs.size
    assert X.shape[1] == var.size
    X, obs, var = dc._helper._data.extract(adata)
    assert X.shape[0] == obs.size
    assert X.shape[1] == var.size
    adata.layers["counts"] = adata.X.round()
    X, obs, var = dc._helper._data.extract(adata, layer="counts")
    assert float(np.sum(X)).is_integer()
    sadata = adata.copy()
    sadata.X = sps.csc_matrix(sadata.X)
    X, obs, var = dc._helper._data.extract(sadata)
    assert isinstance(X, sps.csr_matrix)
    X, obs, var = dc._helper._data.extract(sadata, pre_load=True)
    assert isinstance(X, csps.csr_matrix)
    X, obs, var = dc._helper._data.extract(adata, pre_load=True)
    assert isinstance(X, cp.ndarray)
    eadata = adata.copy()
    eadata.X[5, :] = 0.0
    X, obs, var = dc._helper._data.extract(eadata, empty=True)
    assert X.shape[0] < eadata.shape[0]
    nadata = adata.copy()
    nadata.X = nadata.X * -1
    adata.raw = nadata
    X, obs, var = dc._helper._data.extract(adata, raw=True)
    assert (X < 0).all()
