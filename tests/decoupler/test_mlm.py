from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import statsmodels.api as sm

import rapids_singlecell.decoupler_gpu as dc


def test_fit(
    mat,
    adjmat,
):
    X, obs, var = mat
    n_features, n_fsets = adjmat.shape
    n_samples, _ = X.shape
    adjmat = np.column_stack((np.ones((n_features,)), adjmat))
    inv = np.linalg.inv(np.dot(adjmat.T, adjmat))
    df = n_features - n_fsets - 1
    X_cp = cp.array(X)
    adjmat_cp = cp.array(adjmat)
    coef, t = dc._method_mlm._fit(
        X=adjmat_cp,
        y=X_cp.T,
        inv=inv,
        df=df,
    )
    # Assert output shapes
    assert isinstance(coef, cp.ndarray)
    assert isinstance(t, cp.ndarray)
    print(coef.shape, t.shape)
    assert coef.shape == (n_samples, n_fsets)
    assert t.shape == (n_samples, n_fsets)


@pytest.mark.parametrize("tval", [True, False])
def test_func_mlm(
    mat,
    adjmat,
    tval,
):
    X, obs, var = mat
    X_cp = cp.array(X)
    adjmat_cp = cp.array(adjmat)
    dc_es, dc_pv = dc._method_mlm._func_mlm(mat=X_cp, adj=adjmat_cp, tval=tval)
    st_es, st_pv = np.zeros(dc_es.shape), np.zeros(dc_pv.shape)
    for i in range(st_es.shape[0]):
        y = X[i, :]
        x = sm.add_constant(adjmat)
        model = sm.OLS(y, x)
        res = model.fit()
        if tval:
            st_es[i, :] = res.tvalues[1:]
        else:
            st_es[i, :] = res.params[1:]
        st_pv[i, :] = res.pvalues[1:]
    assert cp.allclose(dc_es, st_es)
    assert cp.allclose(dc_pv, st_pv)
