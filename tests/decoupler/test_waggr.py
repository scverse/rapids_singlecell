from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import rapids_singlecell.decoupler_gpu as dc


def test_funcs(rng):
    x = cp.array([1, 2, 3, 4], dtype=float)
    w = cp.array(rng.random(x.size))
    es = dc._method_waggr._wsum(x=x, w=w)
    assert isinstance(es, float)
    es = dc._method_waggr._wmean(x=x, w=w)
    assert isinstance(es, float)

@pytest.mark.parametrize(
    "fun,times,seed",
    [
        ["wmean", 10, 42],
        ["wsum", 5, 23],
        [lambda x, w: 0, 5, 1],
        ["wmean", 0, 42],
    ],
)
def test_func_waggr(
    mat,
    adjmat,
    fun,
    times,
    seed,
):
    X, obs, var = mat
    X = cp.array(X)
    adjmat = cp.array(adjmat)
    times = 0
    es, pv = dc._method_waggr._func_waggr(mat=X, adj=adjmat, fun=fun, times=times, seed=seed)
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()
