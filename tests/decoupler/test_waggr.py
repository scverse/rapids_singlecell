from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import rapids_singlecell.decoupler_gpu as dc


def test_funcs(rng):
    x = cp.array([[1, 2, 3, 4]], dtype=float)
    w = cp.array([rng.random(x.size)], dtype=float)
    es = dc._method_waggr._wsum(x=x, w=w)
    assert isinstance(es, cp.ndarray)
    es = dc._method_waggr._wmean(x=x, w=w)
    assert isinstance(es, cp.ndarray)


def test_wsum_wmean(mat, adjmat):
    print("\n=== Testing with test data ===")
    X, obs, var = mat
    X = cp.array(X, dtype=cp.float32)
    adjmat = cp.array(adjmat, dtype=cp.float32)

    print(f"X shape: {X.shape}, adjmat shape: {adjmat.shape}")
    print(f"X dtype: {X.dtype}, adjmat dtype: {adjmat.dtype}")
    print(f"X min/max: {cp.min(X):.6f} / {cp.max(X):.6f}")
    print(f"adjmat min/max: {cp.min(adjmat):.6f} / {cp.max(adjmat):.6f}")

    # Test _wsum
    result_actual = dc._method_waggr._wsum(X, adjmat)
    expected_actual = X @ adjmat

    print(
        f"Expected shape: {expected_actual.shape}, Result shape: {result_actual.shape}"
    )
    print(
        f"Expected min/max: {cp.min(expected_actual):.6f} / {cp.max(expected_actual):.6f}"
    )
    print(f"Result min/max: {cp.min(result_actual):.6f} / {cp.max(result_actual):.6f}")

    is_close_actual = cp.allclose(expected_actual, result_actual, rtol=1e-4)
    print(f"wsum results match: {is_close_actual}")

    assert is_close_actual, "_wsum test failed"

    # Test _wmean
    result_wmean = dc._method_waggr._wmean(X, adjmat)
    div = cp.sum(cp.abs(adjmat), axis=0)
    expected_wmean = expected_actual / div
    is_close_actual = cp.allclose(expected_wmean, result_wmean, rtol=1e-4)
    print(f"wmean results match: {is_close_actual}")
    assert is_close_actual, "_wmean test failed"


@pytest.mark.parametrize(
    "fun,times,seed",
    [
        ["wmean", 10, 42],
        ["wsum", 5, 23],
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
    es, pv = dc._method_waggr._func_waggr(
        mat=X, adj=adjmat, fun=fun, times=times, seed=seed
    )
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()
