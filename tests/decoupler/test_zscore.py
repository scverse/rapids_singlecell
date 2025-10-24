from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import rapids_singlecell.decoupler_gpu as dc


@pytest.mark.parametrize("flavor", ["KSEA", "RoKAI"])
def test_func_zscore(
    mat,
    adjmat,
    flavor,
):
    X, obs, var = mat
    X_cp = cp.array(X)
    adjmat_cp = cp.array(adjmat)
    es, pv = dc._method_zscore._func_zscore(mat=X_cp, adj=adjmat_cp, flavor=flavor)
    assert np.isfinite(es).all()
    assert ((0 <= pv) & (pv <= 1)).all()
