from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix

import rapids_singlecell as rsc

X_total = cp.array([[1, 0], [3, 0], [5, 6]], dtype=np.float64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_normalize_total(dtype, sparse):
    if sparse:
        X = csr_matrix(X_total, dtype=dtype)
    else:
        X = X_total.copy().astype(dtype)
    cudata = AnnData(X)

    rsc.pp.normalize_total(cudata, target_sum=1)
    cp.testing.assert_allclose(
        cp.ravel(cudata.X.sum(axis=1)), np.ones(cudata.shape[0], dtype=dtype)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalize_total_layers(dtype):
    cudata = AnnData(csr_matrix(X_total, dtype=dtype))
    cudata.layers["layer"] = cudata.X.copy()

    rsc.pp.normalize_total(cudata, target_sum=1, layer="layer")
    assert np.allclose(
        cudata.layers["layer"].sum(axis=1), np.ones(cudata.shape[0], dtype=dtype)
    )


@pytest.mark.parametrize(
    "sparsity_func", [cp.array, csr_matrix], ids=lambda x: x.__name__
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("theta", [0.01, 1.0, 100, np.Inf])
@pytest.mark.parametrize("clip", [None, 1.0, np.Inf])
def test_normalize_pearson_residuals_values(sparsity_func, dtype, theta, clip):
    # toy data
    X = cp.array([[3, 6], [2, 4], [1, 0]], dtype=dtype)
    ns = cp.sum(X, axis=1)
    ps = cp.sum(X, axis=0) / cp.sum(X)
    mu = cp.outer(ns, ps)

    # compute reference residuals
    if np.isinf(theta):
        # Poisson case
        residuals_reference = (X - mu) / cp.sqrt(mu)
    else:
        # NB case
        residuals_reference = (X - mu) / cp.sqrt(mu + mu**2 / theta)

    # compute output to test
    cudata = AnnData(X=sparsity_func(X, dtype=dtype))
    output_X = rsc.pp.normalize_pearson_residuals(
        cudata, theta=theta, clip=clip, inplace=False
    )

    rsc.pp.normalize_pearson_residuals(cudata, theta=theta, clip=clip, inplace=True)
    assert np.all(np.isin(["pearson_residuals_normalization"], list(cudata.uns.keys())))
    assert np.all(
        np.isin(
            ["theta", "clip", "computed_on"],
            list(cudata.uns["pearson_residuals_normalization"].keys()),
        )
    )

    # test against inplace
    cp.testing.assert_array_equal(cudata.X, output_X)
    if clip is None:
        # default clipping: compare to sqrt(n) threshold
        clipping_threshold = np.sqrt(cudata.shape[0]).astype(dtype)
        assert np.max(output_X) <= clipping_threshold
        assert np.min(output_X) >= -clipping_threshold
    elif np.isinf(clip):
        # no clipping: compare to raw residuals
        assert np.allclose(output_X, residuals_reference)
    else:
        # custom clipping: compare to custom threshold
        assert np.max(output_X) <= clip
        assert np.min(output_X) >= -clip
