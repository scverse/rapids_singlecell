import numpy as np
import cupy as cp
from anndata import AnnData
import rapids_singlecell as rsc
from rapids_singlecell.cunnData import cunnData
import pytest
from scipy.sparse import csr_matrix

X_total = np.array([[1, 0], [3, 0], [5, 6]])


def test_normalize_total():
    adata = AnnData(X_total, dtype=np.float32)
    cudata = cunnData(adata)
    rsc.pp.normalize_total(cudata, target_sum=1)
    cp.testing.assert_allclose(
        cp.ravel(cudata.X.sum(axis=1)), np.ones(cudata.shape[0], dtype=np.float32)
    )


def test_normalize_total_layers():
    adata = AnnData(X_total)
    cudata = cunnData(adata)
    cudata.layers["layer"] = cudata.X.copy()

    rsc.pp.normalize_total(cudata, target_sum=1, layer="layer")
    assert np.allclose(
        cudata.layers["layer"].sum(axis=1), np.ones(cudata.shape[0], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "sparsity_func", [cp.array, csr_matrix], ids=lambda x: x.__name__
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("theta", [0.01, 1.0, 100, np.Inf])
@pytest.mark.parametrize("clip", [None, 1.0, np.Inf])
def test_normalize_pearson_residuals_values(sparsity_func, dtype, theta, clip):
    # toy data
    X = np.array([[3, 6], [2, 4], [1, 0]])
    ns = np.sum(X, axis=1)
    ps = np.sum(X, axis=0) / np.sum(X)
    mu = np.outer(ns, ps)

    # compute reference residuals
    if np.isinf(theta):
        # Poisson case
        residuals_reference = (X - mu) / np.sqrt(mu)
    else:
        # NB case
        residuals_reference = (X - mu) / np.sqrt(mu + mu**2 / theta)

    # compute output to test
    cudata = cunnData(X=sparsity_func(X, dtype=cp.float32))
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
        clipping_threshold = np.sqrt(cudata.shape[0]).astype(np.float32)
        assert np.max(output_X) <= clipping_threshold
        assert np.min(output_X) >= -clipping_threshold
    elif np.isinf(clip):
        # no clipping: compare to raw residuals
        assert np.allclose(output_X, residuals_reference)
    else:
        # custom clipping: compare to custom threshold
        assert np.max(output_X) <= clip
        assert np.min(output_X) >= -clip
