from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from scipy.stats import pearsonr

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._harmony._helper import (
    _choose_colsum_algo,
    _get_colsum_func,
)


def _get_measure(x, base, norm):
    assert norm in ["r", "L2"]

    if norm == "r":
        corr, _ = pearsonr(x, base)
        return corr
    else:
        return np.linalg.norm(x - base) / np.linalg.norm(base)


@pytest.mark.parametrize("correction_method", ["fast", "original"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_harmony_integrate(correction_method, dtype):
    """
    Test that Harmony integrate works.

    This is a very simple test that just checks to see if the Harmony
    integrate wrapper successfully added a new field to ``adata.obsm``
    and makes sure it has the same dimensions as the original PCA table.
    """
    adata = sc.datasets.pbmc68k_reduced()
    rsc.pp.harmony_integrate(
        adata, "bulk_labels", correction_method=correction_method, dtype=dtype
    )
    assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_harmony_integrate_algos(dtype):
    """
    Test that Harmony integrate works.

    This is a very simple test that just checks to see if the Harmony
    integrate wrapper successfully added a new field to ``adata.obsm``
    and makes sure it has the same dimensions as the original PCA table.
    """
    adata = sc.datasets.pbmc68k_reduced()
    rsc.pp.harmony_integrate(
        adata, "bulk_labels", correction_method="fast", dtype=dtype
    )
    fast = adata.obsm["X_pca_harmony"].copy()
    rsc.pp.harmony_integrate(
        adata, "bulk_labels", correction_method="original", dtype=dtype
    )
    slow = adata.obsm["X_pca_harmony"].copy()
    assert _get_measure(fast, slow, "r").min() > 0.99
    assert _get_measure(fast, slow, "L2").max() < 0.1


@pytest.mark.parametrize("algo", ["columns", "atomics", "gemm"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64, cp.int32])
def test_colsum_algo(algo, dtype):
    # Int32 testing for correctness of the algorithm
    if dtype == cp.int32:
        X = cp.random.randint(0, 10, size=(20, 10), dtype=dtype)
    else:
        X = cp.random.randn(20, 10, dtype=dtype)
    algo_func = _get_colsum_func(X.shape[0], X.shape[1], algo)
    algo_out = algo_func(X)
    cupy_out = X.sum(axis=0)
    if dtype == cp.int32:
        cp.testing.assert_array_equal(algo_out, cupy_out)
    elif dtype == cp.float32:
        cp.testing.assert_allclose(algo_out, cupy_out, atol=1e-5)
    else:
        cp.testing.assert_allclose(algo_out, cupy_out)


@pytest.mark.parametrize("compute_capability", ["100", "80"])
def test_choose_colsum_algo(compute_capability):
    # Test that the choose_colsum_algo function returns the correct algorithm
    # for the given shape of the matrix
    for rows in np.arange(1000, 300000, 1000):
        for columns in np.arange(10, 5000, 50):
            algo = _choose_colsum_algo(rows, columns, compute_capability)
            assert algo in ["columns", "atomics", "gemm", "cupy"]
            if columns < 1024 or rows >= 5000:
                assert algo != "cupy"
