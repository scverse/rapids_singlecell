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
        # [lambda x, w: 0, 5, 1],
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


@pytest.mark.parametrize(
    "n_obs,n_var,n_src",
    [
        [5, 10, 8],
        [50, 10, 8],
        [100, 20, 16],
        # [500, 100, 64],
    ],
)


def test_wsum_wmean(mat, adjmat, n_obs, n_var, n_src):
    """Test _wsum and _wmean functions with matrix multiplication."""
    import rapids_singlecell.decoupler_gpu._method_waggr as waggr_module
    
    # Test with small matrices first
    print("\n=== Testing with small matrices ===")
    # A = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.float32)
    # B = cp.array([[1, 4], [2, 5], [3, 6]], dtype=cp.float32)
    cp.random.seed(42)
    A = cp.random.randn(n_obs, n_var, dtype=cp.float32)
    B = cp.random.randn(n_var, n_src, dtype=cp.float32)
    
    # print("Matrix A:")
    # print(A)
    # print("Matrix B:")
    # print(B)
    
    # CuPy matrix multiplication
    expected = A @ B
    # print("Expected result (CuPy @):")
    # print(expected)
    
    # Our custom kernel
    result = waggr_module._wsum(A, B)
    # print("Custom kernel result:")
    # print(result)
    
    # Check if they match
    is_close = cp.allclose(expected, result, rtol=1e-4, atol=1e-6)
    print(f"Results match: {is_close}")
    
    if not is_close:
        print("Difference:")
        print(expected - result)
        print("Max difference:", cp.max(cp.abs(expected - result)))
    
    # Test _wmean
    print("\n=== Testing _wmean ===")
    wmean_result = waggr_module._wmean(A, B)
    # print("Weighted mean result:")
    # print(wmean_result)
    
    # Manual calculation for verification
    div = cp.sum(cp.abs(B), axis=0)
    manual_wmean = expected / div
    # print("Manual weighted mean:")
    # print(manual_wmean)
    wmean_close = cp.allclose(wmean_result, manual_wmean, rtol=1e-4, atol=1e-6)
    print(f"wmean matches manual: {wmean_close}")
    
    # Assertions
    assert is_close, "Small matrix test failed"
    assert wmean_close, "Weighted mean test failed"

def test_wsum_wmean_actual(mat, adjmat):
    import rapids_singlecell.decoupler_gpu._method_waggr as waggr_module
    
    # Test with the actual test data
    print("\n=== Testing with actual test data ===")
    X, obs, var = mat
    X = cp.array(X, dtype=cp.float32)
    adjmat = cp.array(adjmat, dtype=cp.float32)
    
    print(f"X shape: {X.shape}, adjmat shape: {adjmat.shape}")
    print(f"X dtype: {X.dtype}, adjmat dtype: {adjmat.dtype}")
    print(f"X min/max: {cp.min(X):.6f} / {cp.max(X):.6f}")
    print(f"adjmat min/max: {cp.min(adjmat):.6f} / {cp.max(adjmat):.6f}")
    
    # Test _wsum with actual data
    result_actual = waggr_module._wsum(X, adjmat)
    expected_actual = X @ adjmat
    
    print(f"Expected shape: {expected_actual.shape}, Result shape: {result_actual.shape}")
    print(f"Expected min/max: {cp.min(expected_actual):.6f} / {cp.max(expected_actual):.6f}")
    print(f"Result min/max: {cp.min(result_actual):.6f} / {cp.max(result_actual):.6f}")
    
    is_close_actual = cp.allclose(expected_actual, result_actual, rtol=1e-4)
    print(f"Actual data results match: {is_close_actual}")
    
    if not is_close_actual:
        max_diff = cp.max(cp.abs(expected_actual - result_actual))
        print(f"Max difference: {max_diff}")
        print("First few elements of expected:")
        print(expected_actual[:3, :3])
        print("First few elements of result:")
        print(result_actual[:3, :3])
        
        # Check if it's a systematic issue
        print("\nChecking if it's a systematic scaling issue...")
        ratio = result_actual / (expected_actual + 1e-10)
        print(f"Ratio min/max: {cp.min(ratio):.6f} / {cp.max(ratio):.6f}")
        
        # Let's also check the grid configuration
        n_obs, n_var = X.shape
        n_var, n_src = adjmat.shape
        threads_per_block = (16, 16)  # Updated to match the new configuration
        grid_x = (n_src + threads_per_block[0] - 1) // threads_per_block[0]
        grid_y = (n_obs + threads_per_block[1] - 1) // threads_per_block[1]
        grid_size = (grid_x, grid_y)
        print(f"Grid config: n_obs={n_obs}, n_var={n_var}, n_src={n_src}")
        print(f"Grid size: {grid_size}, Threads per block: {threads_per_block}")
        print(f"Total threads: {grid_size[0] * grid_size[1] * threads_per_block[0] * threads_per_block[1]}")
        print(f"Output elements needed: {n_obs * n_src}")
    assert is_close_actual, "Actual data test failed"
