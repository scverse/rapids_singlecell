from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import read_h5ad
from scipy import sparse

from rapids_singlecell.gr import spatial_autocorr

MORAN_I = "moranI"
GEARY_C = "gearyC"


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_consistency(mode):
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)

    spatial_autocorr(dummy_adata, mode=mode)
    df1 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=50)
    df2 = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=50)

    idx_df = df1.index.values
    idx_adata = dummy_adata[:, dummy_adata.var.highly_variable.values].var_names.values

    if mode == "moran":
        UNS_KEY = MORAN_I
    elif mode == "geary":
        UNS_KEY = GEARY_C

    assert UNS_KEY in dummy_adata.uns.keys()
    assert "pval_sim_fdr_bh" in df1
    assert "pval_norm_fdr_bh" in dummy_adata.uns[UNS_KEY]
    assert dummy_adata.uns[UNS_KEY].columns.shape == (4,)
    assert df1.columns.shape == (9,)
    # test pval_norm same
    np.testing.assert_allclose(
        df1["pval_norm"].values, df2["pval_norm"].values, atol=1e-5, rtol=1e-5
    )
    # test highly variable
    assert dummy_adata.uns[UNS_KEY].shape != df1.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_autocorr_sparse(mode, dtype):
    """Test spatial_autocorr with sparse data and different dtypes."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)

    # Convert to sparse with specified dtype
    dummy_adata.X = sparse.csr_matrix(dummy_adata.X, dtype=dtype)

    df = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=None)

    stat_col = "I" if mode == "moran" else "C"
    assert stat_col in df.columns
    assert "pval_norm" in df.columns
    # Check no inf or nan values in the statistic
    assert not np.any(np.isinf(df[stat_col].values))
    # Some nan is expected for zero-variance genes, but not all
    assert not np.all(np.isnan(df[stat_col].values))


@pytest.mark.parametrize("mode", ["moran", "geary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_autocorr_dense(mode, dtype):
    """Test spatial_autocorr with dense data and different dtypes."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)

    # Convert to dense with specified dtype
    if sparse.issparse(dummy_adata.X):
        dummy_adata.X = dummy_adata.X.toarray().astype(dtype)
    else:
        dummy_adata.X = dummy_adata.X.astype(dtype)

    df = spatial_autocorr(dummy_adata, mode=mode, copy=True, n_perms=None)

    stat_col = "I" if mode == "moran" else "C"
    assert stat_col in df.columns
    assert "pval_norm" in df.columns
    # Check no inf or nan values in the statistic
    assert not np.any(np.isinf(df[stat_col].values))
    assert not np.all(np.isnan(df[stat_col].values))


@pytest.mark.parametrize("mode", ["moran", "geary"])
def test_autocorr_sparse_dense_consistency(mode):
    """Test that sparse and dense give consistent results."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    adata_dense = read_h5ad(file)
    adata_sparse = read_h5ad(file)

    # Use float64 for both
    adata_dense.X = adata_dense.X.astype(np.float64)
    adata_sparse.X = sparse.csr_matrix(adata_sparse.X, dtype=np.float64)

    df_dense = spatial_autocorr(adata_dense, mode=mode, copy=True, n_perms=None)
    df_sparse = spatial_autocorr(adata_sparse, mode=mode, copy=True, n_perms=None)

    stat_col = "I" if mode == "moran" else "C"

    # Results should be very close between sparse and dense
    np.testing.assert_allclose(
        df_dense[stat_col].values,
        df_sparse[stat_col].values,
        rtol=1e-5,
        atol=1e-5,
    )


def test_autocorr_dtype_parameter():
    """Test that the dtype parameter works correctly."""
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    adata = read_h5ad(file)

    # Input is float64, but force float32 computation
    adata.X = adata.X.astype(np.float64)

    df_f32 = spatial_autocorr(
        adata, mode="moran", copy=True, n_perms=None, dtype=np.float32
    )
    df_f64 = spatial_autocorr(
        adata, mode="moran", copy=True, n_perms=None, dtype=np.float64
    )

    # Both should produce valid results
    assert not np.any(np.isinf(df_f32["I"].values))
    assert not np.any(np.isinf(df_f64["I"].values))

    # Results should be close but not identical due to precision differences
    np.testing.assert_allclose(
        df_f32["I"].values,
        df_f64["I"].values,
        rtol=1e-4,
        atol=1e-4,
    )


def test_autocorr_float32_nan_raises_error():
    """Test that float32 nan/inf raises an error with helpful message."""
    from anndata import AnnData

    # Create data with a constant gene (zero variance) that causes nan in float32
    np.random.seed(42)
    n_cells = 100
    n_genes = 5

    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    # Make one gene constant - this will cause division by zero (nan)
    X[:, 0] = 1.0

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Create simple connectivity
    from sklearn.neighbors import kneighbors_graph

    positions = np.random.rand(n_cells, 2)
    adj = kneighbors_graph(positions, n_neighbors=5, mode="connectivity")
    adata.obsp["spatial_connectivities"] = adj

    # Run with float32 - should raise error suggesting float64
    with pytest.raises(ValueError, match="float64"):
        spatial_autocorr(adata, mode="moran", copy=True, n_perms=None)

    # With float64, should also raise error (for constant genes) but with bug report message
    with pytest.raises(ValueError, match="bug report"):
        spatial_autocorr(adata, mode="moran", copy=True, n_perms=None, dtype=np.float64)
