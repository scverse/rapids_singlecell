from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k, pbmc3k_processed
from scipy import sparse

import rapids_singlecell as rsc

A_list = [
    [0, 0, 7, 0, 0],
    [8, 5, 0, 2, 0],
    [6, 0, 0, 2, 5],
    [0, 0, 0, 1, 0],
    [8, 8, 2, 1, 0],
    [0, 0, 0, 4, 5],
]

A_pca = np.array(
    [
        [-4.4783009, 5.55508466, 1.73111572, -0.06029139, 0.17292555],
        [5.4855141, -0.42651191, -0.74776055, -0.74532146, 0.74633582],
        [0.01161428, -4.0156662, 2.37252748, -1.33122372, -0.29044446],
        [-3.61934397, 0.48525412, -2.96861931, -1.16312545, -0.33230607],
        [7.14050048, 1.86330409, -0.05786325, 1.25045782, -0.50213107],
        [-4.53998399, -3.46146476, -0.32940009, 2.04950419, 0.20562023],
    ]
)

A_svd = np.array(
    [
        [-0.77034038, -2.00750922, 6.64603489, -0.39669256, -0.22212097],
        [-9.47135856, -0.6326006, -1.33787112, -0.24894361, -1.02044665],
        [-5.90007339, 4.99658727, 0.70712592, -2.15188849, 0.30430008],
        [-0.19132409, 0.42172251, 0.11169531, 0.50977966, -0.71637566],
        [-11.1286238, -2.73045559, 0.08040596, 1.06850585, 0.74173764],
        [-1.50180389, 5.56886849, 1.64034442, 2.24476032, -0.05109001],
    ]
)


def _pbmc3k_normalized() -> AnnData:
    pbmc = pbmc3k()
    pbmc.X = pbmc.X.astype("float64")  # For better accuracy
    sc.pp.filter_genes(pbmc, min_counts=1)
    sc.pp.log1p(pbmc)
    sc.pp.normalize_total(pbmc)
    sc.pp.highly_variable_genes(pbmc)
    return pbmc


@pytest.mark.parametrize("run_sparse", [True, False])
def test_pca_transform(run_sparse):
    A = np.array(A_list).astype("float32")
    if run_sparse:
        A = sparse.csr_matrix(A)
    A_pca_abs = np.abs(A_pca)
    A_svd_abs = np.abs(A_svd)

    adata = AnnData(A)

    rsc.pp.pca(adata, n_comps=4, zero_center=True)

    assert np.linalg.norm(A_pca_abs[:, :4] - np.abs(adata.obsm["X_pca"])) < 2e-05

    rsc.pp.pca(adata, n_comps=4, zero_center=False)
    assert np.linalg.norm(A_svd_abs[:, :4] - np.abs(adata.obsm["X_pca"])) < 2e-05


def test_pca_shapes():
    adata = AnnData(np.random.randn(30, 20))
    rsc.pp.pca(adata)
    assert adata.obsm["X_pca"].shape == (30, 19)

    adata = AnnData(np.random.randn(20, 30))
    rsc.pp.pca(adata)
    assert adata.obsm["X_pca"].shape == (20, 19)
    with pytest.raises(ValueError):
        rsc.pp.pca(adata, n_comps=100)


def test_pca_chunked():
    chunked = pbmc3k_processed()
    default = pbmc3k_processed()
    chunked.X = chunked.X.astype(np.float64)
    default.X = default.X.astype(np.float64)
    rsc.pp.pca(chunked, chunked=True, chunk_size=chunked.shape[0])
    rsc.pp.pca(default)

    np.testing.assert_allclose(
        np.abs(chunked.obsm["X_pca"]),
        np.abs(default.obsm["X_pca"]),
        rtol=1e-7,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        np.abs(chunked.varm["PCs"]), np.abs(default.varm["PCs"]), rtol=1e-7, atol=1e-6
    )
    np.testing.assert_allclose(
        np.abs(chunked.uns["pca"]["variance"]), np.abs(default.uns["pca"]["variance"])
    )

    np.testing.assert_allclose(
        np.abs(chunked.uns["pca"]["variance_ratio"]),
        np.abs(default.uns["pca"]["variance_ratio"]),
    )


def test_pca_reproducible():
    pbmc = pbmc3k_processed()
    pbmc.X = pbmc.X.astype(np.float32)
    rsc.tl.pca(pbmc)
    a = pbmc.obsm["X_pca"]
    rsc.tl.pca(pbmc)
    b = pbmc.obsm["X_pca"]
    np.array_equal(a, b)
    cpbmc = pbmc.copy()
    cpbmc.X = cp.array(cpbmc.X)
    rsc.tl.pca(cpbmc)
    c = pbmc.obsm["X_pca"]
    np.array_equal(a, c)


@pytest.mark.parametrize(
    ("zero_center", "rtol", "atol"), [(True, 1e-7, 1e-6), (False, 1e-5, 1e-5)]
)
def test_pca_sparse(zero_center, rtol, atol):
    sparse_ad = pbmc3k_processed()
    default = pbmc3k_processed()
    sparse_ad.X = sparse.csr_matrix(sparse_ad.X.astype(np.float64))
    default.X = default.X.astype(np.float64)
    rsc.pp.pca(sparse_ad)
    rsc.pp.pca(default)

    np.testing.assert_allclose(
        np.abs(sparse_ad.obsm["X_pca"]),
        np.abs(default.obsm["X_pca"]),
        rtol=1e-7,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        np.abs(sparse_ad.varm["PCs"]), np.abs(default.varm["PCs"]), rtol=1e-7, atol=1e-6
    )
    np.testing.assert_allclose(
        np.abs(sparse_ad.uns["pca"]["variance_ratio"]),
        np.abs(default.uns["pca"]["variance_ratio"]),
        rtol=rtol,
        atol=atol,
    )


def test_mask_length_error():
    """Check error for n_obs / mask length mismatch."""
    adata = AnnData(np.array(A_list).astype("float32"))
    mask_var = np.random.choice([True, False], adata.shape[1] + 1)
    with pytest.raises(
        ValueError, match=r"The shape of the mask do not match the data\."
    ):
        rsc.pp.pca(adata, mask_var=mask_var)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "array_type", ["array", cusparse.csr_matrix, cusparse.csc_matrix]
)
def test_mask_var_argument_equivalence(float_dtype, array_type):
    """Test if pca result is equal when given mask as boolarray vs string"""
    X = cp.random.random((100, 10), dtype=float_dtype)
    if array_type != "array":
        X = array_type(X)
    adata_base = AnnData(X)
    mask_var = np.random.choice([True, False], adata_base.shape[1])

    adata = adata_base.copy()
    rsc.pp.pca(adata, mask_var=mask_var, dtype=float_dtype)

    adata_w_mask = adata_base.copy()
    adata_w_mask.var["mask"] = mask_var
    rsc.pp.pca(adata_w_mask, mask_var="mask", dtype=float_dtype)

    assert cp.allclose(
        adata.X.toarray() if cusparse.issparse(adata.X) else adata.X,
        adata_w_mask.X.toarray()
        if cusparse.issparse(adata_w_mask.X)
        else adata_w_mask.X,
    )


def test_mask():
    adata = sc.datasets.blobs(n_variables=10, n_centers=3, n_observations=100)
    mask_var = np.random.choice([True, False], adata.shape[1])

    adata_masked = adata[:, mask_var].copy()
    rsc.pp.pca(adata, mask_var=mask_var)
    rsc.pp.pca(adata_masked)

    masked_var_loadings = adata.varm["PCs"][~mask_var]
    np.testing.assert_equal(masked_var_loadings, np.zeros_like(masked_var_loadings))

    np.testing.assert_equal(adata.obsm["X_pca"], adata_masked.obsm["X_pca"])
    # There are slight difference based on whether the matrix was column or row major
    np.testing.assert_allclose(
        adata.varm["PCs"][mask_var], adata_masked.varm["PCs"], rtol=1e-11
    )


@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
def test_mask_defaults(float_dtype):
    """
    Test if pca result is equal without highly variable and with-but mask is None
    and if pca takes highly variable as mask as default
    """
    A = cp.array(A_list).astype("float32")
    adata = AnnData(A)

    without_var = rsc.pp.pca(adata, copy=True, dtype=float_dtype)

    mask = np.array([True, True, False, True, False])

    adata.var["highly_variable"] = mask
    with_var = rsc.pp.pca(adata, copy=True, dtype=float_dtype)
    assert without_var.uns["pca"]["params"]["mask_var"] is None
    assert with_var.uns["pca"]["params"]["mask_var"] == "highly_variable"
    assert not np.array_equal(without_var.obsm["X_pca"], with_var.obsm["X_pca"])
    with_no_mask = rsc.pp.pca(adata, mask_var=None, copy=True, dtype=float_dtype)
    assert np.array_equal(without_var.obsm["X_pca"], with_no_mask.obsm["X_pca"])


def test_pca_layer():
    """
    Tests that layers works the same way as .X
    """
    X_adata = _pbmc3k_normalized()
    X_adata.X = X_adata.X.astype(np.float64)

    layer_adata = X_adata.copy()
    layer_adata.layers["counts"] = X_adata.X.copy()
    del layer_adata.X

    rsc.pp.pca(X_adata)
    rsc.pp.pca(layer_adata, layer="counts")

    assert layer_adata.uns["pca"]["params"]["layer"] == "counts"
    assert "layer" not in X_adata.uns["pca"]["params"]

    np.testing.assert_almost_equal(
        X_adata.uns["pca"]["variance"], layer_adata.uns["pca"]["variance"]
    )
    np.testing.assert_almost_equal(
        X_adata.uns["pca"]["variance_ratio"], layer_adata.uns["pca"]["variance_ratio"]
    )
    np.testing.assert_almost_equal(X_adata.obsm["X_pca"], layer_adata.obsm["X_pca"])
    np.testing.assert_almost_equal(X_adata.varm["PCs"], layer_adata.varm["PCs"])


def test_pca_layer_mask():
    adata = sc.datasets.pbmc3k()[:, 1000].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    with pytest.raises(
        ValueError,
        match="There are genes with zero expression. Please remove them before running PCA.",
    ):
        rsc.pp.pca(adata)


# =============================================================================
# Tests for sparse matrix SVD solvers
# =============================================================================


class TestSparseSVDSolvers:
    """Test all sparse SVD solvers through the PCA interface."""

    @pytest.fixture
    def sparse_adata(self):
        """Create a sparse test dataset."""
        rng = np.random.RandomState(42)
        X = sparse.random(500, 100, density=0.2, random_state=rng, format="csr")
        X = X.astype(np.float64)
        return AnnData(X)

    @pytest.fixture
    def pbmc_sparse(self):
        """Get pbmc3k_processed with sparse matrix."""
        pbmc = pbmc3k_processed()
        pbmc.X = sparse.csr_matrix(pbmc.X.astype(np.float64))
        return pbmc

    @pytest.mark.parametrize(
        "svd_solver", ["lanczos", "randomized", "block_krylov", "covariance_eigh"]
    )
    def test_solver_basic(self, sparse_adata, svd_solver):
        """Test that all solvers produce valid PCA output."""
        adata = sparse_adata.copy()
        rsc.pp.pca(adata, n_comps=20, svd_solver=svd_solver, random_state=0)

        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (500, 20)
        assert "pca" in adata.uns
        assert "variance" in adata.uns["pca"]
        assert "variance_ratio" in adata.uns["pca"]
        assert len(adata.uns["pca"]["variance"]) == 20
        # Variance should be positive and decreasing
        var = adata.uns["pca"]["variance"]
        assert np.all(var > 0)
        assert np.all(var[:-1] >= var[1:])

    @pytest.mark.parametrize(
        "svd_solver", ["lanczos", "randomized", "block_krylov", "covariance_eigh"]
    )
    def test_solver_zero_center_false(self, sparse_adata, svd_solver):
        """Test solvers with zero_center=False (truncated SVD mode)."""
        adata = sparse_adata.copy()
        rsc.pp.pca(
            adata, n_comps=10, svd_solver=svd_solver, zero_center=False, random_state=0
        )

        assert adata.obsm["X_pca"].shape == (500, 10)
        assert adata.uns["pca"]["params"]["zero_center"] is False

    @pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "block_krylov"])
    def test_solver_reproducibility(self, sparse_adata, svd_solver):
        """Test that solvers produce reproducible results with same random_state."""
        adata1 = sparse_adata.copy()
        adata2 = sparse_adata.copy()

        rsc.pp.pca(adata1, n_comps=10, svd_solver=svd_solver, random_state=42)
        rsc.pp.pca(adata2, n_comps=10, svd_solver=svd_solver, random_state=42)

        X_pca1 = adata1.obsm["X_pca"]
        X_pca2 = adata2.obsm["X_pca"]
        if hasattr(X_pca1, "get"):
            X_pca1 = X_pca1.get()
        if hasattr(X_pca2, "get"):
            X_pca2 = X_pca2.get()

        np.testing.assert_allclose(X_pca1, X_pca2, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            adata1.uns["pca"]["variance"],
            adata2.uns["pca"]["variance"],
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "block_krylov"])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_solver_dtypes(self, svd_solver, dtype):
        """Test solvers with different data types."""
        rng = np.random.RandomState(42)
        X = sparse.random(100, 50, density=0.3, random_state=rng, format="csr")
        X = X.astype(dtype)
        adata = AnnData(X)

        rsc.pp.pca(adata, svd_solver=svd_solver, n_comps=10, dtype=dtype)

        assert adata.obsm["X_pca"].dtype == np.dtype(dtype)

    @pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "block_krylov"])
    def test_solver_cupy_sparse_input(self, svd_solver):
        """Test solvers with CuPy sparse matrix input."""
        rng = np.random.RandomState(42)
        X_np = sparse.random(200, 100, density=0.2, random_state=rng, format="csr")
        X_cp = cusparse.csr_matrix(X_np.astype(np.float64))
        adata = AnnData(X_cp)

        rsc.pp.pca(adata, svd_solver=svd_solver, n_comps=20, random_state=0)

        assert adata.obsm["X_pca"].shape == (200, 20)

    def test_lanczos_accuracy(self, pbmc_sparse):
        """Test that Lanczos solver matches covariance_eigh accurately."""
        ref = pbmc_sparse.copy()
        rsc.pp.pca(ref, svd_solver="covariance_eigh", n_comps=30)

        test = pbmc_sparse.copy()
        rsc.pp.pca(test, svd_solver="lanczos", random_state=0, n_comps=30)

        ref_pca = ref.obsm["X_pca"]
        test_pca = test.obsm["X_pca"]
        if hasattr(ref_pca, "get"):
            ref_pca = ref_pca.get()
        if hasattr(test_pca, "get"):
            test_pca = test_pca.get()

        # Lanczos should match very closely (allowing for sign differences)
        np.testing.assert_allclose(
            np.abs(test_pca[:, :20]),
            np.abs(ref_pca[:, :20]),
            rtol=0.05,
            atol=0.05,
        )

        # Variance ratios should be very close
        np.testing.assert_allclose(
            test.uns["pca"]["variance_ratio"][:20],
            ref.uns["pca"]["variance_ratio"][:20],
            rtol=0.02,
            atol=0.01,
        )

    @pytest.mark.parametrize("svd_solver", ["randomized", "block_krylov"])
    def test_approximate_solver_variance(self, pbmc_sparse, svd_solver):
        """Test that approximate solvers capture reasonable variance."""
        ref = pbmc_sparse.copy()
        rsc.pp.pca(ref, svd_solver="covariance_eigh", n_comps=30)

        test = pbmc_sparse.copy()
        rsc.pp.pca(test, svd_solver=svd_solver, random_state=0, n_comps=30)

        # Total variance captured should be similar (within 20%)
        ref_total_var = ref.uns["pca"]["variance_ratio"].sum()
        test_total_var = test.uns["pca"]["variance_ratio"].sum()
        assert test_total_var > ref_total_var * 0.8, (
            f"{svd_solver} captured much less variance: {test_total_var:.3f} vs {ref_total_var:.3f}"
        )

        # First few components should capture similar variance
        np.testing.assert_allclose(
            test.uns["pca"]["variance_ratio"][:5],
            ref.uns["pca"]["variance_ratio"][:5],
            rtol=0.15,
            atol=0.02,
            err_msg=f"{svd_solver} variance ratio mismatch for top 5 components",
        )

    @pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "block_krylov"])
    def test_solver_with_mask(self, svd_solver):
        """Test solvers with variable mask."""
        adata = sc.datasets.blobs(n_variables=30, n_centers=3, n_observations=100)
        adata.X = sparse.csr_matrix(adata.X.astype(np.float64))
        mask_var = np.zeros(adata.shape[1], dtype=bool)
        mask_var[:20] = True  # Use first 20 variables

        adata_masked = adata[:, mask_var].copy()
        rsc.pp.pca(adata, svd_solver=svd_solver, mask_var=mask_var, n_comps=5)
        rsc.pp.pca(adata_masked, svd_solver=svd_solver, n_comps=5)

        # Masked variables should have zero loadings
        masked_var_loadings = adata.varm["PCs"][~mask_var]
        np.testing.assert_equal(masked_var_loadings, np.zeros_like(masked_var_loadings))

        # Results should match
        X_pca = adata.obsm["X_pca"]
        X_pca_masked = adata_masked.obsm["X_pca"]
        if hasattr(X_pca, "get"):
            X_pca = X_pca.get()
        if hasattr(X_pca_masked, "get"):
            X_pca_masked = X_pca_masked.get()

        np.testing.assert_allclose(
            np.abs(X_pca),
            np.abs(X_pca_masked),
            rtol=1e-4,
            atol=1e-4,
        )


class TestMeanCenteredOperator:
    """Test the mean-centered sparse matrix operator."""

    def test_mean_centered_matvec(self):
        """Test that mean-centered matvec is computed correctly."""
        from rapids_singlecell.preprocessing._sparse_pca._operators import (
            MeanCenteredOperator,
        )

        rng = np.random.RandomState(42)
        m, n = 50, 20
        X_np = rng.randn(m, n).astype(np.float64)
        X_sparse = cusparse.csr_matrix(cp.array(X_np))
        mean = cp.array(X_np.mean(axis=0))

        wrapper = MeanCenteredOperator(X_sparse, mean)
        v = cp.array(rng.randn(n).astype(np.float64))

        # Compute using wrapper
        result_wrapper = wrapper.dot(v)

        # Compute reference (X - mean) @ v
        X_centered = X_np - mean.get()
        result_ref = X_centered @ v.get()

        np.testing.assert_allclose(
            cp.asnumpy(result_wrapper), result_ref, rtol=1e-10, atol=1e-10
        )

    def test_mean_centered_transpose_matvec(self):
        """Test that transpose matvec is computed correctly."""
        from rapids_singlecell.preprocessing._sparse_pca._operators import (
            MeanCenteredOperator,
        )

        rng = np.random.RandomState(42)
        m, n = 50, 20
        X_np = rng.randn(m, n).astype(np.float64)
        X_sparse = cusparse.csr_matrix(cp.array(X_np))
        mean = cp.array(X_np.mean(axis=0))

        wrapper = MeanCenteredOperator(X_sparse, mean)
        v = cp.array(rng.randn(m).astype(np.float64))

        # Compute using wrapper transpose
        result_wrapper = wrapper.T.dot(v)

        # Compute reference (X - mean).T @ v
        X_centered = X_np - mean.get()
        result_ref = X_centered.T @ v.get()

        np.testing.assert_allclose(
            cp.asnumpy(result_wrapper), result_ref, rtol=1e-10, atol=1e-10
        )

    def test_mean_centered_matrix_multiply(self):
        """Test mean-centered operator with matrix (not vector) input."""
        from rapids_singlecell.preprocessing._sparse_pca._operators import (
            MeanCenteredOperator,
        )

        rng = np.random.RandomState(42)
        m, n, k = 50, 20, 5
        X_np = rng.randn(m, n).astype(np.float64)
        X_sparse = cusparse.csr_matrix(cp.array(X_np))
        mean = cp.array(X_np.mean(axis=0))

        wrapper = MeanCenteredOperator(X_sparse, mean)
        V = cp.array(rng.randn(n, k).astype(np.float64))

        # Compute using wrapper
        result_wrapper = wrapper.dot(V)

        # Compute reference (X - mean) @ V
        X_centered = X_np - mean.get()
        result_ref = X_centered @ V.get()

        np.testing.assert_allclose(
            cp.asnumpy(result_wrapper), result_ref, rtol=1e-10, atol=1e-10
        )
