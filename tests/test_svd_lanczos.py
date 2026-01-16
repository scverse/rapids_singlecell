"""Tests for the Lanczos SVD solver."""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cps
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scanpy.datasets import pbmc3k_processed
from scipy import sparse

import rapids_singlecell as rsc


class TestGpuSparseSvds:
    """Test the gpu_sparse_svds function directly."""

    def test_basic_svd(self):
        """Test basic SVD computation on a random sparse matrix."""
        from rapids_singlecell.preprocessing._sparse_pca._svd_lanczos import (
            gpu_sparse_svds,
        )

        # Create a random sparse matrix
        rng = np.random.RandomState(42)
        m, n = 500, 100
        density = 0.1
        A_np = sparse.random(m, n, density=density, random_state=rng, format="csr")
        A_cp = cps.csr_matrix(A_np.astype(np.float64))

        k = 10
        U, S, Vt = gpu_sparse_svds(A_cp, k=k, random_state=42)

        # Check shapes
        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

        # Check that singular values are sorted descending
        assert cp.all(S[:-1] >= S[1:])

        # Check orthonormality of U
        U_orth = cp.dot(U.T, U)
        assert cp.allclose(U_orth, cp.eye(k), atol=1e-5)

        # Check orthonormality of Vt
        Vt_orth = cp.dot(Vt, Vt.T)
        assert cp.allclose(Vt_orth, cp.eye(k), atol=1e-5)

    def test_svd_accuracy(self):
        """Test SVD accuracy by comparing with CuPy's dense SVD."""
        from rapids_singlecell.preprocessing._sparse_pca._svd_lanczos import (
            gpu_sparse_svds,
        )

        # Create a small dense matrix for comparison
        rng = np.random.RandomState(123)
        m, n = 100, 50
        A_np = rng.randn(m, n).astype(np.float64)
        A_dense = cp.array(A_np)
        A_sparse = cps.csr_matrix(A_dense)

        k = 5
        U_lanczos, S_lanczos, Vt_lanczos = gpu_sparse_svds(
            A_sparse, k=k, random_state=0
        )

        # Reference SVD using CuPy dense
        U_ref, S_ref, Vt_ref = cp.linalg.svd(A_dense, full_matrices=False)
        U_ref = U_ref[:, :k]
        S_ref = S_ref[:k]
        Vt_ref = Vt_ref[:k, :]

        # Compare singular values (should be very close)
        np.testing.assert_allclose(
            cp.asnumpy(S_lanczos), cp.asnumpy(S_ref), rtol=1e-4, atol=1e-6
        )

        # Compare singular vectors (up to sign)
        np.testing.assert_allclose(
            cp.asnumpy(cp.abs(U_lanczos)),
            cp.asnumpy(cp.abs(U_ref)),
            rtol=1e-3,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            cp.asnumpy(cp.abs(Vt_lanczos)),
            cp.asnumpy(cp.abs(Vt_ref)),
            rtol=1e-3,
            atol=1e-4,
        )

    def test_svd_reproducibility(self):
        """Test that results are reproducible with same random_state."""
        from rapids_singlecell.preprocessing._sparse_pca._svd_lanczos import (
            gpu_sparse_svds,
        )

        rng = np.random.RandomState(42)
        m, n = 200, 50
        A_np = sparse.random(m, n, density=0.1, random_state=rng, format="csr")
        A_cp = cps.csr_matrix(A_np.astype(np.float64))

        k = 5
        U1, S1, Vt1 = gpu_sparse_svds(A_cp, k=k, random_state=42)
        U2, S2, Vt2 = gpu_sparse_svds(A_cp, k=k, random_state=42)

        # Use allclose for floating-point comparison (GPU ops may have tiny differences)
        np.testing.assert_allclose(
            cp.asnumpy(S1), cp.asnumpy(S2), rtol=1e-10, atol=1e-14
        )
        np.testing.assert_allclose(
            cp.asnumpy(U1), cp.asnumpy(U2), rtol=1e-10, atol=1e-14
        )
        np.testing.assert_allclose(
            cp.asnumpy(Vt1), cp.asnumpy(Vt2), rtol=1e-10, atol=1e-14
        )


class TestPCALanczosSolver:
    """Test PCA with the Lanczos SVD solver."""

    def test_pca_lanczos_basic(self):
        """Test basic PCA computation with Lanczos solver."""
        A = np.array(
            [
                [0, 0, 7, 0, 0],
                [8, 5, 0, 2, 0],
                [6, 0, 0, 2, 5],
                [0, 0, 0, 1, 0],
                [8, 8, 2, 1, 0],
                [0, 0, 0, 4, 5],
            ]
        ).astype("float64")
        A_sparse = sparse.csr_matrix(A)
        adata = AnnData(A_sparse)

        rsc.pp.pca(adata, n_comps=4, zero_center=True, svd_solver="lanczos")

        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (6, 4)
        assert "pca" in adata.uns
        assert "variance" in adata.uns["pca"]
        assert "variance_ratio" in adata.uns["pca"]

    def test_pca_lanczos_vs_covariance_eigh(self):
        """Compare Lanczos solver results with covariance_eigh solver."""
        pbmc_lanczos = pbmc3k_processed()
        pbmc_eigh = pbmc3k_processed()
        pbmc_lanczos.X = sparse.csr_matrix(pbmc_lanczos.X.astype(np.float64))
        pbmc_eigh.X = sparse.csr_matrix(pbmc_eigh.X.astype(np.float64))

        rsc.pp.pca(pbmc_lanczos, svd_solver="lanczos", random_state=0)
        rsc.pp.pca(pbmc_eigh, svd_solver="covariance_eigh")

        # Convert to numpy if needed
        X_pca_lanczos = pbmc_lanczos.obsm["X_pca"]
        X_pca_eigh = pbmc_eigh.obsm["X_pca"]
        if hasattr(X_pca_lanczos, "get"):
            X_pca_lanczos = X_pca_lanczos.get()
        if hasattr(X_pca_eigh, "get"):
            X_pca_eigh = X_pca_eigh.get()

        # Compare PCA results (allowing for sign differences and numerical differences)
        # Different algorithms may produce slightly different results, especially
        # for small eigenvalues. Use relative tolerance for main comparison.
        np.testing.assert_allclose(
            np.abs(X_pca_lanczos),
            np.abs(X_pca_eigh),
            rtol=2e-2,
            atol=2e-2,
        )

        # Compare variance ratios (should be very close for top components)
        np.testing.assert_allclose(
            pbmc_lanczos.uns["pca"]["variance_ratio"],
            pbmc_eigh.uns["pca"]["variance_ratio"],
            rtol=2e-2,
            atol=1e-3,
        )

    def test_pca_lanczos_uncentered(self):
        """Test Lanczos solver with zero_center=False (truncated SVD mode)."""
        pbmc = pbmc3k_processed()
        pbmc.X = sparse.csr_matrix(pbmc.X.astype(np.float64))

        rsc.pp.pca(pbmc, svd_solver="lanczos", zero_center=False, n_comps=10)

        assert pbmc.obsm["X_pca"].shape == (pbmc.n_obs, 10)
        assert pbmc.uns["pca"]["params"]["zero_center"] is False

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_pca_lanczos_dtypes(self, dtype):
        """Test Lanczos solver with different data types."""
        rng = np.random.RandomState(42)
        X = sparse.random(100, 50, density=0.3, random_state=rng, format="csr")
        X = X.astype(dtype)
        adata = AnnData(X)

        rsc.pp.pca(adata, svd_solver="lanczos", n_comps=10, dtype=dtype)

        assert adata.obsm["X_pca"].dtype == np.dtype(dtype)

    def test_pca_lanczos_cupy_sparse(self):
        """Test Lanczos solver with CuPy sparse matrix input."""
        rng = np.random.RandomState(42)
        X_np = sparse.random(200, 100, density=0.2, random_state=rng, format="csr")
        X_cp = cps.csr_matrix(X_np.astype(np.float64))
        adata = AnnData(X_cp)

        rsc.pp.pca(adata, svd_solver="lanczos", n_comps=20)

        assert adata.obsm["X_pca"].shape == (200, 20)

    def test_pca_lanczos_reproducibility(self):
        """Test reproducibility of Lanczos PCA with same random_state."""
        pbmc1 = pbmc3k_processed()
        pbmc2 = pbmc3k_processed()
        pbmc1.X = sparse.csr_matrix(pbmc1.X.astype(np.float64))
        pbmc2.X = sparse.csr_matrix(pbmc2.X.astype(np.float64))

        rsc.pp.pca(pbmc1, svd_solver="lanczos", random_state=42)
        rsc.pp.pca(pbmc2, svd_solver="lanczos", random_state=42)

        # Convert to numpy if needed
        X_pca1 = pbmc1.obsm["X_pca"]
        X_pca2 = pbmc2.obsm["X_pca"]
        if hasattr(X_pca1, "get"):
            X_pca1 = X_pca1.get()
        if hasattr(X_pca2, "get"):
            X_pca2 = X_pca2.get()

        # Use tolerance for floating-point comparison (GPU ops may have tiny differences)
        np.testing.assert_allclose(X_pca1, X_pca2, rtol=1e-5, atol=1e-6)

    def test_pca_lanczos_with_mask(self):
        """Test Lanczos solver with variable mask."""
        adata = sc.datasets.blobs(n_variables=20, n_centers=3, n_observations=100)
        adata.X = sparse.csr_matrix(adata.X.astype(np.float64))
        mask_var = np.random.choice([True, False], adata.shape[1])

        # Ensure at least 10 variables are selected
        mask_var[:10] = True

        adata_masked = adata[:, mask_var].copy()
        rsc.pp.pca(adata, svd_solver="lanczos", mask_var=mask_var, n_comps=5)
        rsc.pp.pca(adata_masked, svd_solver="lanczos", n_comps=5)

        masked_var_loadings = adata.varm["PCs"][~mask_var]
        np.testing.assert_equal(masked_var_loadings, np.zeros_like(masked_var_loadings))

        # Convert to numpy if needed
        X_pca = adata.obsm["X_pca"]
        X_pca_masked = adata_masked.obsm["X_pca"]
        if hasattr(X_pca, "get"):
            X_pca = X_pca.get()
        if hasattr(X_pca_masked, "get"):
            X_pca_masked = X_pca_masked.get()

        np.testing.assert_allclose(
            np.abs(X_pca),
            np.abs(X_pca_masked),
            rtol=1e-5,
            atol=1e-5,
        )


class TestMeanCenteredSparseMatrix:
    """Test the mean-centered sparse matrix wrapper."""

    def test_mean_centered_matvec(self):
        """Test that mean-centered matvec is computed correctly."""
        from rapids_singlecell.preprocessing._sparse_pca._operators import (
            MeanCenteredOperator as _MeanCenteredSparseMatrix,
        )

        rng = np.random.RandomState(42)
        m, n = 50, 20
        X_np = rng.randn(m, n).astype(np.float64)
        X_sparse = cps.csr_matrix(cp.array(X_np))
        mean = cp.array(X_np.mean(axis=0))

        wrapper = _MeanCenteredSparseMatrix(X_sparse, mean)
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
            MeanCenteredOperator as _MeanCenteredSparseMatrix,
        )

        rng = np.random.RandomState(42)
        m, n = 50, 20
        X_np = rng.randn(m, n).astype(np.float64)
        X_sparse = cps.csr_matrix(cp.array(X_np))
        mean = cp.array(X_np.mean(axis=0))

        wrapper = _MeanCenteredSparseMatrix(X_sparse, mean)
        v = cp.array(rng.randn(m).astype(np.float64))

        # Compute using wrapper transpose
        result_wrapper = wrapper.T.dot(v)

        # Compute reference (X - mean).T @ v
        X_centered = X_np - mean.get()
        result_ref = X_centered.T @ v.get()

        np.testing.assert_allclose(
            cp.asnumpy(result_wrapper), result_ref, rtol=1e-10, atol=1e-10
        )
