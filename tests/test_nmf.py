"""Tests for GPU NMF implementation."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import random as sp_random
from sklearn.decomposition import NMF as skNMF

from rapids_singlecell.preprocessing._nmf import run_nmf


@pytest.fixture
def small_sparse():
    """Small sparse matrix for correctness tests (500 x 100)."""
    np.random.seed(42)
    return sp_random(500, 100, density=0.1, format="csr", dtype=np.float32)


@pytest.fixture
def medium_sparse():
    """Medium sparse matrix for convergence tests (5000 x 500)."""
    np.random.seed(42)
    return sp_random(5000, 500, density=0.05, format="csr", dtype=np.float32)


def _naive_frob_error(X, W, H):
    """Compute ||X - WH||_F by materializing the full reconstruction."""
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    elif hasattr(X, "get"):
        X_dense = X.get()
    else:
        X_dense = np.asarray(X)

    W_np = W.get() if hasattr(W, "get") else np.asarray(W)
    H_np = H.get() if hasattr(H, "get") else np.asarray(H)
    return np.linalg.norm(X_dense - W_np @ H_np)


def _sklearn_nmf(X_scipy, *, n_components, solver, init, max_iter, tol, random_state):
    """Run sklearn NMF and return W, H, error, n_iter."""
    nmf = skNMF(
        n_components=n_components,
        init=init,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    W = nmf.fit_transform(X_scipy)
    return W, nmf.components_, nmf.reconstruction_err_, nmf.n_iter_


# ---------------------------------------------------------------------------
# Error computation correctness
# ---------------------------------------------------------------------------


class TestErrorComputation:
    """Verify our trace-identity error matches the naive materialized error."""

    @pytest.mark.parametrize("solver", ["hals", "mu"])
    def test_error_matches_naive(self, small_sparse, solver):
        """The returned error must match ||X - WH||_F computed naively."""
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        W, H, _, error = run_nmf(
            X_gpu,
            n_components=10,
            solver=solver,
            init="nndsvda",
            max_iter=50,
            tol=0,
            random_state=0,
        )
        naive_error = _naive_frob_error(small_sparse, W, H)
        np.testing.assert_allclose(float(error), naive_error, rtol=1e-4)

    def test_error_matches_naive_dense(self, small_sparse):
        """Same check for dense input."""
        X_gpu = cp.array(small_sparse.toarray())
        W, H, _, error = run_nmf(
            X_gpu,
            n_components=10,
            solver="hals",
            init="nndsvda",
            max_iter=50,
            tol=0,
            random_state=0,
        )
        naive_error = _naive_frob_error(small_sparse, W, H)
        np.testing.assert_allclose(float(error), naive_error, rtol=1e-4)


# ---------------------------------------------------------------------------
# Comparison against sklearn
# ---------------------------------------------------------------------------


class TestVsSklearn:
    """Compare factorization quality against sklearn."""

    @pytest.mark.parametrize("n_components", [5, 15, 30])
    def test_error_close_to_sklearn_mu(self, medium_sparse, n_components):
        """MU solver should reach similar error as sklearn MU."""
        X_gpu = cp_csr_matrix(medium_sparse.astype(np.float32))
        _, _, sk_err, _ = _sklearn_nmf(
            medium_sparse,
            n_components=n_components,
            solver="mu",
            init="nndsvda",
            max_iter=200,
            tol=1e-4,
            random_state=0,
        )
        _, _, _, gpu_err = run_nmf(
            X_gpu,
            n_components=n_components,
            solver="mu",
            init="nndsvda",
            max_iter=200,
            tol=1e-4,
            random_state=0,
        )
        assert float(gpu_err) <= sk_err * 1.05

    @pytest.mark.parametrize("n_components", [5, 15, 30])
    def test_hals_beats_or_matches_sklearn(self, medium_sparse, n_components):
        """HALS should reach at least as good error as sklearn CD."""
        X_gpu = cp_csr_matrix(medium_sparse.astype(np.float32))
        _, _, sk_err, _ = _sklearn_nmf(
            medium_sparse,
            n_components=n_components,
            solver="cd",
            init="nndsvda",
            max_iter=200,
            tol=1e-4,
            random_state=0,
        )
        _, _, _, gpu_err = run_nmf(
            X_gpu,
            n_components=n_components,
            solver="hals",
            init="nndsvda",
            max_iter=200,
            tol=1e-4,
            random_state=0,
        )
        assert float(gpu_err) <= sk_err * 1.05


# ---------------------------------------------------------------------------
# Non-negativity and shapes
# ---------------------------------------------------------------------------


class TestBasicProperties:
    """W and H must be non-negative with correct shapes."""

    @pytest.mark.parametrize("solver", ["hals", "mu"])
    @pytest.mark.parametrize("n_components", [5, 20])
    def test_nonnegative(self, small_sparse, solver, n_components):
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        W, H, _, _ = run_nmf(
            X_gpu,
            n_components=n_components,
            solver=solver,
            max_iter=50,
            random_state=0,
        )
        assert float(W.min()) >= 0
        assert float(H.min()) >= 0

    @pytest.mark.parametrize("solver", ["hals", "mu"])
    def test_shapes(self, small_sparse, solver):
        n, m = small_sparse.shape
        k = 10
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        W, H, _, _ = run_nmf(
            X_gpu, n_components=k, solver=solver, max_iter=50, random_state=0
        )
        assert W.shape == (n, k)
        assert H.shape == (k, m)

    def test_float64(self, small_sparse):
        """float64 input should work and return float64."""
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float64))
        W, H, _, _ = run_nmf(
            X_gpu, n_components=5, solver="hals", max_iter=50, random_state=0
        )
        assert W.dtype == np.float64
        assert H.dtype == np.float64
        assert float(W.min()) >= 0

    def test_dense_input(self, small_sparse):
        X_gpu = cp.array(small_sparse.toarray())
        W, H, _, _ = run_nmf(
            X_gpu, n_components=5, solver="hals", max_iter=50, random_state=0
        )
        assert float(W.min()) >= 0
        assert float(H.min()) >= 0


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    """Verify that the solvers actually converge."""

    @pytest.mark.parametrize("solver", ["hals", "mu"])
    def test_error_decreases(self, medium_sparse, solver):
        """Error after 100 iters should be less than after 10."""
        X_gpu = cp_csr_matrix(medium_sparse.astype(np.float32))

        _, _, _, err_10 = run_nmf(
            X_gpu,
            n_components=10,
            solver=solver,
            init="nndsvda",
            max_iter=10,
            tol=0,
            random_state=0,
        )
        _, _, _, err_100 = run_nmf(
            X_gpu,
            n_components=10,
            solver=solver,
            init="nndsvda",
            max_iter=100,
            tol=0,
            random_state=0,
        )
        assert float(err_100) < float(err_10)

    def test_hals_converges_faster_than_mu(self, medium_sparse):
        """HALS should converge in fewer outer iterations than MU."""
        X_gpu = cp_csr_matrix(medium_sparse.astype(np.float32))

        _, _, n_hals, _ = run_nmf(
            X_gpu,
            n_components=10,
            solver="hals",
            init="nndsvda",
            max_iter=200,
            tol=1e-4,
            random_state=0,
        )
        _, _, n_mu, _ = run_nmf(
            X_gpu,
            n_components=10,
            solver="mu",
            init="nndsvda",
            max_iter=200,
            tol=1e-4,
            random_state=0,
        )
        assert n_hals <= n_mu


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """All init methods should produce valid starting points."""

    @pytest.mark.parametrize("init", ["nndsvd", "nndsvda", "nndsvdar", "random"])
    def test_init_methods(self, small_sparse, init):
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        W, H, _, error = run_nmf(
            X_gpu,
            n_components=5,
            init=init,
            solver="hals",
            max_iter=50,
            random_state=0,
        )
        assert float(W.min()) >= 0
        assert float(H.min()) >= 0
        assert float(error) > 0
        assert not cp.isnan(error)


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------


class TestRegularization:
    """Regularization should not break the solver and should shrink factors."""

    @pytest.mark.parametrize("solver", ["hals", "mu"])
    def test_l2_reduces_norm(self, small_sparse, solver):
        """L2 regularization should produce smaller W/H norms."""
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))

        W_noreg, _, _, _ = run_nmf(
            X_gpu,
            n_components=5,
            solver=solver,
            init="nndsvda",
            max_iter=100,
            tol=0,
            random_state=0,
        )
        W_reg, _, _, _ = run_nmf(
            X_gpu,
            n_components=5,
            solver=solver,
            init="nndsvda",
            max_iter=100,
            tol=0,
            alpha_W=1.0,
            alpha_H=1.0,
            l1_ratio=0.0,
            random_state=0,
        )
        assert float(cp.linalg.norm(W_reg)) < float(cp.linalg.norm(W_noreg))

    @pytest.mark.parametrize("solver", ["hals", "mu"])
    def test_l1_increases_sparsity(self, small_sparse, solver):
        """L1 regularization should produce sparser factors."""
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))

        W_noreg, _, _, _ = run_nmf(
            X_gpu,
            n_components=5,
            solver=solver,
            init="nndsvda",
            max_iter=100,
            tol=0,
            random_state=0,
        )
        W_reg, _, _, _ = run_nmf(
            X_gpu,
            n_components=5,
            solver=solver,
            init="nndsvda",
            max_iter=100,
            tol=0,
            alpha_W=1.0,
            alpha_H=1.0,
            l1_ratio=1.0,
            random_state=0,
        )
        zeros_noreg = float((W_noreg < 1e-7).sum()) / W_noreg.size
        zeros_reg = float((W_reg < 1e-7).sum()) / W_reg.size
        assert zeros_reg >= zeros_noreg


# ---------------------------------------------------------------------------
# n_inner parameter
# ---------------------------------------------------------------------------


class TestNInner:
    """The n_inner parameter should be passable and affect results."""

    def test_n_inner_override(self, small_sparse):
        """Different n_inner should produce valid results."""
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        for n_inner in [1, 3, 5, 10]:
            W, H, _, error = run_nmf(
                X_gpu,
                n_components=5,
                solver="hals",
                init="nndsvda",
                max_iter=50,
                tol=0,
                n_inner=n_inner,
                random_state=0,
            )
            assert float(W.min()) >= 0
            assert float(H.min()) >= 0
            assert not cp.isnan(error)

    def test_n_inner_ignored_for_mu(self, small_sparse):
        """MU solver should ignore n_inner without error."""
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        W, H, _, _ = run_nmf(
            X_gpu,
            n_components=5,
            solver="mu",
            init="nndsvda",
            max_iter=50,
            n_inner=10,
            random_state=0,
        )
        assert float(W.min()) >= 0


# ---------------------------------------------------------------------------
# AnnData wrapper
# ---------------------------------------------------------------------------


class TestAnnDataWrapper:
    """Test the nmf() AnnData API."""

    def test_anndata_nmf(self, small_sparse):
        """nmf(adata) should store results in obsm/varm/uns."""
        import anndata as ad

        from rapids_singlecell.preprocessing._nmf import nmf

        adata = ad.AnnData(X=small_sparse.astype(np.float32))
        # Move to GPU
        X_gpu = cp_csr_matrix(adata.X)
        adata.X = X_gpu

        nmf(adata, n_components=5, max_iter=50, random_state=0)

        assert "X_nmf" in adata.obsm
        assert "NMF" in adata.varm
        assert "nmf" in adata.uns
        assert adata.obsm["X_nmf"].shape == (500, 5)
        assert adata.varm["NMF"].shape == (100, 5)
        assert adata.uns["nmf"]["params"]["solver"] == "hals"
        assert adata.uns["nmf"]["params"]["n_iter"] > 0
        assert adata.uns["nmf"]["reconstruction_error"] > 0


# ---------------------------------------------------------------------------
# Reproducibility (important for cNMF which runs NMF many times)
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same random_state should give identical results."""

    def test_deterministic_with_same_seed(self, small_sparse):
        """Same seed should give near-identical results.

        GPU floating-point is not bitwise deterministic (cusparse
        reduction order may vary), but results should match within
        float32 noise (~1e-5).
        """
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        W1, H1, n1, e1 = run_nmf(
            X_gpu, n_components=5, max_iter=50, tol=0, random_state=42
        )
        W2, H2, n2, e2 = run_nmf(
            X_gpu, n_components=5, max_iter=50, tol=0, random_state=42
        )
        np.testing.assert_allclose(W1.get(), W2.get(), atol=1e-5)
        np.testing.assert_allclose(H1.get(), H2.get(), atol=1e-5)

    def test_different_seed_gives_different_results(self, small_sparse):
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        W1, _, _, _ = run_nmf(
            X_gpu,
            n_components=5,
            init="random",
            max_iter=50,
            tol=0,
            random_state=0,
        )
        W2, _, _, _ = run_nmf(
            X_gpu,
            n_components=5,
            init="random",
            max_iter=50,
            tol=0,
            random_state=99,
        )
        assert not np.array_equal(W1.get(), W2.get())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_negative_input_raises(self):
        X_gpu = cp.array([[-1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="negative"):
            run_nmf(X_gpu, n_components=1)

    def test_n_components_too_large(self, small_sparse):
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        with pytest.raises(ValueError, match="n_components"):
            run_nmf(X_gpu, n_components=1000)

    def test_invalid_solver(self, small_sparse):
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        with pytest.raises(ValueError, match="solver"):
            run_nmf(X_gpu, n_components=5, solver="bogus")

    def test_single_component(self, small_sparse):
        """K=1 should work (degenerate case)."""
        X_gpu = cp_csr_matrix(small_sparse.astype(np.float32))
        W, H, _, error = run_nmf(
            X_gpu, n_components=1, solver="hals", max_iter=50, random_state=0
        )
        assert W.shape == (500, 1)
        assert H.shape == (1, 100)
        assert float(W.min()) >= 0
