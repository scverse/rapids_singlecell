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


@pytest.mark.parametrize("svd_solver", ["covariance_eigh", "randomized"])
def test_pca_transform_sparse_solvers(svd_solver):
    """Test sparse SVD solvers against reference PCA values on small matrix."""
    A = sparse.csr_matrix(np.array(A_list).astype("float64"))
    A_pca_abs = np.abs(A_pca)

    adata = AnnData(A)
    rsc.pp.pca(
        adata, n_comps=4, zero_center=True, svd_solver=svd_solver, random_state=0
    )

    X_pca = adata.obsm["X_pca"]
    if hasattr(X_pca, "get"):
        X_pca = X_pca.get()

    assert np.linalg.norm(A_pca_abs[:, :4] - np.abs(X_pca)) < 2e-05


def test_pca_transform_sparse_block_krylov():
    """Test block_krylov sparse SVD solver (approximate method)."""
    rng = np.random.RandomState(42)
    X = sparse.random(200, 50, density=0.2, random_state=rng, format="csr")
    X = X.astype(np.float64)

    # Get reference from covariance_eigh
    ref = AnnData(X.copy())
    rsc.pp.pca(ref, n_comps=10, svd_solver="covariance_eigh")

    # Test block_krylov solver (approximate method)
    test = AnnData(X.copy())
    rsc.pp.pca(test, n_comps=10, svd_solver="block_krylov", random_state=0)

    # block_krylov is approximate; check variance capture is reasonable
    ref_var_total = np.sum(ref.uns["pca"]["variance_ratio"])
    test_var_total = np.sum(test.uns["pca"]["variance_ratio"])
    assert test_var_total > 0.95 * ref_var_total  # captures at least 95% of variance

    # Check first few variance ratios are in the right ballpark
    np.testing.assert_allclose(
        test.uns["pca"]["variance_ratio"][:3],
        ref.uns["pca"]["variance_ratio"][:3],
        rtol=0.05,
    )


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


@pytest.mark.parametrize(
    "svd_solver", ["lanczos", "randomized", "block_krylov", "covariance_eigh"]
)
def test_pca_sparse_solvers(svd_solver):
    """Test all sparse SVD solvers produce valid output."""
    rng = np.random.RandomState(42)
    X = sparse.random(200, 50, density=0.2, random_state=rng, format="csr")
    adata = AnnData(X.astype(np.float64))

    rsc.pp.pca(adata, n_comps=20, svd_solver=svd_solver, random_state=0)

    assert adata.obsm["X_pca"].shape == (200, 20)
    # Variance should be positive and decreasing
    var = adata.uns["pca"]["variance"]
    assert np.all(var > 0)
    assert np.all(var[:-1] >= var[1:])


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "block_krylov"])
def test_pca_sparse_solver_reproducibility(svd_solver):
    """Test sparse solvers are reproducible with same random_state."""
    rng = np.random.RandomState(42)
    X = sparse.random(200, 50, density=0.2, random_state=rng, format="csr")
    X = X.astype(np.float64)

    adata1 = AnnData(X.copy())
    adata2 = AnnData(X.copy())

    rsc.pp.pca(adata1, n_comps=10, svd_solver=svd_solver, random_state=42)
    rsc.pp.pca(adata2, n_comps=10, svd_solver=svd_solver, random_state=42)

    X_pca1 = adata1.obsm["X_pca"]
    X_pca2 = adata2.obsm["X_pca"]
    if hasattr(X_pca1, "get"):
        X_pca1 = X_pca1.get()
    if hasattr(X_pca2, "get"):
        X_pca2 = X_pca2.get()

    np.testing.assert_allclose(X_pca1, X_pca2, rtol=1e-5, atol=1e-6)


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


@pytest.mark.parametrize(
    "svd_solver", ["lanczos", "randomized", "block_krylov", "covariance_eigh"]
)
def test_pca_sparse_zero_center_false(svd_solver):
    """Test sparse solvers with zero_center=False."""
    rng = np.random.RandomState(42)
    X = sparse.random(200, 50, density=0.2, random_state=rng, format="csr")
    adata = AnnData(X.astype(np.float64))

    rsc.pp.pca(adata, n_comps=10, svd_solver=svd_solver, zero_center=False)

    assert adata.obsm["X_pca"].shape == (200, 10)
    assert adata.uns["pca"]["params"]["zero_center"] is False


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


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "block_krylov"])
def test_mask_sparse_solvers(svd_solver):
    """Test sparse solvers with variable mask."""
    adata = sc.datasets.blobs(n_variables=30, n_centers=3, n_observations=100)
    adata.X = sparse.csr_matrix(adata.X.astype(np.float64))
    mask_var = np.zeros(adata.shape[1], dtype=bool)
    mask_var[:20] = True

    adata_masked = adata[:, mask_var].copy()
    rsc.pp.pca(adata, svd_solver=svd_solver, mask_var=mask_var, n_comps=5)
    rsc.pp.pca(adata_masked, svd_solver=svd_solver, n_comps=5)

    masked_var_loadings = adata.varm["PCs"][~mask_var]
    np.testing.assert_equal(masked_var_loadings, np.zeros_like(masked_var_loadings))


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


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "block_krylov"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_pca_sparse_solver_dtypes(svd_solver, dtype):
    """Test sparse solvers with different dtypes."""
    rng = np.random.RandomState(42)
    X = sparse.random(100, 50, density=0.3, random_state=rng, format="csr")
    adata = AnnData(X.astype(dtype))

    rsc.pp.pca(adata, svd_solver=svd_solver, n_comps=10, dtype=dtype)

    assert adata.obsm["X_pca"].dtype == np.dtype(dtype)


@pytest.mark.parametrize("svd_solver", ["lanczos", "randomized", "block_krylov"])
def test_pca_cupy_sparse_input(svd_solver):
    """Test sparse solvers with CuPy sparse input."""
    rng = np.random.RandomState(42)
    X_np = sparse.random(200, 100, density=0.2, random_state=rng, format="csr")
    X_cp = cusparse.csr_matrix(X_np.astype(np.float64))
    adata = AnnData(X_cp)

    rsc.pp.pca(adata, svd_solver=svd_solver, n_comps=20)

    assert adata.obsm["X_pca"].shape == (200, 20)


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


def test_pca_lanczos_accuracy():
    """Test that Lanczos matches covariance_eigh closely with float64."""
    pbmc = pbmc3k_processed()
    pbmc.X = sparse.csr_matrix(pbmc.X.astype(np.float64))

    ref = pbmc.copy()
    rsc.pp.pca(ref, svd_solver="covariance_eigh", n_comps=10)

    test = pbmc.copy()
    rsc.pp.pca(test, svd_solver="lanczos", random_state=0, n_comps=10)

    ref_pca = ref.obsm["X_pca"]
    test_pca = test.obsm["X_pca"]
    if hasattr(ref_pca, "get"):
        ref_pca = ref_pca.get()
    if hasattr(test_pca, "get"):
        test_pca = test_pca.get()

    # First 7 components (well-separated singular values) should match at machine precision
    np.testing.assert_allclose(
        np.abs(test_pca[:, :7]),
        np.abs(ref_pca[:, :7]),
        rtol=1e-7,
        atol=1e-7,
    )

    # Variance ratios should match very closely for all components
    np.testing.assert_allclose(
        test.uns["pca"]["variance_ratio"],
        ref.uns["pca"]["variance_ratio"],
        rtol=1e-6,
        atol=1e-9,
    )
