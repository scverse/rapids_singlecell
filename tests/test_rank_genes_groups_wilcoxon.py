from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

import rapids_singlecell as rsc


def _to_format(X_dense, fmt):
    """Convert dense numpy array to the specified format."""
    if fmt == "numpy_dense":
        return np.asarray(X_dense)
    if fmt == "scipy_csr":
        return sp.csr_matrix(X_dense)
    if fmt == "scipy_csc":
        return sp.csc_matrix(X_dense)
    if fmt == "cupy_dense":
        return cp.asarray(X_dense)
    if fmt == "cupy_csr":
        return cpsp.csr_matrix(cp.asarray(X_dense))
    if fmt == "cupy_csc":
        return cpsp.csc_matrix(cp.asarray(X_dense))
    raise ValueError(f"Unknown format: {fmt}")


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("tie_correct", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
def test_rank_genes_groups_wilcoxon_matches_scanpy(reference, tie_correct, sparse):
    """Test wilcoxon matches scanpy output across configurations."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")

    if sparse:
        adata_gpu.X = sp.csr_matrix(adata_gpu.X)

    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        n_genes=3,
        reference=reference,
        corr_method="benjamini-hochberg",
        tie_correct=tie_correct,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        n_genes=3,
        reference=reference,
        tie_correct=tie_correct,
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    assert gpu_result["names"].dtype.names == cpu_result["names"].dtype.names
    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        gpu_field = gpu_result[field]
        cpu_field = cpu_result[field]
        assert gpu_field.dtype.names == cpu_field.dtype.names
        for group in gpu_field.dtype.names:
            gpu_values = np.asarray(gpu_field[group], dtype=float)
            cpu_values = np.asarray(cpu_field[group], dtype=float)
            np.testing.assert_allclose(gpu_values, cpu_values, rtol=1e-13, atol=1e-15)

    params = gpu_result["params"]
    assert params["use_raw"] is False
    assert params["corr_method"] == "benjamini-hochberg"
    assert params["tie_correct"] is tie_correct
    assert params["layer"] is None
    assert params["reference"] == reference


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_honors_layer_and_use_raw(reference):
    """Test that layer parameter is respected."""
    np.random.seed(42)
    base = sc.datasets.blobs(n_variables=5, n_centers=3, n_observations=150)
    base.obs["blobs"] = base.obs["blobs"].astype("category")
    base.layers["signal"] = base.X.copy()

    ref_adata = base.copy()
    rsc.tl.rank_genes_groups(
        ref_adata, "blobs", method="wilcoxon", use_raw=False, reference=reference
    )
    reference_names = ref_adata.uns["rank_genes_groups"]["names"].copy()

    rng = np.random.default_rng(0)
    perturbed_matrix = base.X.copy()
    perturbed_matrix[rng.integers(0, 2, perturbed_matrix.shape, dtype=bool)] = 0.0

    layered = base.copy()
    layered.X = perturbed_matrix
    rsc.tl.rank_genes_groups(
        layered,
        "blobs",
        method="wilcoxon",
        layer="signal",
        use_raw=False,
        reference=reference,
    )
    layered_names = layered.uns["rank_genes_groups"]["names"].copy()

    no_layer = base.copy()
    no_layer.X = perturbed_matrix
    rsc.tl.rank_genes_groups(
        no_layer, "blobs", method="wilcoxon", use_raw=False, reference=reference
    )
    no_layer_names = no_layer.uns["rank_genes_groups"]["names"].copy()

    assert layered_names.dtype.names == reference_names.dtype.names
    for group in reference_names.dtype.names:
        assert tuple(layered_names[group]) == tuple(reference_names[group])
    differences = [
        tuple(no_layer_names[group]) != tuple(reference_names[group])
        for group in reference_names.dtype.names
    ]
    assert any(differences)


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_subset_and_bonferroni(reference):
    """Test group subsetting and bonferroni correction."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=5, n_centers=4, n_observations=150)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    groups = ["0", "1", "2"] if reference != "rest" else ["0", "2"]

    rsc.tl.rank_genes_groups(
        adata,
        "blobs",
        method="wilcoxon",
        groups=groups,
        reference=reference,
        use_raw=False,
        n_genes=2,
        corr_method="bonferroni",
    )

    result = adata.uns["rank_genes_groups"]
    expected_groups = tuple(g for g in groups if g != reference)
    assert result["scores"].dtype.names == expected_groups
    assert result["names"].dtype.names == expected_groups
    for group in result["names"].dtype.names:
        observed = np.asarray(result["names"][group])
        assert observed.size == 2
    for group in result["pvals_adj"].dtype.names:
        adjusted = np.asarray(result["pvals_adj"][group])
        assert np.all(adjusted <= 1.0)


@pytest.mark.parametrize(
    "fmt",
    [
        pytest.param("scipy_csr", id="host_csr"),
        pytest.param("scipy_csc", id="host_csc"),
        pytest.param("cupy_dense", id="device_dense"),
    ],
)
def test_wilcoxon_subset_rest_stats_match_scanpy(fmt):
    """groups=... with reference='rest' must use all other cells for stats."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=160)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()
    adata_gpu.X = _to_format(adata_gpu.X, fmt)

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "groups": ["0", "2"],
        "reference": "rest",
        "pts": True,
        "n_genes": 6,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]
    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )

    for key in ("pts", "pts_rest"):
        gpu_pts = gpu_result[key]
        cpu_pts = cpu_result[key]
        for col in gpu_pts.columns:
            np.testing.assert_allclose(
                gpu_pts[col].values, cpu_pts[col].values, rtol=1e-13, atol=1e-15
            )


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc"])
def test_wilcoxon_zero_nnz_host_sparse_does_not_crash(reference, fmt):
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["0"] * 4 + ["1"] * 4 + ["2"] * 4,
                categories=["0", "1", "2"],
            )
        }
    )
    adata = sc.AnnData(
        X=_to_format(np.zeros((12, 5), dtype=np.float32), fmt),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )

    rsc.tl.rank_genes_groups(
        adata,
        "group",
        method="wilcoxon",
        use_raw=False,
        reference=reference,
        pts=True,
    )

    result = adata.uns["rank_genes_groups"]
    for field in ("scores", "pvals"):
        for group in result[field].dtype.names:
            assert np.all(np.isfinite(np.asarray(result[field][group], dtype=float)))


def test_wilcoxon_dense_ovo_chunk_size_matches_unchunked():
    np.random.seed(42)
    base = sc.datasets.blobs(n_variables=9, n_centers=3, n_observations=120)
    base.obs["blobs"] = base.obs["blobs"].astype("category")
    unchunked = base.copy()
    chunked = base.copy()
    unchunked.X = cp.asarray(unchunked.X)
    chunked.X = cp.asarray(chunked.X)

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "1",
        "tie_correct": True,
        "n_genes": 9,
    }
    rsc.tl.rank_genes_groups(unchunked, **kw)
    rsc.tl.rank_genes_groups(chunked, **kw, chunk_size=2)

    for field in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        for group in unchunked.uns["rank_genes_groups"][field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(unchunked.uns["rank_genes_groups"][field][group], float),
                np.asarray(chunked.uns["rank_genes_groups"][field][group], float),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )


@pytest.mark.parametrize(
    "reference_before,reference_after",
    [("rest", "rest"), ("1", "One")],
)
def test_rank_genes_groups_wilcoxon_with_renamed_categories(
    reference_before, reference_after
):
    """Test with renamed category labels."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=4, n_centers=3, n_observations=200)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    # First run with original category names
    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", reference=reference_before
    )
    names = adata.uns["rank_genes_groups"]["names"]
    expected_groups = ("0", "1", "2") if reference_before == "rest" else ("0", "2")
    assert names.dtype.names == expected_groups
    first_run = tuple(names[0])

    adata.rename_categories("blobs", ["Zero", "One", "Two"])
    assert tuple(adata.uns["rank_genes_groups"]["names"][0]) == first_run

    # Second run with renamed category names
    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", reference=reference_after
    )
    renamed_names = adata.uns["rank_genes_groups"]["names"]
    assert tuple(renamed_names[0]) == first_run
    expected_renamed = (
        ("Zero", "One", "Two") if reference_after == "rest" else ("Zero", "Two")
    )
    assert renamed_names.dtype.names == expected_renamed


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_with_unsorted_groups(reference):
    """Test that group order doesn't affect results."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=180)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    bdata = adata.copy()

    groups = ["0", "1", "2", "3"] if reference != "rest" else ["0", "2", "3"]
    groups_reversed = list(reversed(groups))

    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", groups=groups, reference=reference
    )
    rsc.tl.rank_genes_groups(
        bdata, "blobs", method="wilcoxon", groups=groups_reversed, reference=reference
    )

    expected_groups = {g for g in groups if g != reference}
    assert set(adata.uns["rank_genes_groups"]["names"].dtype.names) == expected_groups
    assert set(bdata.uns["rank_genes_groups"]["names"].dtype.names) == expected_groups

    # Pick a group that's not the reference for comparison
    test_group = "3" if reference != "3" else "0"
    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        np.testing.assert_allclose(
            np.asarray(adata.uns["rank_genes_groups"][field][test_group], dtype=float),
            np.asarray(bdata.uns["rank_genes_groups"][field][test_group], dtype=float),
            rtol=1e-13,
            atol=1e-15,
            equal_nan=True,
        )

    assert tuple(adata.uns["rank_genes_groups"]["names"][test_group]) == tuple(
        bdata.uns["rank_genes_groups"]["names"][test_group]
    )


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("pre_load", [True, False])
def test_rank_genes_groups_wilcoxon_pts(reference, pre_load):
    """Test that pts (fraction of cells expressing) is computed correctly."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    # Run with pts=True
    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        tie_correct=False,
        reference=reference,
        pre_load=pre_load,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        tie_correct=False,
        reference=reference,
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    # Check pts DataFrame exists and has correct structure
    assert "pts" in gpu_result
    assert "pts" in cpu_result

    # Check pts values match scanpy
    gpu_pts = gpu_result["pts"]
    cpu_pts = cpu_result["pts"]
    assert list(gpu_pts.columns) == list(cpu_pts.columns)
    assert list(gpu_pts.index) == list(cpu_pts.index)

    for col in gpu_pts.columns:
        np.testing.assert_allclose(
            gpu_pts[col].values, cpu_pts[col].values, rtol=1e-13, atol=1e-15
        )

    # pts_rest only exists when reference='rest'
    if reference == "rest":
        assert "pts_rest" in gpu_result
        assert "pts_rest" in cpu_result

        gpu_pts_rest = gpu_result["pts_rest"]
        cpu_pts_rest = cpu_result["pts_rest"]

        for col in gpu_pts_rest.columns:
            np.testing.assert_allclose(
                gpu_pts_rest[col].values,
                cpu_pts_rest[col].values,
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# Ground-truth validation against scipy.stats.mannwhitneyu
# ============================================================================


def _make_perturbation_adata(
    n_control: int = 200,
    n_treatment: int = 150,
    n_genes: int = 500,
    n_de_genes: int = 50,
    seed: int = 42,
):
    """Two-group perturbation AnnData with count-based log1p data (many ties)."""
    rng = np.random.default_rng(seed)
    n_cells = n_control + n_treatment

    gene_means = rng.gamma(shape=2.0, scale=5.0, size=n_genes)
    X = rng.poisson(lam=gene_means[None, :], size=(n_cells, n_genes)).astype(np.float32)
    for g in range(n_de_genes):
        X[n_control:, g] = rng.poisson(
            lam=gene_means[g] * 1.5, size=n_treatment
        ).astype(np.float32)

    obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["control"] * n_control + ["treatment"] * n_treatment,
                categories=["control", "treatment"],
            ),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _scipy_mannwhitneyu_pvals(adata, *, group, reference, groupby="group"):
    """Per-gene two-sided Mann-Whitney U p-values via scipy (ground truth)."""
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float64)
    mask_g = (adata.obs[groupby] == group).values
    mask_r = (adata.obs[groupby] == reference).values
    return np.array(
        [
            mannwhitneyu(X[mask_g, i], X[mask_r, i], alternative="two-sided").pvalue
            for i in range(X.shape[1])
        ]
    )


@pytest.fixture
def perturbation_adata():
    return _make_perturbation_adata()


class TestWilcoxonAgainstScipy:
    """Validate rsc wilcoxon p-values against scipy.stats.mannwhitneyu."""

    def test_with_continuity_matches_scipy(self, perturbation_adata):
        """use_continuity + tie_correct matches scipy mannwhitneyu to machine eps."""
        adata = perturbation_adata.copy()
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            groups=["treatment"],
            reference="control",
            method="wilcoxon",
            use_raw=False,
            tie_correct=True,
            use_continuity=True,
        )

        rsc_df = (
            sc.get.rank_genes_groups_df(adata, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        scipy_pvals = _scipy_mannwhitneyu_pvals(
            perturbation_adata, group="treatment", reference="control"
        )
        # Align scipy pvals to the same gene order
        gene_to_idx = {g: i for i, g in enumerate(perturbation_adata.var_names)}
        scipy_sorted = np.array([scipy_pvals[gene_to_idx[g]] for g in rsc_df["names"]])

        np.testing.assert_allclose(
            rsc_df["pvals"].values, scipy_sorted, rtol=1e-13, atol=1e-15
        )

    def test_without_continuity_close_to_scipy(self, perturbation_adata):
        """Without continuity correction the gap is only the 0.5 adjustment term."""
        adata = perturbation_adata.copy()
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            groups=["treatment"],
            reference="control",
            method="wilcoxon",
            use_raw=False,
            tie_correct=True,
            use_continuity=False,
        )

        rsc_df = (
            sc.get.rank_genes_groups_df(adata, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        scipy_pvals = _scipy_mannwhitneyu_pvals(
            perturbation_adata, group="treatment", reference="control"
        )
        gene_to_idx = {g: i for i, g in enumerate(perturbation_adata.var_names)}
        scipy_sorted = np.array([scipy_pvals[gene_to_idx[g]] for g in rsc_df["names"]])

        np.testing.assert_allclose(
            rsc_df["pvals"].values, scipy_sorted, rtol=1e-2, atol=1e-15
        )

    @pytest.mark.parametrize("sparse", [True, False])
    def test_sparse_matches_dense(self, perturbation_adata, sparse):
        """Sparse and dense wilcoxon give identical results."""
        adata_dense = perturbation_adata.copy()
        adata_sparse = perturbation_adata.copy()
        adata_sparse.X = sp.csr_matrix(adata_sparse.X)

        kw = {
            "groupby": "group",
            "groups": ["treatment"],
            "reference": "control",
            "method": "wilcoxon",
            "use_raw": False,
            "tie_correct": True,
        }
        rsc.tl.rank_genes_groups(adata_dense, **kw)
        rsc.tl.rank_genes_groups(adata_sparse, **kw)

        dense_df = (
            sc.get.rank_genes_groups_df(adata_dense, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        sparse_df = (
            sc.get.rank_genes_groups_df(adata_sparse, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        np.testing.assert_array_equal(
            dense_df["scores"].values, sparse_df["scores"].values
        )
        np.testing.assert_array_equal(
            dense_df["pvals"].values, sparse_df["pvals"].values
        )


# ============================================================================
# Matrix format coverage: all dispatch paths must agree
# ============================================================================


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize(
    "fmt",
    [
        pytest.param("scipy_csc", id="scipy_csc"),
        pytest.param("cupy_dense", id="cupy_dense"),
        pytest.param("cupy_csr", id="cupy_csr"),
        pytest.param("cupy_csc", id="cupy_csc"),
    ],
)
def test_format_matches_scanpy(reference, fmt):
    """Every matrix format matches scanpy output."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    adata_gpu.X = _to_format(adata_gpu.X, fmt)

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    assert gpu_result["names"].dtype.names == cpu_result["names"].dtype.names
    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

    for field in ("scores", "pvals", "logfoldchanges", "pvals_adj"):
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# Negative values: centered/scaled data must match scanpy across all formats
# ============================================================================


def _make_centered_adata(n_obs=200, n_vars=8, n_centers=3, seed=42):
    """Create AnnData with centered (mean-zero) data containing negatives."""
    np.random.seed(seed)
    adata = sc.datasets.blobs(
        n_variables=n_vars, n_centers=n_centers, n_observations=n_obs
    )
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    # Center each gene to produce negative values
    adata.X = adata.X - adata.X.mean(axis=0)
    return adata


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize(
    "fmt",
    [
        pytest.param("scipy_csc", id="scipy_csc"),
        pytest.param("cupy_dense", id="cupy_dense"),
        pytest.param("cupy_csr", id="cupy_csr"),
        pytest.param("cupy_csc", id="cupy_csc"),
    ],
)
def test_negative_values_match_scanpy(reference, fmt):
    """Centered data (with negatives) matches scanpy across all formats."""
    adata_gpu = _make_centered_adata()
    adata_cpu = adata_gpu.copy()

    # Verify data actually has negatives
    assert adata_gpu.X.min() < 0

    adata_gpu.X = _to_format(adata_gpu.X, fmt)

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

    for field in ("scores", "pvals", "pvals_adj"):
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
            )


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_negative_sparse_matches_dense(reference):
    """Sparse and dense paths give identical results for centered data."""
    adata_dense = _make_centered_adata()
    adata_csr = adata_dense.copy()
    adata_csc = adata_dense.copy()

    adata_csr.X = cpsp.csr_matrix(cp.asarray(adata_dense.X))
    adata_csc.X = cpsp.csc_matrix(cp.asarray(adata_dense.X))

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata_dense, **kw)
    rsc.tl.rank_genes_groups(adata_csr, **kw)
    rsc.tl.rank_genes_groups(adata_csc, **kw)

    dense_result = adata_dense.uns["rank_genes_groups"]
    csr_result = adata_csr.uns["rank_genes_groups"]
    csc_result = adata_csc.uns["rank_genes_groups"]

    for field in ("scores", "pvals"):
        for group in dense_result[field].dtype.names:
            dense_vals = np.asarray(dense_result[field][group], dtype=float)
            csr_vals = np.asarray(csr_result[field][group], dtype=float)
            csc_vals = np.asarray(csc_result[field][group], dtype=float)
            np.testing.assert_allclose(csr_vals, dense_vals, rtol=1e-13, atol=1e-15)
            np.testing.assert_allclose(csc_vals, dense_vals, rtol=1e-13, atol=1e-15)


# ============================================================================
# pre_load: GPU transfer before wilcoxon must match default (lazy transfer)
# ============================================================================


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_pre_load_matches_scanpy(reference):
    """pre_load=True matches scanpy output."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw, pre_load=True)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    assert gpu_result["names"].dtype.names == cpu_result["names"].dtype.names
    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

    for field in ("scores", "pvals", "logfoldchanges", "pvals_adj"):
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# use_continuity with reference="rest" (OVR mode)
# ============================================================================


def test_use_continuity_vs_rest_changes_scores():
    """use_continuity with reference='rest' adjusts z-scores toward zero."""
    np.random.seed(42)
    adata_no = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_no.obs["blobs"] = adata_no.obs["blobs"].astype("category")
    adata_yes = adata_no.copy()

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "rest",
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata_no, **kw, use_continuity=False)
    rsc.tl.rank_genes_groups(adata_yes, **kw, use_continuity=True)

    for group in adata_no.uns["rank_genes_groups"]["scores"].dtype.names:
        scores_no = np.asarray(
            adata_no.uns["rank_genes_groups"]["scores"][group], dtype=float
        )
        scores_yes = np.asarray(
            adata_yes.uns["rank_genes_groups"]["scores"][group], dtype=float
        )
        # Continuity correction moves z-scores toward zero
        assert np.all(np.abs(scores_yes) <= np.abs(scores_no) + 1e-15), (
            f"Group {group}: continuity-corrected |z| should be <= uncorrected |z|"
        )
        # p-values should be valid
        pvals = np.asarray(
            adata_yes.uns["rank_genes_groups"]["pvals"][group], dtype=float
        )
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)


# ============================================================================
# mask_var: gene subsetting
# ============================================================================


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_mask_var_matches_scanpy(reference):
    """mask_var restricts genes and matches scanpy output."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=10, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    mask = np.zeros(adata_gpu.n_vars, dtype=bool)
    mask[:5] = True

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "mask_var": mask,
        "reference": reference,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])
        names = set(gpu_result["names"][group])
        expected_genes = set(adata_gpu.var_names[:5])
        assert names <= expected_genes

    for field in ("scores", "pvals", "logfoldchanges", "pvals_adj"):
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
            )


def test_mask_var_string_key():
    """mask_var accepts a string key from adata.var."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=10, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    adata_gpu.var["highly_variable"] = [True] * 5 + [False] * 5
    adata_cpu.var["highly_variable"] = [True] * 5 + [False] * 5

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "mask_var": "highly_variable",
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    for group in gpu_result["names"].dtype.names:
        assert len(gpu_result["names"][group]) == 5
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

    for field in ("scores", "pvals"):
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# key_added: custom output key
# ============================================================================


def test_key_added_matches_scanpy():
    """key_added stores results under a custom key, matching scanpy."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups(
        adata_gpu, "blobs", method="wilcoxon", use_raw=False, key_added="my_de"
    )
    sc.tl.rank_genes_groups(
        adata_cpu, "blobs", method="wilcoxon", use_raw=False, key_added="my_de"
    )

    assert "my_de" in adata_gpu.uns
    assert "rank_genes_groups" not in adata_gpu.uns

    gpu_result = adata_gpu.uns["my_de"]
    cpu_result = adata_cpu.uns["my_de"]

    for field in ("scores", "pvals", "logfoldchanges", "pvals_adj"):
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# rankby_abs: ranking by absolute score
# ============================================================================


def test_rankby_abs_matches_scanpy():
    """rankby_abs ranks genes by |score| and matches scanpy."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "n_genes": 3,
        "rankby_abs": True,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

        # Scores should be sorted by absolute value (descending)
        abs_scores = np.abs(np.asarray(gpu_result["scores"][group], dtype=float))
        assert np.all(abs_scores[:-1] >= abs_scores[1:]), (
            f"Group {group}: abs scores not sorted descending"
        )

    for field in ("scores", "pvals", "logfoldchanges", "pvals_adj"):
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# Small group warning
# ============================================================================


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_small_group_warning(reference):
    """Groups with <=25 cells trigger a RuntimeWarning."""
    np.random.seed(42)
    n_large = 100
    n_small = 20  # below MIN_GROUP_SIZE_WARNING = 25
    n_cells = n_large + n_small + n_large

    adata = sc.AnnData(
        X=np.random.randn(n_cells, 5).astype(np.float32),
        obs=pd.DataFrame(
            {
                "group": pd.Categorical(
                    ["0"] * n_large + ["1"] * n_small + ["2"] * n_large,
                    categories=["0", "1", "2"],
                ),
            }
        ),
    )

    with pytest.warns(RuntimeWarning, match="normal approximation"):
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            method="wilcoxon",
            use_raw=False,
            reference=reference,
        )


# ============================================================================
# Singlet group rejection
# ============================================================================


def test_singlet_group_raises():
    """A group with only 1 cell raises ValueError."""
    np.random.seed(42)
    adata = sc.AnnData(
        X=np.random.randn(101, 5).astype(np.float32),
        obs=pd.DataFrame(
            {
                "group": pd.Categorical(
                    ["big"] * 100 + ["tiny"],
                    categories=["big", "tiny"],
                ),
            }
        ),
    )

    with pytest.raises(ValueError, match="only contain one sample"):
        rsc.tl.rank_genes_groups(adata, "group", method="wilcoxon", use_raw=False)


# ============================================================================
# Invalid reference raises
# ============================================================================


def test_invalid_reference_raises():
    """A reference not in the categories raises ValueError."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=5, n_centers=3, n_observations=100)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    with pytest.raises(ValueError, match="reference = nonexistent"):
        rsc.tl.rank_genes_groups(
            adata,
            "blobs",
            method="wilcoxon",
            use_raw=False,
            reference="nonexistent",
        )


# ============================================================================
# String group parameter raises
# ============================================================================


def test_string_groups_raises():
    """Passing a bare string as groups raises ValueError."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=5, n_centers=3, n_observations=100)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    with pytest.raises(ValueError, match="Specify a sequence"):
        rsc.tl.rank_genes_groups(
            adata,
            "blobs",
            method="wilcoxon",
            use_raw=False,
            groups="0",
        )


# ============================================================================
# Many groups with reference: 5+ groups, one vs reference
# ============================================================================


def test_many_groups_with_reference():
    """Wilcoxon with many test groups vs a reference produces correct output."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=5, n_observations=300)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    adata_cpu = adata.copy()

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "0",
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    result = adata.uns["rank_genes_groups"]
    assert "0" not in result["names"].dtype.names
    expected = {"1", "2", "3", "4"}
    assert set(result["names"].dtype.names) == expected

    for field in ("scores", "pvals"):
        for group in result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(result[field][group], dtype=float),
                np.asarray(
                    adata_cpu.uns["rank_genes_groups"][field][group], dtype=float
                ),
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# Group subsetting with unselected cells (OVR): unselected cells in "rest"
# ============================================================================


def test_group_subset_vs_rest_unselected_cells():
    """With groups subset and reference='rest', unselected cells go into rest."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=200)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    adata_cpu = adata.copy()

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "groups": ["0", "2"],
        "reference": "rest",
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    for field in ("scores", "pvals"):
        for group in adata.uns["rank_genes_groups"][field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(adata.uns["rank_genes_groups"][field][group], dtype=float),
                np.asarray(
                    adata_cpu.uns["rank_genes_groups"][field][group], dtype=float
                ),
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# Group subsetting with reference: unselected cells excluded from pairwise
# ============================================================================


def test_group_subset_with_reference_unselected_cells():
    """With groups subset and a reference, unselected cells are excluded."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=200)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    adata_cpu = adata.copy()

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "groups": ["0", "1", "2"],
        "reference": "1",
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    result = adata.uns["rank_genes_groups"]
    assert "1" not in result["names"].dtype.names
    assert set(result["names"].dtype.names) == {"0", "2"}

    for field in ("scores", "pvals"):
        for group in result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(result[field][group], dtype=float),
                np.asarray(
                    adata_cpu.uns["rank_genes_groups"][field][group], dtype=float
                ),
                rtol=1e-13,
                atol=1e-15,
            )
