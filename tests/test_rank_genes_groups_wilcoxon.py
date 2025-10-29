from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc

import rapids_singlecell as rsc

cp = pytest.importorskip("cupy")


def _require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA devices available for Wilcoxon test.")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable for Wilcoxon test.")


def test_rank_genes_groups_wilcoxon_matches_scanpy_output():
    _require_cuda()
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups_wilcoxon(
        adata_gpu, "blobs", use_raw=False, n_genes=3, corr_method="benjamini-hochberg"
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        n_genes=3,
        tie_correct=False,
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
            np.testing.assert_allclose(
                gpu_values, cpu_values, rtol=1e-5, atol=1e-6, equal_nan=True
            )

    params = gpu_result["params"]
    assert params["use_raw"] is False
    assert params["corr_method"] == "benjamini-hochberg"
    assert params["tie_correct"] is False
    assert params["layer"] is None


def test_rank_genes_groups_wilcoxon_honors_layer_and_use_raw():
    _require_cuda()
    base = sc.datasets.blobs(n_variables=5, n_centers=3, n_observations=150)
    base.obs["blobs"] = base.obs["blobs"].astype("category")
    base.layers["signal"] = base.X.copy()

    reference = base.copy()
    rsc.tl.rank_genes_groups_wilcoxon(reference, "blobs", use_raw=False)
    reference_names = reference.uns["rank_genes_groups"]["names"].copy()

    rng = np.random.default_rng(0)
    perturbed_matrix = base.X.copy()
    perturbed_matrix[rng.integers(0, 2, perturbed_matrix.shape, dtype=bool)] = 0.0

    layered = base.copy()
    layered.X = perturbed_matrix
    rsc.tl.rank_genes_groups_wilcoxon(layered, "blobs", layer="signal", use_raw=False)
    layered_names = layered.uns["rank_genes_groups"]["names"].copy()

    no_layer = base.copy()
    no_layer.X = perturbed_matrix
    rsc.tl.rank_genes_groups_wilcoxon(no_layer, "blobs", use_raw=False)
    no_layer_names = no_layer.uns["rank_genes_groups"]["names"].copy()

    assert layered_names.dtype.names == reference_names.dtype.names
    for group in reference_names.dtype.names:
        assert tuple(layered_names[group]) == tuple(reference_names[group])
    differences = [
        tuple(no_layer_names[group]) != tuple(reference_names[group])
        for group in reference_names.dtype.names
    ]
    assert any(differences)


def test_rank_genes_groups_wilcoxon_subset_and_bonferroni():
    _require_cuda()
    adata = sc.datasets.blobs(n_variables=5, n_centers=4, n_observations=150)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    rsc.tl.rank_genes_groups_wilcoxon(
        adata,
        "blobs",
        groups=["0", "2"],
        use_raw=False,
        n_genes=2,
        corr_method="bonferroni",
    )

    result = adata.uns["rank_genes_groups"]
    assert result["scores"].dtype.names == ("0", "2")
    assert result["names"].dtype.names == ("0", "2")
    for group in result["names"].dtype.names:
        observed = np.asarray(result["names"][group])
        assert observed.size == 2
    for group in result["pvals_adj"].dtype.names:
        adjusted = np.asarray(result["pvals_adj"][group])
        assert np.all(adjusted <= 1.0)


def test_rank_genes_groups_wilcoxon_with_renamed_categories():
    _require_cuda()
    adata = sc.datasets.blobs(n_variables=4, n_centers=3, n_observations=200)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    rsc.tl.rank_genes_groups_wilcoxon(adata, "blobs")
    names = adata.uns["rank_genes_groups"]["names"]
    assert names.dtype.names == ("0", "1", "2")
    first_run = tuple(names["0"])

    adata.rename_categories("blobs", ["Zero", "One", "Two"])
    assert tuple(adata.uns["rank_genes_groups"]["names"]["0"]) == first_run

    rsc.tl.rank_genes_groups_wilcoxon(adata, "blobs")
    renamed_names = adata.uns["rank_genes_groups"]["names"]
    assert renamed_names.dtype.names == ("Zero", "One", "Two")
    assert tuple(renamed_names["Zero"]) == first_run


def test_rank_genes_groups_wilcoxon_with_unsorted_groups():
    _require_cuda()
    adata = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=180)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    bdata = adata.copy()

    rsc.tl.rank_genes_groups_wilcoxon(adata, "blobs", groups=["0", "2", "3"])
    rsc.tl.rank_genes_groups_wilcoxon(bdata, "blobs", groups=["3", "0", "2"])

    assert adata.uns["rank_genes_groups"]["names"].dtype.names == ("0", "2", "3")
    assert bdata.uns["rank_genes_groups"]["names"].dtype.names == ("3", "0", "2")

    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        np.testing.assert_allclose(
            np.asarray(adata.uns["rank_genes_groups"][field]["3"], dtype=float),
            np.asarray(bdata.uns["rank_genes_groups"][field]["3"], dtype=float),
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        )

    assert tuple(adata.uns["rank_genes_groups"]["names"]["3"]) == tuple(
        bdata.uns["rank_genes_groups"]["names"]["3"]
    )
