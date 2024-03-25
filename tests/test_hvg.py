from pathlib import Path

import cupy as cp
import cupyx as cpx
import numpy as np
import pandas as pd
import pytest
import rapids_singlecell as rsc
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

FILE = Path(__file__).parent / Path("_scripts/seurat_hvg.csv")
FILE_V3 = Path(__file__).parent / Path("_scripts/seurat_hvg_v3.csv.gz")
FILE_V3_BATCH = Path(__file__).parent / Path("_scripts/seurat_hvg_v3_batch.csv")


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_highly_variable_genes_basic(dtype, sparse):
    cudata = sc.datasets.blobs()
    cudata.X = cudata.X.astype(dtype)
    if sparse:
        cudata.X = csr_matrix(cudata.X)
        cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X)
    else:
        cudata.X = cp.array(cudata.X)
    np.random.seed(0)
    cudata.obs["batch"] = np.random.binomial(3, 0.5, size=(cudata.n_obs))
    cudata.obs["batch"] = cudata.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(cudata, batch_key="batch")
    assert "highly_variable_nbatches" in cudata.var.columns
    assert "highly_variable_intersection" in cudata.var.columns

    cudata = sc.datasets.blobs()
    cudata.X = cudata.X.astype(dtype)
    if sparse:
        cudata.X = csr_matrix(cudata.X)
        cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X)
    else:
        cudata.X = cp.array(cudata.X)
    batch = np.random.binomial(4, 0.5, size=(cudata.n_obs))
    cudata.obs["batch"] = batch
    cudata.obs["batch"] = cudata.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(cudata, batch_key="batch", n_top_genes=3)
    assert "highly_variable_nbatches" in cudata.var.columns
    assert cudata.var["highly_variable"].sum() == 3
    highly_var_first_layer = cudata.var["highly_variable"].copy()

    cudata = sc.datasets.blobs()
    cudata.X = cudata.X.astype(dtype)
    if sparse:
        cudata.X = csr_matrix(cudata.X)
        cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X)
    else:
        cudata.X = cp.array(cudata.X)
    if sparse:
        new_layer = cudata.X.toarray()
    else:
        new_layer = cudata.X.copy()
    cp.random.shuffle(new_layer)
    if sparse:
        cudata.layers["test_layer"] = cpx.scipy.sparse.csr_matrix(new_layer)
    else:
        cudata.layers["test_layer"] = new_layer
    cudata.obs["batch"] = batch
    cudata.obs["batch"] = cudata.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(
        cudata, batch_key="batch", n_top_genes=3, layer="test_layer"
    )
    assert "highly_variable_nbatches" in cudata.var.columns
    assert cudata.var["highly_variable"].sum() == 3
    assert (highly_var_first_layer != cudata.var["highly_variable"]).any()

    rsc.pp.highly_variable_genes(cudata)
    no_batch_hvg = cudata.var.highly_variable.copy()
    assert no_batch_hvg.any()
    cudata.obs["batch"] = "batch"
    cudata.obs["batch"] = cudata.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(cudata, batch_key="batch")
    assert np.all(no_batch_hvg == cudata.var.highly_variable)
    assert np.all(cudata.var.highly_variable_intersection == cudata.var.highly_variable)

    cudata.obs["batch"] = "a"
    cudata.obs.batch.loc[::2] = "b"
    rsc.pp.highly_variable_genes(cudata, batch_key="batch")
    assert cudata.var["highly_variable"].any()

    colnames = [
        "means",
        "dispersions",
        "dispersions_norm",
        "highly_variable_nbatches",
        "highly_variable_intersection",
        "highly_variable",
    ]

    assert np.all(np.isin(colnames, cudata.var.columns))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_highly_variable_genes_compare_to_seurat(dtype):
    seurat_hvg_info = pd.read_csv(FILE, sep=" ")

    pbmc = sc.datasets.pbmc68k_reduced()
    pbmc.X = pbmc.raw.X
    pbmc.var_names_make_unique()
    pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype)
    pbmc.X.sort_indices()
    rsc.pp.normalize_total(pbmc, target_sum=1e4)
    rsc.pp.log1p(pbmc)
    rsc.pp.highly_variable_genes(
        pbmc, flavor="seurat", min_mean=0.0125, max_mean=3, min_disp=0.5
    )

    np.testing.assert_array_equal(
        seurat_hvg_info["highly_variable"], pbmc.var["highly_variable"]
    )

    # (still) Not equal to tolerance rtol=2e-05, atol=2e-05
    # np.testing.assert_allclose(4, 3.9999, rtol=2e-05, atol=2e-05)
    np.testing.assert_allclose(
        seurat_hvg_info["means"],
        pbmc.var["means"],
        rtol=2e-05,
        atol=2e-05,
    )
    np.testing.assert_allclose(
        seurat_hvg_info["dispersions"],
        pbmc.var["dispersions"],
        rtol=2e-05,
        atol=2e-05,
    )
    np.testing.assert_allclose(
        seurat_hvg_info["dispersions_norm"],
        pbmc.var["dispersions_norm"],
        rtol=2e-05,
        atol=2e-05,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_higly_variable_genes_compare_to_seurat_v3(dtype):
    seurat_hvg_info = pd.read_csv(
        FILE_V3, sep=" ", dtype={"variances_norm": np.float64}
    )

    pbmc = sc.datasets.pbmc3k()
    pbmc.var_names_make_unique()
    pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype)

    rsc.pp.highly_variable_genes(pbmc, n_top_genes=1000, flavor="seurat_v3")

    np.testing.assert_array_equal(
        seurat_hvg_info["highly_variable"], pbmc.var["highly_variable"]
    )
    np.testing.assert_allclose(
        seurat_hvg_info["variances"],
        pbmc.var["variances"],
        rtol=2e-05,
        atol=2e-05,
    )

    batch = np.zeros((pbmc.shape[0]), dtype=int)
    batch[1500:] = 1
    pbmc.obs["batch"] = batch
    rsc.pp.highly_variable_genes(
        pbmc, n_top_genes=4000, flavor="seurat_v3", batch_key="batch"
    )
    df = pbmc.var.copy()
    df.sort_values(
        ["highly_variable_nbatches", "highly_variable_rank"],
        ascending=[False, True],
        na_position="last",
        inplace=True,
    )
    df = df.iloc[:4000]
    seurat_hvg_info_batch = pd.read_csv(
        FILE_V3_BATCH, sep=" ", dtype={"variances_norm": np.float64}
    )

    # ranks might be slightly different due to many genes having same normalized var
    seu = pd.Index(seurat_hvg_info_batch["x"].values)
    assert len(seu.intersection(df.index)) / 4000 > 0.95

    rsc.pp.log1p(pbmc)
    with pytest.warns(
        UserWarning,
        match="`flavor='seurat_v3'` expects raw count data, but non-integers were found.",
    ):
        rsc.pp.highly_variable_genes(pbmc, n_top_genes=1000, flavor="seurat_v3")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", [True, False])
def test_seurat_v3_mean_var_output_with_batchkey(dtype, sparse):
    pbmc = sc.datasets.pbmc3k()
    pbmc.var_names_make_unique()
    if sparse:
        pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype)
    else:
        pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype).toarray()
    n_cells = pbmc.shape[0]
    batch = np.zeros((n_cells), dtype=int)
    batch[1500:] = 1
    pbmc.obs["batch"] = batch

    # true_mean, true_var = _get_mean_var(pbmc.X)
    if sparse:
        X = pbmc.X.toarray().astype(cp.float64)
    else:
        X = pbmc.X.copy().astype(cp.float64)

    true_mean = cp.mean(X, axis=0)
    true_var = cp.var(X, axis=0, dtype=np.float64, ddof=1)

    rsc.pp.highly_variable_genes(
        pbmc, batch_key="batch", flavor="seurat_v3", n_top_genes=4000
    )
    cp.testing.assert_allclose(true_mean, pbmc.var["means"], rtol=2e-05, atol=2e-05)
    cp.testing.assert_allclose(true_var, pbmc.var["variances"], rtol=2e-05, atol=2e-05)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cellranger_n_top_genes_warning(dtype):
    X = cp.random.poisson(2, (100, 30), dtype=dtype)
    cudata = AnnData(X=cpx.scipy.sparse.csr_matrix(X, dtype=dtype))
    rsc.pp.normalize_total(cudata, target_sum=1e4)
    rsc.pp.log1p(cudata)

    with pytest.warns(
        UserWarning,
        match="`n_top_genes` > number of normalized dispersions, returning all genes with normalized dispersions.",
    ):
        rsc.pp.highly_variable_genes(cudata, n_top_genes=1000, flavor="cell_ranger")


def _check_pearson_hvg_columns(output_df, n_top_genes):
    assert pd.api.types.is_float_dtype(output_df["residual_variances"].dtype)

    assert output_df["highly_variable"].values.dtype is np.dtype("bool")
    assert np.sum(output_df["highly_variable"]) == n_top_genes

    assert np.nanmax(output_df["highly_variable_rank"].values) <= n_top_genes - 1


@pytest.mark.parametrize("clip", [None, np.Inf, 30])
@pytest.mark.parametrize("theta", [100, np.Inf])
@pytest.mark.parametrize("n_top_genes", [100, 200])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", [True, False])
def test_highly_variable_genes_pearson_residuals_general(
    clip, theta, n_top_genes, dtype, sparse
):
    cudata = sc.datasets.pbmc3k().copy()
    cudata = cudata[:1000, :500]
    np.random.seed(42)
    cudata.obs["batch"] = np.random.randint(0, 3, size=cudata.shape[0])
    sc.pp.filter_genes(cudata, min_cells=1)
    # cleanup var
    del cudata.var
    if sparse:
        cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X, dtype=dtype)
    else:
        cudata.X = cp.array(cudata.X.toarray(), dtype=dtype)
    # compute reference output
    residuals_reference = rsc.pp.normalize_pearson_residuals(
        cudata, clip=clip, theta=theta, inplace=False
    )
    residual_variances_reference = cp.var(residuals_reference, axis=0)

    # compute output to be tested
    rsc.pp.highly_variable_genes(
        cudata,
        flavor="pearson_residuals",
        n_top_genes=n_top_genes,
        clip=clip,
        theta=theta,
    )

    # check output is complete
    for key in [
        "highly_variable",
        "means",
        "variances",
        "residual_variances",
        "highly_variable_rank",
    ]:
        assert key in cudata.var.columns

    assert np.allclose(
        cudata.var["residual_variances"].values, residual_variances_reference
    )

    # check hvg flag
    hvg_idx = np.where(cudata.var["highly_variable"])[0]
    topn_idx = np.sort(
        np.argsort(-cudata.var["residual_variances"].values)[:n_top_genes]
    )
    assert np.all(hvg_idx == topn_idx)

    # check ranks
    assert np.nanmin(cudata.var["highly_variable_rank"].values) == 0

    # more general checks on ranks, hvg flag and residual variance
    _check_pearson_hvg_columns(cudata.var, n_top_genes)


@pytest.mark.parametrize("n_top_genes", [100, 200])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_highly_variable_genes_pearson_residuals_batch(n_top_genes, dtype):
    cudata = sc.datasets.pbmc3k().copy()
    cudata = cudata[:1000, :500].copy()
    np.random.seed(42)
    cudata.obs["batch"] = np.random.randint(0, 3, size=cudata.shape[0])
    sc.pp.filter_genes(cudata, min_cells=1)
    # cleanup var
    del cudata.var

    cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X, dtype=dtype)
    n_genes = cudata.shape[1]

    rsc.pp.highly_variable_genes(
        cudata,
        flavor="pearson_residuals",
        n_top_genes=n_top_genes,
        batch_key="batch",
    )

    # check output is complete
    for key in [
        "highly_variable",
        "means",
        "variances",
        "residual_variances",
        "highly_variable_rank",
        "highly_variable_nbatches",
        "highly_variable_intersection",
    ]:
        assert key in cudata.var.keys()

    # general checks on ranks, hvg flag and residual variance
    _check_pearson_hvg_columns(cudata.var, n_top_genes)

    # check intersection flag
    nbatches = len(np.unique(cudata.obs["batch"]))
    assert cudata.var["highly_variable_intersection"].values.dtype is np.dtype("bool")
    assert np.sum(cudata.var["highly_variable_intersection"]) <= n_top_genes * nbatches
    assert np.all(
        cudata.var["highly_variable"][cudata.var.highly_variable_intersection]
    )

    # check ranks (with batch_key these are the median of within-batch ranks)
    assert pd.api.types.is_float_dtype(cudata.var["highly_variable_rank"].dtype)

    # check nbatches
    assert cudata.var["highly_variable_nbatches"].values.dtype is np.dtype("int")
    assert np.min(cudata.var["highly_variable_nbatches"].values) >= 0
    assert np.max(cudata.var["highly_variable_nbatches"].values) <= nbatches

    assert len(cudata.var) == n_genes
