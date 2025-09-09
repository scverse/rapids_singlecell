from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from pandas import DataFrame, Series
from pertpy import data as dt
from pytest import fixture, mark
from scipy import sparse as sp

from rapids_singlecell.pertpy_gpu._distances import Distance


@pytest.fixture
def cp_rng():  # TODO(selmanozleyen): Think of a way to integrate this with decoupler's rng fixture
    rng = cp.random.default_rng(seed=42)
    return rng


@pytest.fixture
def np_rng():  # TODO(selmanozleyen): Think of a way to integrate this with decoupler's rng fixture
    rng = np.random.default_rng(seed=42)
    return rng


actual_distances = [
    # Euclidean distances and related
    # "euclidean",
    # "mean_absolute_error",
    # "mean_pairwise",
    # "mse",
    "edistance",
    # Other
    # "cosine_distance",
    # "kendalltau_distance",
    # "mmd",
    # "pearson_distance",
    # "spearman_distance",
    # "t_test",
    # "mahalanobis",
]
# semi_distances = ["r2_distance", "sym_kldiv", "ks_test"]
# non_distances = ["classifier_proba"]
# onesided_only = ["classifier_cp"]
# pseudo_counts_distances = ["nb_ll"]
# lognorm_counts_distances = ["mean_var_distribution"]
all_distances = actual_distances  # + semi_distances + non_distances + lognorm_counts_distances + pseudo_counts_distances  # + onesided_only
semi_distances = []
non_distances = []
onesided_only = []
pseudo_counts_distances = []
lognorm_counts_distances = []


@fixture
def adata(request):
    low_subsample_distances = [
        "sym_kldiv",
        "t_test",
        "ks_test",
        "classifier_proba",
        "classifier_cp",
        "mahalanobis",
        "mean_var_distribution",
    ]
    no_subsample_distances = [
        "mahalanobis"
    ]  # mahalanobis only works on the full data without subsampling

    distance = request.node.callspec.params["distance"]

    adata = dt.distance_example()
    if distance not in no_subsample_distances:
        if distance in low_subsample_distances:
            adata = sc.pp.subsample(adata, 0.1, copy=True)
        else:
            adata = sc.pp.subsample(adata, 0.001, copy=True)

    adata = adata[
        :, np.random.default_rng().choice(adata.n_vars, 100, replace=False)
    ].copy()

    adata.layers["lognorm"] = adata.X.copy()
    adata.layers["counts"] = cp.round(adata.X.toarray()).astype(int)
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=5)
    if distance in lognorm_counts_distances:
        groups = np.unique(adata.obs["perturbation"])
        # KDE is slow, subset to 3 groups for speed up
        adata = adata[adata.obs["perturbation"].isin(groups[0:3])].copy()

    adata.X = cp.asarray(adata.X.toarray())
    for l_key in adata.layers.keys():
        if sp.issparse(adata.layers[l_key]):
            from cupyx.scipy.sparse import coo_matrix, csc_matrix, csr_matrix

            if sp.isspmatrix_csr(adata.layers[l_key]):
                adata.layers[l_key] = csr_matrix(adata.layers[l_key])
            elif sp.isspmatrix_csc(adata.layers[l_key]):
                adata.layers[l_key] = csc_matrix(adata.layers[l_key])
            elif sp.isspmatrix_coo(adata.layers[l_key]):
                adata.layers[l_key] = coo_matrix(adata.layers[l_key])
        else:
            adata.layers[l_key] = cp.asarray(adata.layers[l_key])
    adata.layers["lognorm"] = cp.asarray(adata.layers["lognorm"].toarray())
    adata.layers["counts"] = cp.asarray(adata.layers["counts"])
    adata.obsm["X_pca"] = cp.asarray(adata.obsm["X_pca"])

    return adata


@fixture
def distance_obj(request):
    distance = request.node.callspec.params["distance"]
    if distance in lognorm_counts_distances:
        d = Distance(distance, layer_key="lognorm")
    elif distance in pseudo_counts_distances:
        d = Distance(distance, layer_key="counts")
    else:
        d = Distance(distance, obsm_key="X_pca")
    return d


@fixture
@mark.parametrize("distance", all_distances)
def pairwise_distance(adata, distance_obj, distance):
    return distance_obj.pairwise(adata, groupby="perturbation", show_progressbar=False)


@mark.parametrize("distance", actual_distances + semi_distances)
def test_distance_axioms(pairwise_distance, distance):
    # This is equivalent to testing for a semimetric, defined as fulfilling all axioms except triangle inequality.
    # (M1) Definiteness
    assert all(np.diag(pairwise_distance.values) == 0)  # distance to self is 0

    # (M2) Positivity
    assert len(pairwise_distance) == np.sum(
        pairwise_distance.values == 0
    )  # distance to other is not 0
    assert all(pairwise_distance.values.flatten() >= 0)  # distance is non-negative

    # (M3) Symmetry
    assert np.sum(pairwise_distance.values - pairwise_distance.values.T) == 0


@mark.parametrize("distance", actual_distances)
def test_triangle_inequality(pairwise_distance, distance, np_rng):
    # Test if distances are well-defined in accordance with metric axioms
    # (M4) Triangle inequality (we just probe this for a few random triplets)
    # Some tests are not well defined for the triangle inequality. We skip those.
    if distance in {"mahalanobis"}:
        return

    for _ in range(5):
        triplet = np_rng.choice(pairwise_distance.index, size=3, replace=False)
        assert (
            pairwise_distance.loc[triplet[0], triplet[1]]
            + pairwise_distance.loc[triplet[1], triplet[2]]
            >= pairwise_distance.loc[triplet[0], triplet[2]]
        )


@mark.parametrize("distance", all_distances)
def test_distance_layers(pairwise_distance, distance):
    assert isinstance(pairwise_distance, DataFrame)
    assert pairwise_distance.columns.equals(pairwise_distance.index)
    assert (
        np.sum(pairwise_distance.values - pairwise_distance.values.T) == 0
    )  # symmetry


@mark.parametrize("distance", actual_distances + pseudo_counts_distances)
def test_distance_counts(adata, distance):
    if (
        distance != "mahalanobis"
    ):  # skip, doesn't work because covariance matrix is a singular matrix, not invertible
        distance = pt.tl.Distance(distance, layer_key="counts")
        df = distance.pairwise(adata, groupby="perturbation")
        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0


@mark.parametrize("distance", all_distances)
def test_mutually_exclusive_keys(distance):
    with pytest.raises(ValueError):
        _ = Distance(distance, layer_key="counts", obsm_key="X_pca")


@mark.parametrize("distance", actual_distances + semi_distances + non_distances)
def test_distance_output_type(distance, cp_rng):
    # Test if distances are outputting floats
    dist = Distance(distance)
    X = cp_rng.standard_normal(size=(50, 10))
    Y = cp_rng.standard_normal(size=(50, 10))
    d = dist(X, Y)
    d = float(d.get())
    assert isinstance(d, float)


@mark.parametrize("distance", all_distances + onesided_only)
def test_distance_onesided(adata, distance_obj, distance):
    # Test consistency of one-sided distance results
    selected_group = adata.obs.perturbation.unique()[0]
    df = distance_obj.onesided_distances(
        adata, groupby="perturbation", selected_group=selected_group
    )
    assert isinstance(df, Series)
    assert df.loc[selected_group] == 0  # distance to self is 0


def test_bootstrap_distance_output_type(cp_rng):
    # Test if distances are outputting floats
    d = Distance(metric="edistance")
    X = cp_rng.standard_normal(size=(50, 10))
    Y = cp_rng.standard_normal(size=(50, 10))
    d = d.bootstrap(X, Y, n_bootstrap=3)
    assert hasattr(d, "mean")
    assert hasattr(d, "variance")


@mark.parametrize("distance", ["edistance"])
def test_bootstrap_distance_pairwise(adata, distance):
    # Test consistency of pairwise distance results
    dist = Distance(distance, obsm_key="X_pca")
    bootstrap_output = dist.pairwise(
        adata, groupby="perturbation", bootstrap=True, n_bootstrap=3
    )

    assert isinstance(bootstrap_output, tuple)

    mean = bootstrap_output[0]
    var = bootstrap_output[1]

    assert mean.columns.equals(mean.index)
    assert np.sum(mean.values - mean.values.T) == 0  # symmetry
    assert np.sum(var.values - var.values.T) == 0  # symmetry


# @mark.parametrize("distance", ["edistance"])
# def test_bootstrap_distance_onesided(adata, distance):
#     # Test consistency of one-sided distance results
#     selected_group = adata.obs.perturbation.unique()[0]
#     d = Distance(distance, obsm_key="X_pca")
#     bootstrap_output = d.onesided_distances(
#         adata,
#         groupby="perturbation",
#         selected_group=selected_group,
#         bootstrap=True,
#         n_bootstrap=3,
#     )

#     assert isinstance(bootstrap_output, tuple)


# def test_compare_distance(rng):
#     X = rng.standard_normal(size=(50, 10))
#     Y = rng.standard_normal(size=(50, 10))
#     C = rng.standard_normal(size=(50, 10))
#     d = Distance()
#     res_simple = d.compare_distance(X, Y, C, mode="simple")
#     assert isinstance(res_simple.get(), float)
#     res_scaled = d.compare_distance(X, Y, C, mode="scaled")
#     assert isinstance(res_scaled.get(), float)
#     with pytest.raises(ValueError):
#         d.compare_distance(X, Y, C, mode="new_mode")
