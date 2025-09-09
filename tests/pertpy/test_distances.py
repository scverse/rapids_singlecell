from __future__ import annotations

import numpy as np
import pertpy as pt
import pytest
import scanpy as sc
from pandas import DataFrame, Series
from pytest import fixture, mark


@pytest.fixture
def rng():  # TODO(selmanozleyen): Think of a way to integrate this with decoupler's rng fixture
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

    adata = pt.dt.distance_example()
    if distance not in no_subsample_distances:
        if distance in low_subsample_distances:
            adata = sc.pp.subsample(adata, 0.1, copy=True)
        else:
            adata = sc.pp.subsample(adata, 0.001, copy=True)

    adata = adata[
        :, np.random.default_rng().choice(adata.n_vars, 100, replace=False)
    ].copy()

    adata.layers["lognorm"] = adata.X.copy()
    adata.layers["counts"] = np.round(adata.X.toarray()).astype(int)
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=5)
    if distance in lognorm_counts_distances:
        groups = np.unique(adata.obs["perturbation"])
        # KDE is slow, subset to 3 groups for speed up
        adata = adata[adata.obs["perturbation"].isin(groups[0:3])].copy()

    return adata


@fixture
def distance_obj(request):
    distance = request.node.callspec.params["distance"]
    if distance in lognorm_counts_distances:
        Distance = pt.tl.Distance(distance, layer_key="lognorm")
    elif distance in pseudo_counts_distances:
        Distance = pt.tl.Distance(distance, layer_key="counts")
    else:
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
    return Distance


@fixture
@mark.parametrize("distance", all_distances)
def pairwise_distance(adata, distance_obj, distance):
    return distance_obj.pairwise(adata, groupby="perturbation", show_progressbar=True)


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
def test_triangle_inequality(pairwise_distance, distance, rng):
    # Test if distances are well-defined in accordance with metric axioms
    # (M4) Triangle inequality (we just probe this for a few random triplets)
    # Some tests are not well defined for the triangle inequality. We skip those.
    if distance in {"mahalanobis"}:
        return

    for _ in range(5):
        triplet = rng.choice(pairwise_distance.index, size=3, replace=False)
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
        _ = pt.tl.Distance(distance, layer_key="counts", obsm_key="X_pca")


@mark.parametrize("distance", actual_distances + semi_distances + non_distances)
def test_distance_output_type(distance, rng):
    # Test if distances are outputting floats
    Distance = pt.tl.Distance(distance)
    X = rng.normal(size=(50, 10))
    Y = rng.normal(size=(50, 10))
    d = Distance(X, Y)
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


def test_bootstrap_distance_output_type(rng):
    # Test if distances are outputting floats
    Distance = pt.tl.Distance(metric="edistance")
    X = rng.normal(size=(50, 10))
    Y = rng.normal(size=(50, 10))
    d = Distance.bootstrap(X, Y, n_bootstrap=3)
    assert hasattr(d, "mean")
    assert hasattr(d, "variance")


@mark.parametrize("distance", ["edistance"])
def test_bootstrap_distance_pairwise(adata, distance):
    # Test consistency of pairwise distance results
    Distance = pt.tl.Distance(distance, obsm_key="X_pca")
    bootstrap_output = Distance.pairwise(
        adata, groupby="perturbation", bootstrap=True, n_bootstrap=3
    )

    assert isinstance(bootstrap_output, tuple)

    mean = bootstrap_output[0]
    var = bootstrap_output[1]

    assert mean.columns.equals(mean.index)
    assert np.sum(mean.values - mean.values.T) == 0  # symmetry
    assert np.sum(var.values - var.values.T) == 0  # symmetry


@mark.parametrize("distance", ["edistance"])
def test_bootstrap_distance_onesided(adata, distance):
    # Test consistency of one-sided distance results
    selected_group = adata.obs.perturbation.unique()[0]
    Distance = pt.tl.Distance(distance, obsm_key="X_pca")
    bootstrap_output = Distance.onesided_distances(
        adata,
        groupby="perturbation",
        selected_group=selected_group,
        bootstrap=True,
        n_bootstrap=3,
    )

    assert isinstance(bootstrap_output, tuple)


def test_compare_distance(rng):
    X = rng.normal(size=(50, 10))
    Y = rng.normal(size=(50, 10))
    C = rng.normal(size=(50, 10))
    Distance = pt.tl.Distance()
    res_simple = Distance.compare_distance(X, Y, C, mode="simple")
    assert isinstance(res_simple, float)
    res_scaled = Distance.compare_distance(X, Y, C, mode="scaled")
    assert isinstance(res_scaled, float)
    with pytest.raises(ValueError):
        Distance.compare_distance(X, Y, C, mode="new_mode")
