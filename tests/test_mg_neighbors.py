from __future__ import annotations

import itertools

import numpy as np
import pytest
from scanpy.datasets import pbmc68k_reduced

import rapids_singlecell as rsc


def _calc_recall(distances, reference_distances, tolerance=0.95):
    hits = 0
    total = 0
    for (p_start, p_stop), (g_start, g_stop) in zip(
        itertools.pairwise(distances.indptr),
        itertools.pairwise(reference_distances.indptr),
    ):
        inter = np.intersect1d(
            distances.indices[p_start:p_stop],
            reference_distances.indices[g_start:g_stop],
        )
        hits += inter.size
        total += p_stop - p_start  # number of predicted neighbors

    recall = hits / total
    assert recall > tolerance


@pytest.mark.parametrize("algo", ["mg_ivfflat", "mg_ivfpq"])
def test_mg_neighbors(algo):
    if algo == "mg_ivfflat":
        other_algo = "ivfflat"
    else:
        other_algo = "ivfpq"

    k = 15
    adata = pbmc68k_reduced()
    n_rows = adata.shape[0]
    rsc.pp.neighbors(adata, n_pcs=50, n_neighbors=k, algorithm=algo)

    assert adata.obsp["distances"].shape == (n_rows, n_rows)
    assert adata.obsp["connectivities"].shape == (n_rows, n_rows)
    distances = adata.obsp["distances"].copy()
    rsc.pp.neighbors(adata, n_pcs=50, n_neighbors=k, algorithm=other_algo)
    np.testing.assert_array_equal(adata.obsp["distances"].indptr, distances.indptr)
    _calc_recall(distances, adata.obsp["distances"])


@pytest.mark.parametrize("algo", ["nn_descent", "ivfpq"])
def test_all_neighbors(algo):
    adata = pbmc68k_reduced()
    n_rows = adata.shape[0]
    if algo == "ivfpq":
        algorithm_kwds = {
            "algo": "ivf_pq",
            "n_lists": 15,
        }
        tolerance = 0.85
    else:
        algorithm_kwds = {
            "algo": "nn_descent",
        }
        tolerance = 0.95
    rsc.pp.neighbors(
        adata,
        n_pcs=50,
        n_neighbors=15,
        algorithm="all_neighbors",
        algorithm_kwds=algorithm_kwds,
    )
    assert adata.obsp["distances"].shape == (n_rows, n_rows)
    assert adata.obsp["connectivities"].shape == (n_rows, n_rows)
    distances = adata.obsp["distances"].copy()
    rsc.pp.neighbors(adata, n_pcs=50, n_neighbors=15, algorithm=algo)
    _calc_recall(distances, adata.obsp["distances"], tolerance=tolerance)
