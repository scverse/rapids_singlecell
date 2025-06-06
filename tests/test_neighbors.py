from __future__ import annotations

import itertools

import cupy as cp
import numpy as np
import pytest
import scanpy.external as sce
from anndata import AnnData
from bbknn.matrix import trimming as trimming_cpu
from scanpy.datasets import pbmc68k_reduced

from rapids_singlecell.get import X_to_GPU
from rapids_singlecell.pp import bbknn, neighbors
from rapids_singlecell.preprocessing._neighbors import _trimming as trimming_gpu

# the input data
X = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])

distances_euclidean = [
    [0.0, 2.0, 0.0, 4.123105525970459],
    [2.0, 0.0, 0.0, 5.0],
    [0.0, 6.324555397033691, 0.0, 5.385164737701416],
    [4.123105525970459, 5.0, 0.0, 0.0],
]

connectivities_umap = [
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.5849691143165735, 0.8277419907567016],
    [0.0, 0.5849691143165735, 0.0, 1.0],
    [1.0, 0.8277419907567016, 1.0, 0.0],
]


@pytest.mark.parametrize("algo", ["brute", "ivfflat", "cagra"])
def test_umap_connectivities_euclidean(algo):
    adata = AnnData(X=X)
    neighbors(adata, n_neighbors=3, algorithm=algo)
    assert np.allclose(adata.obsp["distances"].toarray(), distances_euclidean)
    assert np.allclose(adata.obsp["connectivities"].toarray(), connectivities_umap)


@pytest.mark.parametrize("algo", ["brute", "ivfflat", "cagra", "ivfpq", "nn_descent"])
def test_algo(algo):
    adata = pbmc68k_reduced()
    neighbors(adata, n_neighbors=5, algorithm=algo)


def test_nn_descent_intermediate_graph_degree():
    adata = pbmc68k_reduced()
    neighbors(
        adata,
        n_neighbors=5,
        algorithm="nn_descent",
        algorithm_kwds={"intermediate_graph_degree": 10},
    )


@pytest.mark.parametrize("algo", ["ivfflat", "ivfpq"])
def test_ivf_algorithm_kwds(algo):
    adata = pbmc68k_reduced()
    neighbors(
        adata,
        n_neighbors=5,
        algorithm=algo,
        algorithm_kwds={"n_lists": 10, "n_probes": 10},
    )


@pytest.mark.parametrize("algo", ["nn_descent", "ivfpq"])
def test_indices_approx_nn(algo):
    adata = pbmc68k_reduced()
    brute_data = adata.copy()
    neighbors(adata, n_neighbors=5, algorithm=algo)
    neighbors(brute_data, n_neighbors=5, algorithm="brute")
    counter = 0
    for (rsc_start, rsc_stop), (b_start, b_stop) in zip(
        itertools.pairwise(adata.obsp["distances"].indptr),
        itertools.pairwise(brute_data.obsp["distances"].indptr),
    ):
        counter += len(
            np.intersect1d(
                adata.obsp["distances"].indices[rsc_start:rsc_stop],
                brute_data.obsp["distances"].indices[b_start:b_stop],
            )
        )
    assert counter / b_stop > 0.9


key = "test"


def test_neighbors_key_added():
    adata = pbmc68k_reduced()
    del adata.obsp
    neighbors(adata, n_neighbors=5, random_state=0)
    neighbors(adata, n_neighbors=5, random_state=0, key_added=key)

    conns_key = adata.uns[key]["connectivities_key"]
    dists_key = adata.uns[key]["distances_key"]

    assert adata.uns["neighbors"]["params"] == adata.uns[key]["params"]
    assert np.allclose(
        adata.obsp["connectivities"].toarray(), adata.obsp[conns_key].toarray()
    )
    assert np.allclose(
        adata.obsp["distances"].toarray(), adata.obsp[dists_key].toarray()
    )


def test_bbknn():
    """
    Test the bbknn function against the scanpy implementation. We want more than a 90% overlap between the two.
    This is calculated by the number of shared indices between the two sparse distance matrices divided by the number of indices in the original implementation.
    """

    adata = pbmc68k_reduced()
    sce.pp.bbknn(
        adata,
        n_pcs=15,
        batch_key="phase",
        computation="pynndescent",
        metric="euclidean",
    )
    bbknn(adata, n_pcs=15, batch_key="phase", algorithm="brute", key_added="rapids")
    counter = 0
    for (rsc_start, rsc_stop), (b_start, b_stop) in zip(
        itertools.pairwise(adata.obsp["rapids_distances"].indptr),
        itertools.pairwise(adata.obsp["distances"].indptr),
    ):
        counter += len(
            np.intersect1d(
                adata.obsp["rapids_distances"].indices[rsc_start:rsc_stop],
                adata.obsp["distances"].indices[b_start:b_stop],
            )
        )
    assert counter / b_stop > 0.9


def test_trimming():
    adata = pbmc68k_reduced()
    cnts_gpu = X_to_GPU(adata.obsp["connectivities"]).astype(np.float32)
    cnts_cpu = adata.obsp["connectivities"].astype(np.float32)

    cnts_cpu = trimming_cpu(cnts_cpu, 5)
    cnts_gpu = trimming_gpu(cnts_gpu, 5)

    cp.testing.assert_array_equal(cnts_cpu.data, cnts_gpu.data)
    cp.testing.assert_array_equal(cnts_cpu.indices, cnts_gpu.indices)
    cp.testing.assert_array_equal(cnts_cpu.indptr, cnts_gpu.indptr)
