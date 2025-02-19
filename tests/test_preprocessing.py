from __future__ import annotations

import cupy as cp
import pytest
from cupy.testing import assert_allclose
from cupyx.scipy.sparse import csc_matrix, csr_matrix, issparse
from scanpy.datasets import pbmc68k_reduced

import rapids_singlecell as rsc


@pytest.mark.parametrize("array_type", (csr_matrix, csc_matrix, cp.ndarray))
@pytest.mark.parametrize(
    ("max_cells", "max_counts", "min_cells", "min_counts"),
    [
        (100, None, None, None),
        (None, 100, None, None),
        (None, None, 20, None),
        (None, None, None, 20),
    ],
)
def test_filter_genes(array_type, max_cells, max_counts, min_cells, min_counts):
    adata = pbmc68k_reduced()
    adata.X = adata.raw.X.todense()
    rsc.get.anndata_to_GPU(adata)
    adata_casted = adata.copy()
    if array_type is not cp.ndarray:
        adata_casted.X = array_type(adata_casted.X)
    rsc.pp.filter_genes(
        adata,
        max_cells=max_cells,
        max_counts=max_counts,
        min_cells=min_cells,
        min_counts=min_counts,
    )
    rsc.pp.filter_genes(
        adata_casted,
        max_cells=max_cells,
        max_counts=max_counts,
        min_cells=min_cells,
        min_counts=min_counts,
    )
    X = adata_casted.X
    if issparse(X):
        X = X.todense()

    assert_allclose(X, adata.X, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("array_type", (csr_matrix, csc_matrix, cp.ndarray))
@pytest.mark.parametrize(
    ("max_genes", "max_counts", "min_genes", "min_counts"),
    [
        (100, None, None, None),
        (None, 100, None, None),
        (None, None, 20, None),
        (None, None, None, 20),
    ],
)
def test_filter_cells(array_type, max_genes, max_counts, min_genes, min_counts):
    adata = pbmc68k_reduced()
    adata.X = adata.raw.X.todense()
    rsc.get.anndata_to_GPU(adata)
    adata_casted = adata.copy()
    if array_type is not cp.ndarray:
        adata_casted.X = array_type(adata_casted.X)
    rsc.pp.filter_cells(
        adata,
        max_genes=max_genes,
        max_counts=max_counts,
        min_genes=min_genes,
        min_counts=min_counts,
    )
    rsc.pp.filter_cells(
        adata_casted,
        max_genes=max_genes,
        max_counts=max_counts,
        min_genes=min_genes,
        min_counts=min_counts,
    )
    X = adata_casted.X
    if issparse(X):
        X = X.todense()
    assert_allclose(X, adata.X, rtol=1e-5, atol=1e-5)
