from __future__ import annotations

import numpy as np
import pytest
from scanpy.datasets import pbmc3k_processed, pbmc68k_reduced

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


def _setup_data(data_kind):
    """Create GPU in-memory and Dask versions of a dataset."""
    if data_kind == "dense":
        adata = pbmc68k_reduced()
        dask_data = adata.copy()
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        rsc.get.anndata_to_GPU(adata)
        groupby = "bulk_labels"
    elif data_kind == "sparse":
        adata = pbmc3k_processed()
        org_var_names = adata.var_names
        adata = adata.raw.to_adata()
        adata = adata[:, org_var_names].copy()
        dask_data = adata.copy()
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        rsc.get.anndata_to_GPU(adata)
        groupby = "louvain"
    return adata, dask_data, groupby


def _compare_scores(gpu_result, dask_result):
    """Assert that GPU and Dask results have matching scores per gene."""
    assert set(gpu_result["names"].dtype.names) == set(dask_result["names"].dtype.names)

    for group in gpu_result["scores"].dtype.names:
        gpu_names = list(gpu_result["names"][group])
        gpu_scores = np.asarray(gpu_result["scores"][group], dtype=float)
        dask_names = list(dask_result["names"][group])
        dask_scores = np.asarray(dask_result["scores"][group], dtype=float)

        gpu_score_map = dict(zip(gpu_names, gpu_scores))
        dask_score_map = dict(zip(dask_names, dask_scores))

        for gene in gpu_score_map:
            np.testing.assert_allclose(
                gpu_score_map[gene],
                dask_score_map[gene],
                rtol=1e-7,
                atol=1e-8,
                err_msg=f"Score mismatch for gene {gene} in group {group}",
            )


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_wilcoxon_binned_dask(client, data_kind):
    """Test wilcoxon_binned with dask arrays matches GPU in-memory results."""
    adata, dask_data, groupby = _setup_data(data_kind)

    rsc.tl.rank_genes_groups(
        adata, groupby=groupby, method="wilcoxon_binned", use_raw=False
    )
    rsc.tl.rank_genes_groups(
        dask_data, groupby=groupby, method="wilcoxon_binned", use_raw=False
    )

    _compare_scores(adata.uns["rank_genes_groups"], dask_data.uns["rank_genes_groups"])


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_wilcoxon_binned_dask_group_subset(client, data_kind):
    """Test wilcoxon_binned with group subset matches GPU in-memory."""
    adata, dask_data, groupby = _setup_data(data_kind)

    # Pick a subset of groups from the data
    all_groups = list(adata.obs[groupby].cat.categories)
    groups = all_groups[:3]

    rsc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="wilcoxon_binned",
        groups=groups,
        use_raw=False,
    )
    rsc.tl.rank_genes_groups(
        dask_data,
        groupby=groupby,
        method="wilcoxon_binned",
        groups=groups,
        use_raw=False,
    )

    _compare_scores(adata.uns["rank_genes_groups"], dask_data.uns["rank_genes_groups"])


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_wilcoxon_binned_dask_reference(client, data_kind):
    """Test wilcoxon_binned with reference group matches GPU in-memory."""
    adata, dask_data, groupby = _setup_data(data_kind)

    reference = str(adata.obs[groupby].cat.categories[0])

    rsc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="wilcoxon_binned",
        reference=reference,
        use_raw=False,
    )
    rsc.tl.rank_genes_groups(
        dask_data,
        groupby=groupby,
        method="wilcoxon_binned",
        reference=reference,
        use_raw=False,
    )

    _compare_scores(adata.uns["rank_genes_groups"], dask_data.uns["rank_genes_groups"])
