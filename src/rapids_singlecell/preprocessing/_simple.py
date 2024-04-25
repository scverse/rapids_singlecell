from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from ._qc import calculate_qc_metrics

if TYPE_CHECKING:
    from anndata import AnnData


def flag_gene_family(
    adata: AnnData,
    *,
    gene_family_name: str,
    gene_family_prefix: str = None,
    gene_list: list = None,
) -> None:
    """
    Flags a gene or gene_family in .var with boolean. (e.g all mitochondrial genes).
    Please only choose gene_family prefix or gene_list

    Parameters
    ----------
        adata
            AnnData object

        gene_family_name
            name of columns in .var where you want to store informationa as a boolean

        gene_family_prefix
            prefix of the gene family (eg. mt- for all mitochondrial genes in mice)

        gene_list
            list of genes to flag in `.var`

    Returns
    -------
        adds the boolean column in `.var`

    """
    if gene_family_prefix:
        adata.var[gene_family_name] = cp.asnumpy(
            adata.var.index.str.startswith(gene_family_prefix)
        ).ravel()
    if gene_list:
        adata.var[gene_family_name] = cp.asnumpy(
            adata.var.index.isin(gene_list)
        ).ravel()


def filter_genes(
    adata: AnnData,
    *,
    qc_var: str = "n_cells_by_counts",
    min_count: int = None,
    max_count: int = None,
    verbose: bool = True,
) -> None:
    """
    Filter genes based on number of cells or counts.

    Filters genes, that have greater than a max number of genes or less than
    a minimum number of a feature in a given :attr:`~anndata.AnnData.var` columns. Can so far only be used for numerical columns.
    You can run this function on 'n_cells' or 'n_counts' with a previous columns in :attr:`~anndata.AnnData.var`.

    Parameters
    ----------
        adata:
            AnnData object

        qc_var
            column in :attr:`~anndata.AnnData.var` with numerical entries to filter against

        min_count
            Lower bound on number of a given feature to keep gene

        max_count
            Upper bound on number of a given feature to keep gene

        verbose
            Print number of discarded genes

    Returns
    -------
        a filtered :class:`~anndata.AnnData` object inplace

    """
    if qc_var in adata.var.keys():
        if min_count is not None and max_count is not None:
            thr = (adata.var[qc_var] <= max_count) & (min_count <= adata.var[qc_var])
        elif min_count is not None:
            thr = adata.var[qc_var] >= min_count
        elif max_count is not None:
            thr = adata.var[qc_var] <= max_count

        if verbose:
            print(
                f"filtered out {adata.var.shape[0]-thr.sum()} genes based on {qc_var}"
            )

        adata._inplace_subset_var(thr)

    elif qc_var in [
        "n_cells_by_counts",
        "total_counts",
        "mean_counts",
        "pct_dropout_by_counts",
    ]:
        if verbose:
            print(
                "Running `calculate_qc_metrics` for 'n_cells_by_counts','total_counts','mean_counts' or 'pct_dropout_by_counts'"
            )
        calculate_qc_metrics(adata=adata, log1p=False)
        if min_count is not None and max_count is not None:
            thr = (adata.var[qc_var] <= max_count) & (min_count <= adata.var[qc_var])
        elif min_count is not None:
            thr = adata.var[qc_var] >= min_count
        elif max_count is not None:
            thr = adata.var[qc_var] <= max_count
        if verbose:
            print(
                f"filtered out {adata.var.shape[0]-thr.sum()} genes based on {qc_var}"
            )

        adata._inplace_subset_var(thr)
    else:
        print("please check qc_var")


def filter_cells(
    adata: AnnData,
    *,
    qc_var: str,
    min_count: float = None,
    max_count: float = None,
    verbose: bool = True,
) -> None:
    """\
    Filter cell outliers based on counts and numbers of genes expressed.

    Filter cells based on numerical columns in the :attr:`~anndata.AnnData.obs` by selecting those with a feature count greater than a specified maximum or less than a specified minimum.
    It is recommended to run :func:`calculate_qc_metrics` before using this function. You can run this function on n_genes or n_counts before running :func:`calculate_qc_metrics`.

    Parameters
    ----------
        adata:
            AnnData object
        qc_var
            column in .obs with numerical entries to filter against
        min_count
            Lower bound on number of a given feature to keep cell
        max_count
            Upper bound on number of a given feature to keep cell
        verbose
            Print number of discarded cells

    Returns
    -------
       a filtered :class:`~anndata.AnnData` object inplace

    """
    if qc_var in adata.obs.keys():
        inter = np.array
        if min_count is not None and max_count is not None:
            inter = (adata.obs[qc_var] <= max_count) & (min_count <= adata.obs[qc_var])
        elif min_count is not None:
            inter = adata.obs[qc_var] >= min_count
        elif max_count is not None:
            inter = adata.obs[qc_var] <= max_count
        else:
            print("Please specify a cutoff to filter against")
        if verbose:
            print(f"filtered out {adata.obs.shape[0]-inter.sum()} cells")
        adata._inplace_subset_obs(inter)
    elif qc_var in ["n_genes_by_counts", "total_counts"]:
        if verbose:
            print(
                "Running `calculate_qc_metrics` for 'n_cells_by_counts' or 'total_counts'"
            )
        calculate_qc_metrics(adata, log1p=False)
        inter = np.array
        if min_count is not None and max_count is not None:
            inter = (adata.obs[qc_var] <= max_count) & (min_count <= adata.obs[qc_var])
        elif min_count is not None:
            inter = adata.obs[qc_var] >= min_count
        elif max_count is not None:
            inter = adata.obs[qc_var] <= max_count
        else:
            print("Please specify a cutoff to filter against")
        if verbose:
            print(f"filtered out {adata.obs.shape[0]-inter.sum()} cells")
        adata._inplace_subset_obs(inter)
    else:
        print("Please check qc_var.")


def filter_highly_variable(adata: AnnData) -> None:
    """
    Filters the :class:`~anndata.AnnData` object for highly_variable genes. Run highly_varible_genes first.

    Returns
    -------
        updates :class:`~anndata.AnnData` object to only contain highly variable genes.

    """
    if "highly_variable" in adata.var.keys():
        adata._inplace_subset_var(adata.var["highly_variable"])
    else:
        print("Please calculate highly variable genes first")
