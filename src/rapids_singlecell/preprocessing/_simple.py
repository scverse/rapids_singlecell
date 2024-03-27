from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cupyx.scipy import sparse
from scanpy.get import _get_obs_rep

from ._utils import _check_gpu_X

if TYPE_CHECKING:
    from anndata import AnnData


def calculate_qc_metrics(
    adata: AnnData,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: str | list = None,
    log1p: bool = True,
    layer: str = None,
) -> None:
    """\
    Calculates basic qc Parameters. Calculates number of genes per cell (n_genes) and number of counts per cell (n_counts).
    Loosely based on calculate_qc_metrics from scanpy [Wolf et al. 2018]. Updates :attr:`~anndata.AnnData.obs` and :attr:`~anndata.AnnData.var`  with columns with qc data.

    Parameters
    ----------
        adata
            AnnData object
        expr_type
            Name of kind of values in X.
        var_type
            The kind of thing the variables are.
        qc_vars
            Keys for boolean columns of :attr:`~anndata.AnnData.var` which identify variables you could want to control for (e.g. Mito).
            Run flag_gene_family first
        log1p
            Set to `False` to skip computing `log1p` transformed annotations.
        layer
            If provided, use :attr:`~anndata.AnnData.layers` for expression values instead of :attr:`~anndata.AnnData.X`.

    Returns
    -------
        adds the following columns in :attr:`~anndata.AnnData.obs` :
            `total_{var_type}_by_{expr_type}`
                E.g. 'total_genes_by_counts'. Number of genes with positive counts in a cell.
            `total_{expr_type}`
                E.g. 'total_counts'. Total number of counts for a cell.
            for `qc_var` in `qc_vars`
                `total_{expr_type}_{qc_var}`
                    number of counts per qc_var (e.g total counts mitochondrial genes)
                `pct_{expr_type}_{qc_var}`
                    Proportion of counts of qc_var (percent of counts mitochondrial genes)

        adds the following columns in :attr:`~anndata.AnnData.var` :
            `total_{expr_type}`
                E.g. 'total_counts'. Sum of counts for a gene.
            `n_genes_by_{expr_type}`
                E.g. 'n_cells_by_counts'. Number of cells this expression is measured in.
            `mean_{expr_type}`
                E.g. "mean_counts". Mean expression over all cells.
            `pct_dropout_by_{expr_type}`
                E.g. 'pct_dropout_by_counts'. Percentage of cells this feature does not appear in.

    """

    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X)

    sums_cells = cp.zeros(X.shape[0], dtype=X.dtype)
    sums_genes = cp.zeros(X.shape[1], dtype=X.dtype)
    cell_ex = cp.zeros(X.shape[0], dtype=cp.int32)
    gene_ex = cp.zeros(X.shape[1], dtype=cp.int32)
    if sparse.issparse(X):
        if sparse.isspmatrix_csr(X):
            from ._kernels._qc_kernels import _sparse_qc_csr

            block = (32,)
            grid = (int(math.ceil(X.shape[0] / block[0])),)
            sparse_qc_csr = _sparse_qc_csr(X.data.dtype)
            sparse_qc_csr(
                grid,
                block,
                (
                    X.indptr,
                    X.indices,
                    X.data,
                    sums_cells,
                    sums_genes,
                    cell_ex,
                    gene_ex,
                    X.shape[0],
                ),
            )
        elif sparse.isspmatrix_csc(X):
            from ._kernels._qc_kernels import _sparse_qc_csc

            block = (32,)
            grid = (int(math.ceil(X.shape[1] / block[0])),)
            sparse_qc_csc = _sparse_qc_csc(X.data.dtype)
            sparse_qc_csc(
                grid,
                block,
                (
                    X.indptr,
                    X.indices,
                    X.data,
                    sums_cells,
                    sums_genes,
                    cell_ex,
                    gene_ex,
                    X.shape[1],
                ),
            )
        else:
            raise ValueError("Please use a csr or csc matrix")
    else:
        from ._kernels._qc_kernels import _sparse_qc_dense

        if not X.flags.c_contiguous:
            X = cp.asarray(X, order="C")
        block = (16, 16)
        grid = (
            int(math.ceil(X.shape[0] / block[0])),
            int(math.ceil(X.shape[1] / block[1])),
        )
        sparse_qc_dense = _sparse_qc_dense(X.dtype)
        sparse_qc_dense(
            grid,
            block,
            (X, sums_cells, sums_genes, cell_ex, gene_ex, X.shape[0], X.shape[1]),
        )

    # .var
    adata.var[f"n_cells_by_{expr_type}"] = cp.asnumpy(gene_ex)
    adata.var[f"total_{expr_type}"] = cp.asnumpy(sums_genes)
    mean_array = sums_genes / adata.n_obs
    adata.var[f"mean_{expr_type}"] = cp.asnumpy(mean_array)
    adata.var[f"pct_dropout_by_{expr_type}"] = cp.asnumpy(
        (1 - gene_ex / adata.n_obs) * 100
    )
    if log1p:
        adata.var[f"log1p_total_{expr_type}"] = cp.asnumpy(cp.log1p(sums_genes))
        adata.var[f"log1p_mean_{expr_type}"] = cp.asnumpy(cp.log1p(mean_array))
    # .obs
    adata.obs[f"n_{var_type}_by_{expr_type}"] = cp.asnumpy(cell_ex)
    adata.obs[f"total_{expr_type}"] = cp.asnumpy(sums_cells)
    if log1p:
        adata.obs[f"log1p_n_{var_type}_by_{expr_type}"] = cp.asnumpy(cp.log1p(cell_ex))
        adata.obs[f"log1p_total_{expr_type}"] = cp.asnumpy(cp.log1p(sums_cells))

    if qc_vars:
        if isinstance(qc_vars, str):
            qc_vars = [qc_vars]
        for qc_var in qc_vars:
            sums_cells_sub = cp.zeros(X.shape[0], dtype=X.dtype)
            mask = cp.array(adata.var[qc_var], dtype=cp.bool_)
            if sparse.issparse(X):
                if sparse.isspmatrix_csr(X):
                    from ._kernels._qc_kernels import _sparse_qc_csr_sub

                    block = (32,)
                    grid = (int(math.ceil(X.shape[0] / block[0])),)
                    sparse_qc_csr_sub = _sparse_qc_csr_sub(X.data.dtype)
                    sparse_qc_csr_sub(
                        grid,
                        block,
                        (X.indptr, X.indices, X.data, sums_cells_sub, mask, X.shape[0]),
                    )
                elif sparse.isspmatrix_csc(X):
                    from ._kernels._qc_kernels import _sparse_qc_csc_sub

                    block = (32,)
                    grid = (int(math.ceil(X.shape[1] / block[0])),)
                    sparse_qc_csc_sub = _sparse_qc_csc_sub(X.data.dtype)
                    sparse_qc_csc_sub(
                        grid,
                        block,
                        (X.indptr, X.indices, X.data, sums_cells_sub, mask, X.shape[1]),
                    )

            else:
                from ._kernels._qc_kernels import _sparse_qc_dense_sub

                block = (16, 16)
                grid = (
                    int(math.ceil(X.shape[0] / block[0])),
                    int(math.ceil(X.shape[1] / block[1])),
                )
                sparse_qc_dense_sub = _sparse_qc_dense_sub(X.dtype)
                sparse_qc_dense_sub(
                    grid, block, (X, sums_cells_sub, mask, X.shape[0], X.shape[1])
                )
            adata.obs[f"total_{expr_type}_{qc_var}"] = cp.asnumpy(sums_cells_sub)
            adata.obs[f"pct_{expr_type}_{qc_var}"] = cp.asnumpy(
                sums_cells_sub / sums_cells * 100
            )
            if log1p:
                adata.obs[f"log1p_total_{expr_type}_{qc_var}"] = cp.asnumpy(
                    cp.log1p(sums_cells_sub)
                )


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
