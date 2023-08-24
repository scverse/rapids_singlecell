import math
from typing import Union

import cupy as cp
import cupyx as cpx
import numpy as np
from anndata import AnnData
from scanpy.get import _get_obs_rep

from rapids_singlecell.cunnData import cunnData

from ._utils import _check_gpu_X


def calculate_qc_metrics(
    cudata: Union[cunnData, AnnData],
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: Union[str, list] = None,
    log1p: bool = True,
    layer: str = None,
) -> None:
    """\
    Calculates basic qc Parameters. Calculates number of genes per cell (n_genes) and number of counts per cell (n_counts).
    Loosly based on calculate_qc_metrics from scanpy [Wolf et al. 2018]. Updates :attr:`.obs` and :attr:`.var`  with columns with qc data.

    Parameters
    ----------
        cudata
            cunnData object
        expr_type
            Name of kind of values in X.
        var_type
            The kind of thing the variables are.
        qc_vars
            Keys for boolean columns of :attr:`.var` which identify variables you could want to control for (e.g. Mito).
            Run flag_gene_family first
        log1p
            Set to `False` to skip computing `log1p` transformed annotations.
        layer
            If provided, use :attr:`.layers` for expression values instead of :attr:`.X`.

    Returns
    -------
        adds the following columns in :attr:`.obs` :
            `total_{var_type}_by_{expr_type}`
                E.g. 'total_genes_by_counts'. Number of genes with positive counts in a cell.
            `total_{expr_type}`
                E.g. 'total_counts'. Total number of counts for a cell.
            for `qc_var` in `qc_vars`
                `total_{expr_type}_{qc_var}`
                    number of counts per qc_var (e.g total counts mitochondrial genes)
                `pct_{expr_type}_{qc_var}`
                    Proportion of counts of qc_var (percent of counts mitochondrial genes)

        adds the following columns in :attr:`.var` :
            `total_{expr_type}`
                E.g. 'total_counts'. Sum of counts for a gene.
            `n_genes_by_{expr_type}`
                E.g. 'n_cells_by_counts'. Number of cells this expression is measured in.
            `mean_{expr_type}`
                E.g. "mean_counts". Mean expression over all cells.
            `pct_dropout_by_{expr_type}`
                E.g. 'pct_dropout_by_counts'. Percentage of cells this feature does not appear in.

    """

    X = _get_obs_rep(cudata, layer=layer)

    if isinstance(X, AnnData):
        _check_gpu_X(X)

    sums_cells = cp.zeros(X.shape[0], dtype=X.dtype)
    sums_genes = cp.zeros(X.shape[1], dtype=X.dtype)
    cell_ex = cp.zeros(X.shape[0], dtype=cp.int32)
    gene_ex = cp.zeros(X.shape[1], dtype=cp.int32)
    if cpx.scipy.sparse.issparse(X):
        if cpx.scipy.sparse.isspmatrix_csr(X):
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
        elif cpx.scipy.sparse.isspmatrix_csc(X):
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
    cudata.var[f"n_cells_by_{expr_type}"] = cp.asnumpy(gene_ex)
    cudata.var[f"total_{expr_type}"] = cp.asnumpy(sums_genes)
    mean_array = sums_genes / cudata.n_obs
    cudata.var[f"mean_{expr_type}"] = cp.asnumpy(mean_array)
    cudata.var[f"pct_dropout_by_{expr_type}"] = cp.asnumpy(
        (1 - gene_ex / cudata.n_obs) * 100
    )
    if log1p:
        cudata.var[f"log1p_total_{expr_type}"] = cp.asnumpy(cp.log1p(sums_genes))
        cudata.var[f"log1p_mean_{expr_type}"] = cp.asnumpy(cp.log1p(mean_array))
    # .obs
    cudata.obs[f"n_{var_type}_by_{expr_type}"] = cp.asnumpy(cell_ex)
    cudata.obs[f"total_{expr_type}"] = cp.asnumpy(sums_cells)
    if log1p:
        cudata.obs[f"log1p_n_{var_type}_by_{expr_type}"] = cp.asnumpy(cp.log1p(cell_ex))
        cudata.obs[f"log1p_total_{expr_type}"] = cp.asnumpy(cp.log1p(sums_cells))

    if qc_vars:
        if isinstance(qc_vars, str):
            qc_vars = [qc_vars]
        for qc_var in qc_vars:
            sums_cells_sub = cp.zeros(X.shape[0], dtype=X.dtype)
            mask = cp.array(cudata.var[qc_var], dtype=cp.bool_)
            if cpx.scipy.sparse.issparse(X):
                if cpx.scipy.sparse.isspmatrix_csr(X):
                    from ._kernels._qc_kernels import _sparse_qc_csr_sub

                    block = (32,)
                    grid = (int(math.ceil(X.shape[0] / block[0])),)
                    sparse_qc_csr_sub = _sparse_qc_csr_sub(X.data.dtype)
                    sparse_qc_csr_sub(
                        grid,
                        block,
                        (X.indptr, X.indices, X.data, sums_cells_sub, mask, X.shape[0]),
                    )
                elif cpx.scipy.sparse.isspmatrix_csc(X):
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
            cudata.obs[f"total_{expr_type}_{qc_var}"] = cp.asnumpy(sums_cells_sub)
            cudata.obs[f"pct_{expr_type}_{qc_var}"] = cp.asnumpy(
                sums_cells_sub / sums_cells * 100
            )
            if log1p:
                cudata.obs[f"log1p_total_{expr_type}_{qc_var}"] = cp.asnumpy(
                    cp.log1p(sums_cells_sub)
                )


def flag_gene_family(
    cudata: cunnData,
    gene_family_name: str,
    gene_family_prefix: str = None,
    gene_list: list = None,
) -> None:
    """
    Flags a gene or gene_familiy in .var with boolean. (e.g all mitochondrial genes).
    Please only choose gene_family prefix or gene_list

    Parameters
    ----------
        cudata
            cunnData object

        gene_family_name
            name of colums in .var where you want to store informationa as a boolean

        gene_family_prefix
            prefix of the gene familiy (eg. mt- for all mitochondrial genes in mice)

        gene_list
            list of genes to flag in `.var`

    Returns
    -------
        adds the boolean column in `.var`

    """
    if gene_family_prefix:
        cudata.var[gene_family_name] = cp.asnumpy(
            cudata.var.index.str.startswith(gene_family_prefix)
        ).ravel()
    if gene_list:
        cudata.var[gene_family_name] = cp.asnumpy(
            cudata.var.index.isin(gene_list)
        ).ravel()


def filter_genes(
    cudata: cunnData,
    qc_var: str = "n_cells_by_counts",
    min_count: int = None,
    max_count: int = None,
    verbose: bool = True,
) -> None:
    """
    Filter genes based on number of cells or counts.

    Filters genes, that have greater than a max number of genes or less than
    a minimum number of a feature in a given :attr:`.var` columns. Can so far only be used for numerical columns.
    You can run this function on 'n_cells' or 'n_counts' with a previous columns in :attr:`.var`.

    Parameters
    ----------
        cudata:
            cunnData object

        qc_var
            column in :attr:`.var` with numerical entries to filter against

        min_count
            Lower bound on number of a given feature to keep gene

        max_count
            Upper bound on number of a given feature to keep gene

        verbose
            Print number of discarded genes

    Returns
    -------
        a filtered :class:`~rapids_singlecell.cunnData.cunnData` object inplace

    """
    if qc_var in cudata.var.keys():
        if min_count is not None and max_count is not None:
            thr = np.where(
                (cudata.var[qc_var] <= max_count) & (min_count <= cudata.var[qc_var])
            )[0]
        elif min_count is not None:
            thr = np.where(cudata.var[qc_var] >= min_count)[0]
        elif max_count is not None:
            thr = np.where(cudata.var[qc_var] <= max_count)[0]

        if verbose:
            print(
                f"filtered out {cudata.var.shape[0]-thr.shape[0]} genes based on {qc_var}"
            )

        cudata._inplace_subset_var(thr)

    elif qc_var in [
        "n_cells_by_counts",
        "total_counts",
        "mean_counts",
        "pct_dropout_by_counts",
    ]:
        print(
            "Running `calculate_qc_metrics` for 'n_cells_by_counts','total_counts','mean_counts' or 'pct_dropout_by_counts'"
        )
        calculate_qc_metrics(cudata=cudata, log1p=False)
        if min_count is not None and max_count is not None:
            thr = np.where(
                (cudata.var[qc_var] <= max_count) & (min_count <= cudata.var[qc_var])
            )[0]
        elif min_count is not None:
            thr = np.where(cudata.var[qc_var] >= min_count)[0]
        elif max_count is not None:
            thr = np.where(cudata.var[qc_var] <= max_count)[0]

        if verbose:
            print(
                f"filtered out {cudata.var.shape[0]-thr.shape[0]} genes based on {qc_var}"
            )

        cudata._inplace_subset_var(thr)
    else:
        print("please check qc_var")


def filter_cells(
    cudata: cunnData,
    qc_var: str,
    min_count: float = None,
    max_count: float = None,
    verbose: bool = True,
) -> None:
    """\
    Filter cell outliers based on counts and numbers of genes expressed.

    Filter cells based on numerical columns in the :attr:`.obs` by selecting those with a feature count greater than a specified maximum or less than a specified minimum.
    It is recommended to run :func:`calculate_qc_metrics` before using this function. You can run this function on n_genes or n_counts before running :func:`calculate_qc_metrics`.

    Parameters
    ----------
        cudata:
            cunnData object
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
       a filtered :class:`~rapids_singlecell.cunnData.cunnData` object inplace

    """
    if qc_var in cudata.obs.keys():
        inter = np.array
        if min_count is not None and max_count is not None:
            inter = np.where(
                (cudata.obs[qc_var] < max_count) & (min_count < cudata.obs[qc_var])
            )[0]
        elif min_count is not None:
            inter = np.where(cudata.obs[qc_var] > min_count)[0]
        elif max_count is not None:
            inter = np.where(cudata.obs[qc_var] < max_count)[0]
        else:
            print("Please specify a cutoff to filter against")
        if verbose:
            print(f"filtered out {cudata.obs.shape[0]-inter.shape[0]} cells")
        cudata._inplace_subset_obs(inter)
    elif qc_var in ["n_genes_by_counts", "total_counts"]:
        print(
            "Running `calculate_qc_metrics` for 'n_cells_by_counts' or 'total_counts'"
        )
        calculate_qc_metrics(cudata, log1p=False)
        inter = np.array
        if min_count is not None and max_count is not None:
            inter = np.where(
                (cudata.obs[qc_var] < max_count) & (min_count < cudata.obs[qc_var])
            )[0]
        elif min_count is not None:
            inter = np.where(cudata.obs[qc_var] > min_count)[0]
        elif max_count is not None:
            inter = np.where(cudata.obs[qc_var] < max_count)[0]
        else:
            print("Please specify a cutoff to filter against")
        if verbose:
            print(f"filtered out {cudata.obs.shape[0]-inter.shape[0]} cells")
        cudata._inplace_subset_obs(inter)
    else:
        print("Please check qc_var.")


def filter_highly_variable(cudata: cunnData) -> None:
    """
    Filters the :class:`~rapids_singlecell.cunnData.cunnData` object for highly_variable genes. Run highly_varible_genes first.

    Returns
    -------
        updates :class:`~rapids_singlecell.cunnData.cunnData` object to only contain highly variable genes.

    """
    if "highly_variable" in cudata.var.keys():
        cudata._inplace_subset_var(cudata.var["highly_variable"])
    else:
        print("Please calculate highly variable genes first")