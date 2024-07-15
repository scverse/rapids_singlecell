from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
from cuml.dask.common.part_utils import _extract_partitions
from cuml.internals.memory_utils import with_cupy_rmm
from cupyx.scipy import sparse
from scanpy.get import _get_obs_rep

from rapids_singlecell._compat import DaskArray, DaskClient, _get_dask_client

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
    client: DaskClient | None = None,
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
        client
            Dask client to use for computation. If `None`, the default client is used. Only used if `X` is a Dask array.

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

    _check_gpu_X(X, allow_dask=True)

    sums_cells, sums_genes, cell_ex, gene_ex = _first_pass_qc(X, client=client)
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
            mask = cp.array(adata.var[qc_var], dtype=cp.bool_)
            sums_cells_sub = _second_pass_qc(X, mask, client=client)

            adata.obs[f"total_{expr_type}_{qc_var}"] = cp.asnumpy(sums_cells_sub)
            adata.obs[f"pct_{expr_type}_{qc_var}"] = cp.asnumpy(
                sums_cells_sub / sums_cells * 100
            )
            if log1p:
                adata.obs[f"log1p_total_{expr_type}_{qc_var}"] = cp.asnumpy(
                    cp.log1p(sums_cells_sub)
                )


def _first_pass_qc(X, client=None):
    if isinstance(X, DaskArray):
        return _first_pass_qc_dask(X, client=client)

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
    return sums_cells, sums_genes, cell_ex, gene_ex


@with_cupy_rmm
def _first_pass_qc_dask(X, client=None):
    import dask

    client = _get_dask_client(client)

    if isinstance(X._meta, sparse.csr_matrix):
        from ._kernels._qc_kernels import _sparse_qc_csr

        sparse_qc_csr = _sparse_qc_csr(X.dtype)
        sparse_qc_csr.compile()

        def __qc_calc(X_part):
            sums_cells = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            sums_genes = cp.zeros((X_part.shape[1], 1), dtype=X_part.dtype)
            cell_ex = cp.zeros(X_part.shape[0], dtype=cp.int32)
            gene_ex = cp.zeros((X_part.shape[1], 1), dtype=cp.int32)
            block = (32,)
            grid = (int(math.ceil(X_part.shape[0] / block[0])),)
            sparse_qc_csr(
                grid,
                block,
                (
                    X_part.indptr,
                    X_part.indices,
                    X_part.data,
                    sums_cells,
                    sums_genes,
                    cell_ex,
                    gene_ex,
                    X_part.shape[0],
                ),
            )
            return sums_cells, sums_genes, cell_ex, gene_ex

    elif isinstance(X._meta, cp.ndarray):
        from ._kernels._qc_kernels import _sparse_qc_dense

        sparse_qc_dense = _sparse_qc_dense(X.dtype)
        sparse_qc_dense.compile()

        def __qc_calc(X_part):
            sums_cells = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            sums_genes = cp.zeros((X_part.shape[1], 1), dtype=X_part.dtype)
            cell_ex = cp.zeros(X_part.shape[0], dtype=cp.int32)
            gene_ex = cp.zeros((X_part.shape[1], 1), dtype=cp.int32)
            if not X_part.flags.c_contiguous:
                X_part = cp.asarray(X_part, order="C")
            block = (16, 16)
            grid = (
                int(math.ceil(X_part.shape[0] / block[0])),
                int(math.ceil(X_part.shape[1] / block[1])),
            )
            sparse_qc_dense(
                grid,
                block,
                (
                    X_part,
                    sums_cells,
                    sums_genes,
                    cell_ex,
                    gene_ex,
                    X_part.shape[0],
                    X_part.shape[1],
                ),
            )
            return sums_cells, sums_genes, cell_ex, gene_ex
    else:
        raise ValueError(
            "Please use a cupy csr_matrix or cp.ndarray. csc_matrix are not supported with dask."
        )

    parts = client.sync(_extract_partitions, X)
    futures = [client.submit(__qc_calc, part, workers=[w]) for w, part in parts]
    # Gather results from futures
    results = client.gather(futures)

    # Initialize lists to hold the Dask arrays
    sums_cells_objs = []
    sums_genes_objs = []
    cell_ex_objs = []
    gene_ex_objs = []

    # Process each result
    for sums_cells, sums_genes, cell_ex, gene_ex in results:
        # Append the arrays to their respective lists as Dask arrays
        sums_cells_objs.append(
            dask.array.from_array(sums_cells, chunks=sums_cells.shape)
        )
        sums_genes_objs.append(
            dask.array.from_array(sums_genes, chunks=sums_genes.shape)
        )
        cell_ex_objs.append(dask.array.from_array(cell_ex, chunks=cell_ex.shape))
        gene_ex_objs.append(dask.array.from_array(gene_ex, chunks=gene_ex.shape))
    sums_cells = dask.array.concatenate(sums_cells_objs)
    sums_genes = dask.array.concatenate(sums_genes_objs, axis=1).sum(axis=1)
    cell_ex = dask.array.concatenate(cell_ex_objs)
    gene_ex = dask.array.concatenate(gene_ex_objs, axis=1).sum(axis=1)
    sums_cells, sums_genes, cell_ex, gene_ex = dask.compute(
        sums_cells, sums_genes, cell_ex, gene_ex
    )
    return sums_cells.ravel(), sums_genes.ravel(), cell_ex.ravel(), gene_ex.ravel()


def _second_pass_qc(X, mask, client=None):
    if isinstance(X, DaskArray):
        return _second_pass_qc_dask(X, mask, client=client)
    sums_cells_sub = cp.zeros(X.shape[0], dtype=X.dtype)
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
    return sums_cells_sub


@with_cupy_rmm
def _second_pass_qc_dask(X, mask, client=None):
    client = _get_dask_client(client)

    if isinstance(X._meta, sparse.csr_matrix):
        from ._kernels._qc_kernels import _sparse_qc_csr_sub

        sparse_qc_csr = _sparse_qc_csr_sub(X.dtype)
        sparse_qc_csr.compile()

        def __qc_calc(X_part):
            sums_cells_sub = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            block = (32,)
            grid = (int(math.ceil(X_part.shape[0] / block[0])),)
            sparse_qc_csr(
                grid,
                block,
                (
                    X_part.indptr,
                    X_part.indices,
                    X_part.data,
                    sums_cells_sub,
                    mask,
                    X_part.shape[0],
                ),
            )
            return sums_cells_sub

    elif isinstance(X._meta, cp.ndarray):
        from ._kernels._qc_kernels import _sparse_qc_dense_sub

        sparse_qc_dense = _sparse_qc_dense_sub(X.dtype)
        sparse_qc_dense.compile()

        def __qc_calc(X_part):
            sums_cells_sub = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            if not X_part.flags.c_contiguous:
                X_part = cp.asarray(X_part, order="C")
            block = (16, 16)
            grid = (
                int(math.ceil(X_part.shape[0] / block[0])),
                int(math.ceil(X_part.shape[1] / block[1])),
            )
            sparse_qc_dense(
                grid,
                block,
                (X_part, sums_cells_sub, mask, X_part.shape[0], X_part.shape[1]),
            )
            return sums_cells_sub

    sums_cells_sub = X.map_blocks(
        __qc_calc, dtype=X.dtype, meta=cp.array([]), drop_axis=1
    ).compute()
    return sums_cells_sub.ravel()
