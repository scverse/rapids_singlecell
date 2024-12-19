from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
from cuml.internals.memory_utils import with_cupy_rmm
from cupyx.scipy import sparse
from scanpy.get import _get_obs_rep

from rapids_singlecell._compat import DaskArray

from ._utils import _check_gpu_X

if TYPE_CHECKING:
    from anndata import AnnData

    from rapids_singlecell._utils import ArrayTypesDask


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

    _check_gpu_X(X, allow_dask=True)

    sums_cells, sums_genes, genes_per_cell, cells_per_gene = _basic_qc(X)
    # .var
    adata.var[f"n_cells_by_{expr_type}"] = cp.asnumpy(cells_per_gene)
    adata.var[f"total_{expr_type}"] = cp.asnumpy(sums_genes)
    mean_array = sums_genes / adata.n_obs
    adata.var[f"mean_{expr_type}"] = cp.asnumpy(mean_array)
    adata.var[f"pct_dropout_by_{expr_type}"] = cp.asnumpy(
        (1 - cells_per_gene / adata.n_obs) * 100
    )
    if log1p:
        adata.var[f"log1p_total_{expr_type}"] = cp.asnumpy(cp.log1p(sums_genes))
        adata.var[f"log1p_mean_{expr_type}"] = cp.asnumpy(cp.log1p(mean_array))
    # .obs
    adata.obs[f"n_{var_type}_by_{expr_type}"] = cp.asnumpy(genes_per_cell)
    adata.obs[f"total_{expr_type}"] = cp.asnumpy(sums_cells)
    if log1p:
        adata.obs[f"log1p_n_{var_type}_by_{expr_type}"] = cp.asnumpy(
            cp.log1p(genes_per_cell)
        )
        adata.obs[f"log1p_total_{expr_type}"] = cp.asnumpy(cp.log1p(sums_cells))

    if qc_vars:
        if isinstance(qc_vars, str):
            qc_vars = [qc_vars]
        for qc_var in qc_vars:
            mask = cp.array(adata.var[qc_var], dtype=cp.bool_)
            sums_cells_sub = _geneset_qc(X, mask)

            adata.obs[f"total_{expr_type}_{qc_var}"] = cp.asnumpy(sums_cells_sub)
            adata.obs[f"pct_{expr_type}_{qc_var}"] = cp.asnumpy(
                sums_cells_sub / sums_cells * 100
            )
            if log1p:
                adata.obs[f"log1p_total_{expr_type}_{qc_var}"] = cp.asnumpy(
                    cp.log1p(sums_cells_sub)
                )


def _basic_qc(
    X: ArrayTypesDask,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    if isinstance(X, DaskArray):
        return _basic_qc_dask(X)

    sums_cells = cp.zeros(X.shape[0], dtype=X.dtype)
    sums_genes = cp.zeros(X.shape[1], dtype=X.dtype)
    genes_per_cell = cp.zeros(X.shape[0], dtype=cp.int32)
    cells_per_gene = cp.zeros(X.shape[1], dtype=cp.int32)
    if sparse.issparse(X):
        if sparse.isspmatrix_csr(X):
            from ._kernels._qc_kernels import _sparse_qc_csr

            block = (32,)
            grid = (int(math.ceil(X.shape[0] / block[0])),)
            call_shape = X.shape[0]
            sparse_qc_kernel = _sparse_qc_csr(X.data.dtype)

        elif sparse.isspmatrix_csc(X):
            from ._kernels._qc_kernels import _sparse_qc_csc

            block = (32,)
            grid = (int(math.ceil(X.shape[1] / block[0])),)
            call_shape = X.shape[1]
            sparse_qc_kernel = _sparse_qc_csc(X.data.dtype)

        else:
            raise ValueError("Please use a csr or csc matrix")
        sparse_qc_kernel(
            grid,
            block,
            (
                X.indptr,
                X.indices,
                X.data,
                sums_cells,
                sums_genes,
                genes_per_cell,
                cells_per_gene,
                call_shape,
            ),
        )
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
            (
                X,
                sums_cells,
                sums_genes,
                genes_per_cell,
                cells_per_gene,
                X.shape[0],
                X.shape[1],
            ),
        )
    return sums_cells, sums_genes, genes_per_cell, cells_per_gene


@with_cupy_rmm
def _basic_qc_dask(
    X: DaskArray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    import dask

    if isinstance(X._meta, sparse.csr_matrix):
        from ._kernels._qc_kernels_dask import (
            _sparse_qc_csr_dask_cells,
            _sparse_qc_csr_dask_genes,
        )

        sparse_qc_csr_cells = _sparse_qc_csr_dask_cells(X.dtype)
        sparse_qc_csr_cells.compile()

        def __qc_calc_1(X_part):
            sums_cells = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            genes_per_cell = cp.zeros(X_part.shape[0], dtype=cp.int32)
            block = (32,)
            grid = (int(math.ceil(X_part.shape[0] / block[0])),)

            sparse_qc_csr_cells(
                grid,
                block,
                (
                    X_part.indptr,
                    X_part.indices,
                    X_part.data,
                    sums_cells,
                    genes_per_cell,
                    X_part.shape[0],
                ),
            )
            return cp.stack([sums_cells, genes_per_cell.astype(X_part.dtype)], axis=1)

        sparse_qc_csr_genes = _sparse_qc_csr_dask_genes(X.dtype)
        sparse_qc_csr_genes.compile()

        def __qc_calc_2(X_part):
            sums_genes = cp.zeros(X_part.shape[1], dtype=X_part.dtype)
            cells_per_gene = cp.zeros(X_part.shape[1], dtype=cp.int32)
            block = (32,)
            grid = (int(math.ceil(X_part.nnz / block[0])),)
            sparse_qc_csr_genes(
                grid,
                block,
                (
                    X_part.indices,
                    X_part.data,
                    sums_genes,
                    cells_per_gene,
                    X_part.nnz,
                ),
            )
            return cp.vstack([sums_genes, cells_per_gene.astype(X_part.dtype)])[
                None, ...
            ]

    elif isinstance(X._meta, cp.ndarray):
        from ._kernels._qc_kernels_dask import (
            _sparse_qc_dense_cells,
            _sparse_qc_dense_genes,
        )

        sparse_qc_dense_cells = _sparse_qc_dense_cells(X.dtype)
        sparse_qc_dense_cells.compile()

        def __qc_calc_1(X_part):
            sums_cells = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            genes_per_cell = cp.zeros(X_part.shape[0], dtype=cp.int32)
            if not X_part.flags.c_contiguous:
                X_part = cp.asarray(X_part, order="C")
            block = (16, 16)
            grid = (
                int(math.ceil(X_part.shape[0] / block[0])),
                int(math.ceil(X_part.shape[1] / block[1])),
            )
            sparse_qc_dense_cells(
                grid,
                block,
                (
                    X_part,
                    sums_cells,
                    genes_per_cell,
                    X_part.shape[0],
                    X_part.shape[1],
                ),
            )
            return cp.stack([sums_cells, genes_per_cell.astype(X_part.dtype)], axis=1)

        sparse_qc_dense_genes = _sparse_qc_dense_genes(X.dtype)
        sparse_qc_dense_genes.compile()

        def __qc_calc_2(X_part):
            sums_genes = cp.zeros((X_part.shape[1]), dtype=X_part.dtype)
            cells_per_gene = cp.zeros((X_part.shape[1]), dtype=cp.int32)
            if not X_part.flags.c_contiguous:
                X_part = cp.asarray(X_part, order="C")
            block = (16, 16)
            grid = (
                int(math.ceil(X_part.shape[0] / block[0])),
                int(math.ceil(X_part.shape[1] / block[1])),
            )
            sparse_qc_dense_genes(
                grid,
                block,
                (
                    X_part,
                    sums_genes,
                    cells_per_gene,
                    X_part.shape[0],
                    X_part.shape[1],
                ),
            )
            return cp.vstack([sums_genes, cells_per_gene.astype(X_part.dtype)])[
                None, ...
            ]
    else:
        raise ValueError(
            "Please use a cupy csr_matrix or cp.ndarray. csc_matrix are not supported with dask."
        )

    cell_results = X.map_blocks(
        __qc_calc_1,
        chunks=(X.chunks[0], (2,)),
        dtype=X.dtype,
        meta=cp.empty((0, 2), dtype=X.dtype),
    )
    sums_cells = cell_results[:, 0]
    genes_per_cell = cell_results[:, 1]

    n_blocks = X.blocks.size
    sums_genes, cells_per_gene = X.map_blocks(
        __qc_calc_2,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (2,), (X.shape[1],)),
        dtype=X.dtype,
        meta=cp.array([]),
    ).sum(axis=0)

    sums_cells, genes_per_cell, sums_genes, cells_per_gene = dask.compute(
        sums_cells, genes_per_cell, sums_genes, cells_per_gene
    )

    return (
        sums_cells.ravel(),
        sums_genes.ravel(),
        genes_per_cell.ravel().astype(cp.int32),
        cells_per_gene.ravel().astype(cp.int32),
    )


def _geneset_qc(X: ArrayTypesDask, mask: cp.ndarray) -> cp.ndarray:
    if isinstance(X, DaskArray):
        return _geneset_qc_dask(X, mask)
    sums_cells_sub = cp.zeros(X.shape[0], dtype=X.dtype)
    if sparse.issparse(X):
        if sparse.isspmatrix_csr(X):
            from ._kernels._qc_kernels import _sparse_qc_csr_sub

            block = (32,)
            grid = (int(math.ceil(X.shape[0] / block[0])),)
            call_shape = X.shape[0]
            sparse_qc_sub = _sparse_qc_csr_sub(X.data.dtype)

        elif sparse.isspmatrix_csc(X):
            from ._kernels._qc_kernels import _sparse_qc_csc_sub

            block = (32,)
            grid = (int(math.ceil(X.shape[1] / block[0])),)
            call_shape = X.shape[1]
            sparse_qc_sub = _sparse_qc_csc_sub(X.data.dtype)

        sparse_qc_sub(
            grid,
            block,
            (X.indptr, X.indices, X.data, sums_cells_sub, mask, call_shape),
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
def _geneset_qc_dask(X: DaskArray, mask: cp.ndarray) -> cp.ndarray:
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
