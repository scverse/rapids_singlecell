from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    get_args,
)

import cupy as cp
from anndata import AnnData
from cupyx.scipy import sparse as cp_sparse
from scanpy._utils import _resolve_axis
from scanpy.get._aggregated import _combine_categories

from rapids_singlecell._compat import (
    DaskArray,
    _meta_dense,
)
from rapids_singlecell.get import _check_mask
from rapids_singlecell.preprocessing._utils import _check_gpu_X

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

Array = Union[cp.ndarray, cp_sparse.csc_matrix, cp_sparse.csr_matrix]  # noqa: UP007
AggType = Literal["count_nonzero", "mean", "sum", "var"]


class Aggregate:
    """\
    Functionality for generic grouping and aggregating.

    There is currently support for count_nonzero, sum, mean, and variance.

    Params
    ------
    groupby
        :class:`~pandas.Categorical` containing values for grouping by.
    data
        Data matrix for aggregation.
    mask
        Mask to be used for aggregation.
    """

    def __init__(
        self,
        groupby: pd.Categorical,
        data: Array,
        *,
        mask: NDArray[np.bool_] | None = None,
    ) -> None:
        self.mask = mask
        self.groupby = cp.array(groupby.codes, dtype=cp.int32)
        self.n_cells = cp.array(cp.bincount(self.groupby), dtype=cp.float64).reshape(
            -1, 1
        )
        self.data = data

    groupby: cp.ndarray
    data: Array

    def _get_mask(self):
        if self.mask is not None:
            return cp.array(self.mask)
        else:
            return cp.ones(self.data.shape[0], dtype=bool)

    def count_mean_var_dask(self, dof: int = 1, split_every: int = 2):
        """
        This function is used to calculate the sum, mean, and variance of the data matrix.
        It automatically detects sparse vs dense matrices and uses the appropriate
        CUDA kernel for aggregation.
        """
        import dask.array as da

        assert dof >= 0
        from ._kernels._aggr_kernels import (
            _get_aggr_dense_kernel_C,
            _get_aggr_sparse_kernel,
        )

        if isinstance(self.data._meta, cp.ndarray):
            kernel = _get_aggr_dense_kernel_C(self.data.dtype)
            is_sparse = False
        else:
            kernel = _get_aggr_sparse_kernel(self.data.dtype)
            is_sparse = True

        kernel.compile()
        n_groups = self.n_cells.shape[0]

        def __aggregate_dask(X_part, mask_part, groupby_part):
            out = cp.zeros((1, 3, n_groups, self.data.shape[1]), dtype=cp.float64)
            threads_per_block = 512

            if is_sparse:
                # Sparse matrix kernel parameters
                grid = (X_part.shape[0],)
                kernel_args = (
                    X_part.indptr,
                    X_part.indices,
                    X_part.data,
                )
            else:
                # Dense matrix kernel parameters
                N = X_part.shape[0] * X_part.shape[1]

                blocks = min(
                    (N + threads_per_block - 1) // threads_per_block,
                    cp.cuda.Device().attributes["MultiProcessorCount"] * 8,
                )
                grid = (blocks,)
                kernel_args = (X_part,)

            kernel(
                grid,
                (threads_per_block,),
                (
                    *kernel_args,
                    out,
                    groupby_part,
                    mask_part,
                    X_part.shape[0],
                    X_part.shape[1],
                    n_groups,
                ),
            )
            return out

        # Prepare Dask arrays
        mask = self._get_mask()
        mask_dask = da.from_array(
            mask, chunks=(self.data.chunks[0]), meta=_meta_dense(mask.dtype)
        )
        groupby_dask = da.from_array(
            self.groupby,
            chunks=(self.data.chunks[0]),
            meta=_meta_dense(self.groupby.dtype),
        )

        # Apply aggregation across all blocks
        out = da.map_blocks(
            __aggregate_dask,
            self.data,
            mask_dask[..., None],
            groupby_dask[..., None],
            meta=cp.empty([], dtype=cp.float64),
            dtype=cp.float64,
            new_axis=(1, 2),
            chunks=(
                (1,) * self.data.blocks.size,
                (3,),
                (n_groups,),
                (self.data.shape[1],),
            ),
        )

        # Compute final aggregated results
        out = out.sum(axis=0, split_every=split_every).compute()
        sums, counts, sq_sums = out[0], out[1], out[2]

        # Calculate statistics
        counts = counts.astype(cp.int32)
        means = sums / self.n_cells
        var = sq_sums / self.n_cells - cp.power(means, 2)
        var *= self.n_cells / (self.n_cells - dof)

        return {"mean": means, "var": var, "sum": sums, "count_nonzero": counts}

    def count_mean_var_sparse(self, dof: int = 1):
        """
        This function is used to calculate the sum, mean, and variance of the sparse data matrix.
        It uses a custom cuda-kernel to perform the aggregation.
        """

        assert dof >= 0
        from ._kernels._aggr_kernels import (
            _get_aggr_sparse_kernel,
            _get_aggr_sparse_kernel_csc,
        )

        out = cp.zeros(
            (3, self.n_cells.shape[0] * self.data.shape[1]), dtype=cp.float64
        )

        block = (512,)
        if self.data.format == "csc":
            grid = (self.data.shape[1],)
            aggr_kernel = _get_aggr_sparse_kernel_csc(self.data.dtype)
        else:
            grid = (self.data.shape[0],)
            aggr_kernel = _get_aggr_sparse_kernel(self.data.dtype)
        mask = self._get_mask()
        aggr_kernel(
            grid,
            block,
            (
                self.data.indptr,
                self.data.indices,
                self.data.data,
                out,
                self.groupby,
                mask,
                self.data.shape[0],
                self.data.shape[1],
                self.n_cells.shape[0],
            ),
        )
        sums, counts, sq_sums = out[0, :], out[1, :], out[2, :]
        sums = sums.reshape(self.n_cells.shape[0], self.data.shape[1])
        sq_sums = sq_sums.reshape(self.n_cells.shape[0], self.data.shape[1])
        counts = counts.reshape(self.n_cells.shape[0], self.data.shape[1])
        counts = counts.astype(cp.int32)
        means = sums / self.n_cells
        var = sq_sums / self.n_cells - means**2
        var *= self.n_cells / (self.n_cells - dof)

        results = {"sum": sums, "count_nonzero": counts, "mean": means, "var": var}
        return results

    def count_mean_var_sparse_sparse(self, funcs, dof: int = 1):
        """
        This function is used to calculate the sum, mean, and variance of the sparse data matrix.
        It uses a custom cuda-kernel to perform the aggregation.
        """

        assert dof >= 0
        from ._kernels._aggr_kernels import (
            _get_aggr_sparse_sparse_kernel,
            _get_sparse_var_kernel,
        )

        if self.data.format == "csc":
            self.data = self.data.tocsr()

        src_row = cp.zeros(self.data.nnz, dtype=cp.int32)
        src_col = cp.zeros(self.data.nnz, dtype=cp.int32)
        src_data = cp.zeros(self.data.nnz, dtype=cp.float64)
        block = (128,)
        grid = (self.data.shape[0],)
        aggr_kernel = _get_aggr_sparse_sparse_kernel(self.data.dtype)
        mask = self._get_mask()
        aggr_kernel(
            grid,
            block,
            (
                self.data.indptr,
                self.data.indices,
                self.data.data,
                src_row,
                src_col,
                src_data,
                self.groupby,
                mask,
                self.data.shape[0],
            ),
        )

        keys = cp.stack([src_col, src_row])
        order = cp.lexsort(keys)

        src_row = src_row[order]
        src_col = src_col[order]
        src_data = src_data[order]

        _sum_duplicates_diff = cp.ElementwiseKernel(
            "raw T row, raw T col",
            "T diff",
            """
            T diff_out = 1;
            if (i == 0 || row[i - 1] == row[i] && col[i - 1] == col[i]) {
            diff_out = 0;
            }
            diff = diff_out;
            """,
            "cupyx_scipy_sparse_coo_sum_duplicates_diff",
        )

        diff = _sum_duplicates_diff(src_row, src_col, size=src_row.size)
        index = cp.cumsum(diff, dtype=cp.int32)
        nnz = index[-1].get()

        # calculate the rows and indices
        rows = cp.zeros(nnz + 1, dtype=cp.int32)
        indices = cp.zeros(nnz + 1, dtype=cp.int32)

        cp.ElementwiseKernel(
            "int32 src_row, int32 src_col, int32 index",
            "raw int32 rows, raw int32 indices",
            """
            rows[index] = src_row;
            indices[index] = src_col;
            """,
            "cupyx_scipy_sparse_coo_sum_duplicates_assign",
        )(src_row, src_col, index, rows, indices)

        # Calculate the indptr
        transitions = cp.where(cp.diff(rows) > 0)[0]
        indptr = cp.zeros(9, dtype=cp.int32)
        indptr[1:-1] = transitions + 1
        indptr[-1] = nnz + 1

        # Calculate the the sparse matrices
        results = {}

        if "sum" in funcs:
            sums = cp.zeros(nnz + 1, dtype=cp.float64)
            cp.ElementwiseKernel(
                "float64 src, int32 index",
                "raw float64 sums",
                """
                atomicAdd(&sums[index], src);
                """,
                "create_sum_sparse_matrix",
            )(src_data, index, sums)

            results["sum"] = cp_sparse.csr_matrix(
                (sums, indices, indptr),
                shape=(self.n_cells.shape[0], self.data.shape[1]),
            )
        if "var" in funcs or "mean" in funcs:
            means = cp.zeros(nnz + 1, dtype=cp.float64)
            var = cp.zeros(nnz + 1, dtype=cp.float64)
            cp.ElementwiseKernel(
                "float64 src, int32 rows, int32 index, raw float64 ncells",
                "raw float64 means, raw float64 var",
                """
                atomicAdd(&means[index], src / ncells[rows]);
                atomicAdd(&var[index], (src * src) / ncells[rows]);
                """,
                "create_mean_var_sparse_matrix",
            )(src_data, src_row, index, self.n_cells, means, var)
            if "mean" in funcs:
                results["mean"] = cp_sparse.csr_matrix(
                    (means, indices, indptr),
                    shape=(self.n_cells.shape[0], self.data.shape[1]),
                )
            if "var" in funcs:
                var = cp_sparse.csr_matrix(
                    (var, indices, indptr),
                    shape=(self.n_cells.shape[0], self.data.shape[1]),
                )

                sparse_var = _get_sparse_var_kernel(var.dtype)
                sparse_var(
                    grid,
                    block,
                    (
                        var.indptr,
                        var.indices,
                        var.data,
                        means,
                        self.n_cells,
                        dof,
                        var.shape[0],
                    ),
                )
                results["var"] = var
        if "count_nonzero" in funcs:
            counts = cp.zeros(nnz + 1, dtype=cp.float32)
            cp.ElementwiseKernel(
                "float64 src,int32 index",
                "raw float32 counts",
                """
                if (src != 0){
                    atomicAdd(&counts[index], 1.0f);
                }
                """,
                "create_count_nonzero_sparse_matrix",
            )(src_data, index, counts)
            results["count_nonzero"] = cp_sparse.csr_matrix(
                (counts, indices, indptr),
                shape=(self.n_cells.shape[0], self.data.shape[1]),
            )

        return results

    def count_mean_var_dense(self, dof: int = 1):
        """
        This function is used to calculate the sum, mean, and variance of the sparse data matrix.
        It uses a custom cuda-kernel to perform the aggregation.
        """

        assert dof >= 0
        from ._kernels._aggr_kernels import (
            _get_aggr_dense_kernel_C,
            _get_aggr_dense_kernel_F,
        )

        out = cp.zeros((3, self.n_cells.shape[0], self.data.shape[1]), dtype=cp.float64)

        N = self.data.shape[0] * self.data.shape[1]
        threads_per_block = 512
        blocks = min(
            (N + threads_per_block - 1) // threads_per_block,
            cp.cuda.Device().attributes["MultiProcessorCount"] * 8,
        )
        if self.data.flags.c_contiguous:
            aggr_kernel = _get_aggr_dense_kernel_C(self.data.dtype)
        else:
            aggr_kernel = _get_aggr_dense_kernel_F(self.data.dtype)
        mask = self._get_mask()
        aggr_kernel(
            (blocks,),
            (threads_per_block,),
            (
                self.data,
                out,
                self.groupby,
                mask,
                self.data.shape[0],
                self.data.shape[1],
                self.n_cells.shape[0],
            ),
        )
        sums, counts, sq_sums = out[0], out[1], out[2]
        sums = sums.reshape(self.n_cells.shape[0], self.data.shape[1])
        counts = counts.reshape(self.n_cells.shape[0], self.data.shape[1])
        sq_sums = sq_sums.reshape(self.n_cells.shape[0], self.data.shape[1])
        counts = counts.astype(cp.int32)
        means = sums / self.n_cells
        var = sq_sums / self.n_cells - cp.power(means, 2)
        var *= self.n_cells / (self.n_cells - dof)

        results = {"sum": sums, "count_nonzero": counts, "mean": means, "var": var}

        return results


def aggregate(
    adata: AnnData,
    by: str | Collection[str],
    func: AggType | Iterable[AggType],
    *,
    axis: Literal["obs", 0, "var", 1] | None = None,
    mask: NDArray[np.bool_] | str | None = None,
    dof: int = 1,
    layer: str | None = None,
    obsm: str | None = None,
    varm: str | None = None,
    return_sparse: bool = False,
    **kwargs,
) -> AnnData:
    """\
    Aggregate data matrix based on some categorical grouping.

    This function is useful for pseudobulking as well as plotting.

    Aggregation to perform is specified by `func`, which can be a single metric or a
    list of metrics. Each metric is computed over the group and results in a new layer
    in the output `AnnData` object.

    If none of `layer`, `obsm`, or `varm` are passed in, `X` will be used for aggregation data.
    If `func` only has length 1 or is just an `AggType`, then aggregation data is written to `X`.
    Otherwise, it is written to `layers` or `xxxm` as appropriate for the dimensions of the aggregation data.

    Parameters
    ----------
    adata
        :class:`~anndata.AnnData` to be aggregated.
    by
        Key of the column to be grouped-by.
    func
        How to aggregate.
    axis
        Axis on which to find group by column.
    mask
        Boolean mask (or key to column containing mask) to apply along the axis.
    dof
        Degrees of freedom for variance. Defaults to 1.
    layer
        If not None, key for aggregation data.
    obsm
        If not None, key for aggregation data.
    varm
        If not None, key for aggregation data.
    return_sparse
        Whether to return a sparse matrix. Only works for sparse input data.

    Returns
    -------
    Aggregated :class:`~anndata.AnnData`.

    Examples
    --------

    Calculating mean expression and number of nonzero entries per cluster:

    >>> import scanpy as sc, pandas as pd
    >>> import rapids_singlecell as rsc
    >>> pbmc = sc.datasets.pbmc3k_processed().raw.to_adata()
    >>> rsc.get.anndata_to_GPU(pbmc)
    >>> pbmc.shape
    (2638, 13714)
    >>> aggregated = rsc.get.aggregate(pbmc, by="louvain", func=["mean", "count_nonzero"])
    >>> aggregated
    AnnData object with n_obs × n_vars = 8 × 13714
        obs: 'louvain'
        var: 'n_cells'
        layers: 'mean', 'count_nonzero'

    We can group over multiple columns:

    >>> pbmc.obs["percent_mito_binned"] = pd.cut(pbmc.obs["percent_mito"], bins=5)
    >>> rsc.get.aggregate(pbmc, by=["louvain", "percent_mito_binned"], func=["mean", "count_nonzero"])
    AnnData object with n_obs × n_vars = 40 × 13714
        obs: 'louvain', 'percent_mito_binned'
        var: 'n_cells'
        layers: 'mean', 'count_nonzero'

    Note that this filters out any combination of groups that wasn't present in the original data.
    """
    if axis is None:
        axis = 1 if varm else 0
    axis, axis_name = _resolve_axis(axis)
    if mask is not None:
        mask = _check_mask(adata, mask, axis_name)
    data = adata.X
    if sum(p is not None for p in [varm, obsm, layer]) > 1:
        raise TypeError("Please only provide one (or none) of varm, obsm, or layer")

    if varm is not None:
        if axis != 1:
            raise ValueError("varm can only be used when axis is 1")
        data = adata.varm[varm]
    elif obsm is not None:
        if axis != 0:
            raise ValueError("obsm can only be used when axis is 0")
        data = adata.obsm[obsm]
    elif layer is not None:
        data = adata.layers[layer]
        if axis == 1:
            data = data.T
    elif axis == 1:
        # i.e., all of `varm`, `obsm`, `layers` are None so we use `X` which must be transposed
        data = data.T
    _check_gpu_X(data, allow_dask=True)
    dim_df = getattr(adata, axis_name)
    categorical, new_label_df = _combine_categories(dim_df, by)
    groupby = Aggregate(groupby=categorical, data=data, mask=mask)

    funcs = set([func] if isinstance(func, str) else func)
    if unknown := funcs - set(get_args(AggType)):
        raise ValueError(f"func {unknown} is not one of {get_args(AggType)}")

    if isinstance(data, cp.ndarray):
        result = groupby.count_mean_var_dense(dof)
    elif isinstance(data, DaskArray):
        if "split_every" in kwargs:
            assert isinstance(kwargs["split_every"], int)
            assert kwargs["split_every"] > 0
            split_every = kwargs["split_every"]
        else:
            split_every = 2
        result = groupby.count_mean_var_dask(dof, split_every=split_every)

    else:
        if return_sparse:
            result = groupby.count_mean_var_sparse_sparse(funcs, dof)
        else:
            result = groupby.count_mean_var_sparse(dof)
    layers = {}

    if "sum" in funcs:
        layers["sum"] = result["sum"]
    if "mean" in funcs:
        layers["mean"] = result["mean"]
    if "count_nonzero" in funcs:
        layers["count_nonzero"] = result["count_nonzero"]
    if "var" in funcs:
        layers["var"] = result["var"]

    result = AnnData(
        layers=layers,
        obs=new_label_df,
        var=getattr(adata, "var" if axis == 0 else "obs"),
    )
    if axis == 1:
        return result.T
    else:
        return result
