from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, get_args

import cupy as cp
from anndata import AnnData
from cupyx.scipy import sparse as cp_sparse
from scanpy._utils import _resolve_axis
from scanpy.get._aggregated import _combine_categories, sparse_indicator

from rapids_singlecell.get import _check_mask
from rapids_singlecell.preprocessing._utils import _check_gpu_X

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

Array = Union[cp.ndarray, cp_sparse.csc_matrix, cp_sparse.csr_matrix]
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
        self.indicator_matrix = cp_sparse.coo_matrix(
            sparse_indicator(groupby, mask=mask)
        )
        self.mask = mask
        self.groupby = cp.array(groupby.codes, dtype=cp.int32)
        self.data = data

    groupby: cp.ndarray
    indicator_matrix: cp_sparse.coo_matrix
    data: Array

    def _get_mask(self):
        if self.mask is not None:
            return cp.array(self.mask)
        else:
            return cp.ones(self.data.shape[0], dtype=bool)

    def count_mean_var_sparse(self, dof: int = 1):
        """
        This function is used to calculate the sum, mean, and variance of the sparse data matrix.
        It uses a custom cuda-kernel to perform the aggregation.
        """

        assert dof >= 0
        from ._kernels._aggr_kernels import _get_aggr_sparse_kernel

        if self.data.format == "csc":
            self.data = self.data.tocsr()
        means = cp.zeros(
            (self.indicator_matrix.shape[0], self.data.shape[1]), dtype=cp.float64
        )
        var = cp.zeros(
            (self.indicator_matrix.shape[0], self.data.shape[1]), dtype=cp.float64
        )
        sums = cp.zeros(
            (self.indicator_matrix.shape[0], self.data.shape[1]), dtype=cp.float64
        )
        counts = cp.zeros(
            (self.indicator_matrix.shape[0], self.data.shape[1]), dtype=cp.int32
        )
        block = (128,)
        grid = (self.data.shape[0],)
        aggr_kernel = _get_aggr_sparse_kernel(self.data.dtype)
        mask = self._get_mask()
        n_cells = self.indicator_matrix.sum(axis=1).astype(cp.float64)
        aggr_kernel(
            grid,
            block,
            (
                self.data.indptr,
                self.data.indices,
                self.data.data,
                counts,
                sums,
                means,
                var,
                self.groupby,
                n_cells,
                mask,
                self.data.shape[0],
                self.data.shape[1],
            ),
        )

        var = var - cp.power(means, 2)
        var *= n_cells / (n_cells - dof)
        return sums, counts, means, var

    def count_mean_var_dense(self, dof: int = 1):
        """
        This function is used to calculate the sum, mean, and variance of the sparse data matrix.
        It uses a custom cuda-kernel to perform the aggregation.
        """

        assert dof >= 0
        from ._kernels._aggr_kernels import _get_aggr_dense_kernel

        means = cp.zeros(
            (self.indicator_matrix.shape[0], self.data.shape[1]), dtype=cp.float64
        )
        var = cp.zeros(
            (self.indicator_matrix.shape[0], self.data.shape[1]), dtype=cp.float64
        )
        sums = cp.zeros(
            (self.indicator_matrix.shape[0], self.data.shape[1]), dtype=cp.float64
        )
        counts = cp.zeros(
            (self.indicator_matrix.shape[0], self.data.shape[1]), dtype=cp.int32
        )
        block = (128,)
        grid = (self.data.shape[0],)
        aggr_kernel = _get_aggr_dense_kernel(self.data.dtype)
        mask = self._get_mask()
        n_cells = self.indicator_matrix.sum(axis=1).astype(cp.float64)
        aggr_kernel(
            grid,
            block,
            (
                self.data.data,
                counts,
                sums,
                means,
                var,
                self.groupby,
                n_cells,
                mask,
                self.data.shape[0],
                self.data.shape[1],
            ),
        )

        var = var - cp.power(means, 2)
        var *= n_cells / (n_cells - dof)
        return sums, counts, means, var


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

    Params
    ------
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
    _check_gpu_X(data)
    dim_df = getattr(adata, axis_name)
    categorical, new_label_df = _combine_categories(dim_df, by)
    # Actual computation

    groupby = Aggregate(groupby=categorical, data=data, mask=mask)

    if isinstance(data, cp.ndarray):
        sums_, counts_, means_, vars_ = groupby.count_mean_var_dense(dof)
    else:
        sums_, counts_, means_, vars_ = groupby.count_mean_var_sparse(dof)
    layers = {}

    funcs = set([func] if isinstance(func, str) else func)
    if unknown := funcs - set(get_args(AggType)):
        raise ValueError(f"func {unknown} is not one of {get_args(AggType)}")

    if "sum" in funcs:
        layers["sum"] = sums_
    if "mean" in funcs:
        layers["mean"] = means_
    if "count_nonzero" in funcs:
        layers["count_nonzero"] = counts_
    if "var" in funcs:
        layers["var"] = vars_

    result = AnnData(
        layers=layers,
        obs=new_label_df,
        var=getattr(adata, "var" if axis == 0 else "obs"),
    )
    if axis == 1:
        return result.T
    else:
        return result
