from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Union

import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy.sparse import csr_matrix
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell._compat import (
    DaskArray,
    _meta_dense,
    _meta_sparse,
)

from ._utils import _check_gpu_X, _check_nonnegative_integers

if TYPE_CHECKING:
    from anndata import AnnData
    from cupyx.scipy.sparse import spmatrix

    from rapids_singlecell._utils import ArrayTypesDask


def normalize_total(
    adata: AnnData,
    *,
    target_sum: int | None = None,
    layer: int | str = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, csr_matrix, cp.ndarray, None]:  # noqa: UP007
    """\
    Normalizes rows in matrix so they sum to `target_sum`.

    Parameters
    ----------
        adata
            AnnData object

        target_sum
            If `None`, after normalization, each observation (cell) has a total count
            equal to the median of total counts for observations (cells) before normalization.

        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.

        inplace
            Whether to update `adata` or return the matrix.

        copy
            Whether to return a copy or update `adata`. Not compatible with inplace=False.
    Returns
    -------
        Returns a normalized copy or  updates `adata` with a normalized version of
        the original `adata.X` and `adata.layers['layer']`, depending on `inplace`.
    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()
    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X, allow_dask=True)

    if not inplace:
        X = X.copy()

    if sparse.isspmatrix_csc(X):
        X = X.tocsr()
    if not target_sum:
        target_sum = _get_target_sum(X)

    X = _normalize_total(X, target_sum)

    if inplace:
        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata
    elif not inplace:
        return X


def _normalize_total(X: ArrayTypesDask, target_sum: int):
    if isinstance(X, sparse.csr_matrix):
        return _normalize_total_csr(X, target_sum)
    elif isinstance(X, DaskArray):
        return _normalize_total_dask(X, target_sum)
    elif isinstance(X, cp.ndarray):
        from rapids_singlecell._cuda import _norm_cuda as _nc

        if not X.flags.c_contiguous:
            X = cp.asarray(X, order="C")
        _nc.mul_dense(
            X.data.ptr,
            nrows=X.shape[0],
            ncols=X.shape[1],
            target_sum=float(target_sum),
            itemsize=cp.dtype(X.dtype).itemsize,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return X
    else:
        raise ValueError(f"Cannot normalize {type(X)}")


def _normalize_total_csr(X: sparse.csr_matrix, target_sum: int) -> sparse.csr_matrix:
    from rapids_singlecell._cuda import _norm_cuda as _nc

    _nc.mul_csr(
        X.indptr.data.ptr,
        X.data.data.ptr,
        nrows=X.shape[0],
        target_sum=float(target_sum),
        itemsize=cp.dtype(X.dtype).itemsize,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return X


def _normalize_total_dask(X: DaskArray, target_sum: int) -> DaskArray:
    if isinstance(X._meta, sparse.csr_matrix):
        from rapids_singlecell._cuda import _norm_cuda as _nc

        def __mul(X_part):
            _nc.mul_csr(
                X_part.indptr.data.ptr,
                X_part.data.data.ptr,
                nrows=X_part.shape[0],
                target_sum=float(target_sum),
                itemsize=cp.dtype(X_part.data.dtype).itemsize,
                stream=cp.cuda.get_current_stream().ptr,
            )
            return X_part

        X = X.map_blocks(__mul, meta=_meta_sparse(X.dtype))
    elif isinstance(X._meta, cp.ndarray):
        from rapids_singlecell._cuda import _norm_cuda as _nc

        def __mul(X_part):
            _nc.mul_dense(
                X_part.data.ptr,
                nrows=X_part.shape[0],
                ncols=X_part.shape[1],
                target_sum=float(target_sum),
                itemsize=cp.dtype(X_part.dtype).itemsize,
                stream=cp.cuda.get_current_stream().ptr,
            )
            return X_part

        X = X.map_blocks(__mul, meta=_meta_dense(X.dtype))
    else:
        raise ValueError(f"Cannot normalize {type(X)}")
    return X


def _get_target_sum(X: ArrayTypesDask) -> int:
    if isinstance(X, sparse.csr_matrix):
        return _get_target_sum_csr(X)
    elif isinstance(X, DaskArray):
        return _get_target_sum_dask(X)
    else:
        return cp.median(X.sum(axis=1))


def _get_target_sum_csr(X: sparse.csr_matrix) -> int:
    from rapids_singlecell._cuda import _norm_cuda as _nc

    counts_per_cell = cp.zeros(X.shape[0], dtype=X.dtype)
    _nc.sum_major(
        X.indptr.data.ptr,
        X.data.data.ptr,
        sums=counts_per_cell.data.ptr,
        major=X.shape[0],
        itemsize=cp.dtype(X.dtype).itemsize,
        stream=cp.cuda.get_current_stream().ptr,
    )
    counts_per_cell = counts_per_cell[counts_per_cell > 0]
    target_sum = cp.median(counts_per_cell)
    return target_sum


def _get_target_sum_dask(X: DaskArray) -> int:
    if isinstance(X._meta, sparse.csr_matrix):
        from rapids_singlecell._cuda import _norm_cuda as _nc

        def __sum(X_part):
            counts_per_cell = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            _nc.sum_major(
                X_part.indptr.data.ptr,
                X_part.data.data.ptr,
                sums=counts_per_cell.data.ptr,
                major=X_part.shape[0],
                itemsize=cp.dtype(X_part.dtype).itemsize,
                stream=cp.cuda.get_current_stream().ptr,
            )
            return counts_per_cell

    elif isinstance(X._meta, cp.ndarray):

        def __sum(X_part):
            return X_part.sum(axis=1)
    else:
        raise ValueError(f"Cannot compute target sum for {type(X)}")
    target_sum_chunk_matrices = X.map_blocks(
        __sum,
        meta=cp.array((1.0,), dtype=X.dtype),
        dtype=X.dtype,
        chunks=(X.chunksize[0],),
        drop_axis=1,
    )
    counts_per_cell = target_sum_chunk_matrices.compute()
    counts_per_cell = counts_per_cell[counts_per_cell > 0]
    target_sum = cp.median(counts_per_cell)
    return target_sum


def _calc_log1p(X: ArrayTypesDask) -> ArrayTypesDask:
    if isinstance(X, DaskArray):
        meta = _meta_sparse if isinstance(X._meta, csr_matrix) else _meta_dense
        X = X.map_blocks(_calc_log1p, meta=meta(X.dtype))
    else:
        X = X.copy()
        if sparse.issparse(X):
            X = X.log1p()
        else:
            X = cp.log1p(X)
    return X


def log1p(
    adata: AnnData,
    *,
    layer: str | None = None,
    obsm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, spmatrix, cp.ndarray, None]:  # noqa: UP007
    """\
    Calculated the natural logarithm of one plus the sparse matrix.


    Parameters
    ----------
        adata
            AnnData object
        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.
        obsm
            Entry of `.obsm` to transform.
        inplace
            Whether to update `adata` or return the matrix.
        copy
            Whether to return a copy or update `adata`. Not compatible with `inplace=False`.

    Returns
    -------
    The resulting sparse matrix after applying the natural logarithm of one plus the input matrix. \
    If `copy` is set to True, returns the new sparse matrix. Otherwise, updates the `adata` object \
    in-place and returns None.

    """
    if copy:
        if not inplace:
            raise ValueError("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()
    X = _get_obs_rep(adata, layer=layer, obsm=obsm)

    _check_gpu_X(X, allow_dask=True)

    if not inplace:
        X = X.copy()
    """
        if isinstance(X, cp.ndarray):
        X = cp.log1p(X)
    elif sparse.issparse(X):
        X = X.log1p()
    elif isinstance(X, DaskArray):
        if isinstance(X._meta, cp.ndarray):
            X = X.map_blocks(lambda x: cp.log1p(x), meta=_meta_dense(X.dtype))
        elif isinstance(X._meta, sparse.csr_matrix):
            X = X.map_blocks(lambda x: x.log1p(), meta=_meta_sparse(X.dtype))
    """
    X = _calc_log1p(X)
    adata.uns["log1p"] = {"base": None}
    if inplace:
        _set_obs_rep(adata, X, layer=layer, obsm=obsm)

    if copy:
        return adata
    elif not inplace:
        return X


def normalize_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: float | None = None,
    check_values: bool = True,
    layer: str | None = None,
    inplace: bool = True,
) -> Union[cp.ndarray, None]:  # noqa: UP007
    """\
    Applies analytic Pearson residual normalization, based on Lause21.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to `sqrt(n_obs)`
    and overdispersion `theta=100` is used.

    Parameters
    ----------
        adata
            AnnData object
        theta
            The negative binomial overdispersion parameter theta for Pearson residuals.
            Higher values correspond to less overdispersion `(var = mean + mean^2/theta)`, and `theta=np.Inf` corresponds to a Poisson model.
        clip
            Determines if and how residuals are clipped:
            If None, residuals are clipped to the interval [-sqrt(n_obs), sqrt(n_obs)], where n_obs is the number of cells in the dataset (default behavior).
            If any scalar c, residuals are clipped to the interval `[-c, c]`. Set `clip=np.Inf` for no clipping.
        check_values
            If True, checks if counts in selected layer are integers as expected by this function,
            and return a warning if non-integers are found. Otherwise, proceed without checking. Setting this to False can speed up code for large datasets.
        layer
            Layer to use as input instead of :attr:`~anndata.AnnData.X`. If None, :attr:`~anndata.AnnData.X` is used.
        inplace
            If True, update AnnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    -------
        If `inplace=True`, :attr:`~anndata.AnnData.X` or the selected layer in :attr:`~anndata.AnnData.layers` is updated with the normalized values. \
        If `inplace=False` the normalized matrix is returned.

    """
    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X, require_cf=True)

    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    computed_on = layer if layer else "adata.X"
    settings_dict = {"theta": theta, "clip": clip, "computed_on": computed_on}
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    if clip is None:
        n = X.shape[0]
        clip = math.sqrt(n)
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")
    inv_theta = 1.0 / theta
    # sums_cells = cp.zeros(X.shape[0], dtype=X.dtype)
    # sums_genes = cp.zeros(X.shape[1], dtype=X.dtype)

    from rapids_singlecell.preprocessing._qc import _basic_qc

    sums_cells, sums_genes, _, _ = _basic_qc(X)
    inv_sum_total = float(1 / sums_genes.sum())
    residuals = cp.zeros(X.shape, dtype=X.dtype)

    if sparse.issparse(X):
        from rapids_singlecell._cuda import _pr_cuda as _pr

        if sparse.isspmatrix_csc(X):
            _pr.sparse_norm_res_csc(
                X.indptr.data.ptr,
                X.indices.data.ptr,
                X.data.data.ptr,
                sums_cells=sums_cells.data.ptr,
                sums_genes=sums_genes.data.ptr,
                residuals=residuals.data.ptr,
                inv_sum_total=float(inv_sum_total),
                clip=float(clip),
                inv_theta=float(inv_theta),
                n_cells=X.shape[0],
                n_genes=X.shape[1],
                itemsize=cp.dtype(X.dtype).itemsize,
                stream=cp.cuda.get_current_stream().ptr,
            )
        elif sparse.isspmatrix_csr(X):
            _pr.sparse_norm_res_csr(
                X.indptr.data.ptr,
                X.indices.data.ptr,
                X.data.data.ptr,
                sums_cells=sums_cells.data.ptr,
                sums_genes=sums_genes.data.ptr,
                residuals=residuals.data.ptr,
                inv_sum_total=float(inv_sum_total),
                clip=float(clip),
                inv_theta=float(inv_theta),
                n_cells=X.shape[0],
                n_genes=X.shape[1],
                itemsize=cp.dtype(X.dtype).itemsize,
                stream=cp.cuda.get_current_stream().ptr,
            )
        else:
            raise ValueError(
                "Please transform you sparse matrix into CSR or CSC format."
            )
    else:
        from rapids_singlecell._cuda import _pr_cuda as _pr

        _pr.dense_norm_res(
            X.data.ptr,
            residuals=residuals.data.ptr,
            sums_cells=sums_cells.data.ptr,
            sums_genes=sums_genes.data.ptr,
            inv_sum_total=float(inv_sum_total),
            clip=float(clip),
            inv_theta=float(inv_theta),
            n_cells=residuals.shape[0],
            n_genes=residuals.shape[1],
            itemsize=cp.dtype(X.dtype).itemsize,
            stream=cp.cuda.get_current_stream().ptr,
        )

    if inplace is True:
        adata.uns["pearson_residuals_normalization"] = settings_dict
        _set_obs_rep(adata, residuals, layer=layer)
    else:
        return residuals
