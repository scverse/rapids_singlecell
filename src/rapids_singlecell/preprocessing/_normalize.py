from __future__ import annotations

import math
import warnings
from functools import partial
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
    target_sum: float | None = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    layer: str | None = None,
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

        exclude_highly_expressed
            Exclude (very) highly expressed genes for the computation of the
            normalization factor (size factor) for each cell. A gene is considered
            highly expressed, if it has more than `max_fraction` of the total counts
            in at least one cell. The not-excluded genes will sum up to
            `target_sum`.

        max_fraction
            If `exclude_highly_expressed=True`, consider cells as highly expressed
            that have more counts than `max_fraction` of the original total counts
            in at least one cell.

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

    if exclude_highly_expressed:
        if isinstance(X, DaskArray):
            raise NotImplementedError(
                "`exclude_highly_expressed` is not supported for Dask arrays."
            )
        if not 0 < max_fraction < 1:
            raise ValueError(
                f"`max_fraction` must be between 0 and 1, got {max_fraction}."
            )
        X = _normalize_total_exclude(X, target_sum, max_fraction)
    else:
        if target_sum is None:
            target_sum = _get_target_sum(X)
        X = _normalize_total(X, target_sum)

    if inplace:
        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata
    elif not inplace:
        return X


def _normalize_total(X: ArrayTypesDask, target_sum: float):
    if isinstance(X, sparse.csr_matrix):
        return _normalize_total_csr(X, target_sum)
    elif isinstance(X, DaskArray):
        return _normalize_total_dask(X, target_sum)
    elif isinstance(X, cp.ndarray):
        from ._kernels._norm_kernel import _mul_dense

        if not X.flags.c_contiguous:
            X = cp.asarray(X, order="C")
        mul_kernel = _mul_dense(X.dtype)
        mul_kernel(
            (math.ceil(X.shape[0] / 128),),
            (128,),
            (X, X.shape[0], X.shape[1], int(target_sum)),
        )
        return X
    else:
        raise ValueError(f"Cannot normalize {type(X)}")


def _normalize_total_exclude(
    X: csr_matrix | cp.ndarray,
    target_sum: float | None,
    max_fraction: float,
) -> csr_matrix | cp.ndarray:
    if isinstance(X, sparse.csr_matrix):
        return _normalize_total_exclude_csr(X, target_sum, max_fraction)
    elif isinstance(X, cp.ndarray):
        return _normalize_total_exclude_dense(X, target_sum, max_fraction)
    else:
        raise ValueError(f"Cannot normalize {type(X)}")


def _normalize_total_exclude_csr(
    X: csr_matrix,
    target_sum: float | None,
    max_fraction: float,
) -> csr_matrix:
    from ._kernels._norm_kernel import _find_hi_genes_csr, _get_sparse_sum_major

    n_cells, n_genes = X.shape

    # Step 1: Compute initial row sums
    counts_per_cell = cp.zeros(n_cells, dtype=X.dtype)
    sum_kernel = _get_sparse_sum_major(X.dtype)
    sum_kernel(
        (n_cells,),
        (64,),
        (X.indptr, X.data, counts_per_cell, n_cells),
    )

    # Step 2: Find highly expressed genes (no nnz-sized allocations)
    thresholds = (max_fraction * counts_per_cell).astype(X.dtype)
    gene_is_hi = cp.zeros(n_genes, dtype=cp.bool_)
    hi_kernel = _find_hi_genes_csr(X.dtype)
    hi_kernel(
        (math.ceil(n_cells / 128),),
        (128,),
        (X.indptr, X.indices, X.data, thresholds, gene_is_hi, n_cells),
    )

    if target_sum is not None:
        # Fused: masked row sum + scale in a single pass
        from ._kernels._norm_kernel import _masked_mul_csr

        fused_kernel = _masked_mul_csr(X.dtype)
        fused_kernel(
            (math.ceil(n_cells / 128),),
            (128,),
            (
                X.indptr,
                X.indices,
                X.data,
                gene_is_hi,
                n_cells,
                X.dtype.type(target_sum),
            ),
        )
    else:
        # Two-pass: need adjusted counts for median first
        from ._kernels._norm_kernel import _masked_sum_major, _prescaled_mul_csr

        adjusted_counts = cp.zeros(n_cells, dtype=X.dtype)
        masked_kernel = _masked_sum_major(X.dtype)
        masked_kernel(
            (n_cells,),
            (64,),
            (X.indptr, X.indices, X.data, gene_is_hi, adjusted_counts, n_cells),
        )

        nonzero_counts = adjusted_counts[adjusted_counts > 0]
        target_sum = cp.median(nonzero_counts)

        scales = cp.zeros(n_cells, dtype=X.dtype)
        nonzero = adjusted_counts > 0
        scales[nonzero] = cp.array(target_sum, dtype=X.dtype) / adjusted_counts[nonzero]

        prescaled_kernel = _prescaled_mul_csr(X.dtype)
        prescaled_kernel(
            (math.ceil(n_cells / 128),),
            (128,),
            (X.indptr, X.data, scales, n_cells),
        )
    return X


def _normalize_total_exclude_dense(
    X: cp.ndarray,
    target_sum: float | None,
    max_fraction: float,
) -> cp.ndarray:
    from ._kernels._norm_kernel import _prescaled_mul_dense

    if not X.flags.c_contiguous:
        X = cp.asarray(X, order="C")

    n_cells, n_genes = X.shape

    # Step 1: Compute initial row sums
    counts_per_cell = X.sum(axis=1)

    # Step 2: Find highly expressed genes
    hi_exp = X > max_fraction * counts_per_cell.reshape(-1, 1)
    gene_subset = ~hi_exp.any(axis=0)

    # Step 3: Recompute row sums excluding highly expressed genes
    adjusted_counts = X[:, gene_subset].sum(axis=1)

    # Step 4: Compute target_sum from median of adjusted counts if not provided
    if target_sum is None:
        nonzero_counts = adjusted_counts[adjusted_counts > 0]
        target_sum = cp.median(nonzero_counts)

    # Step 5: Compute per-cell scale factors and apply
    scales = cp.zeros(n_cells, dtype=X.dtype)
    mask = adjusted_counts > 0
    scales[mask] = cp.array(target_sum, dtype=X.dtype) / adjusted_counts[mask]

    mul_kernel = _prescaled_mul_dense(X.dtype)
    mul_kernel(
        (math.ceil(n_cells / 128),),
        (128,),
        (X, scales, n_cells, n_genes),
    )
    return X


def _normalize_total_csr(X: sparse.csr_matrix, target_sum: int) -> sparse.csr_matrix:
    from ._kernels._norm_kernel import _mul_csr

    mul_kernel = _mul_csr(X.dtype)
    mul_kernel(
        (math.ceil(X.shape[0] / 128),),
        (128,),
        (X.indptr, X.data, X.shape[0], int(target_sum)),
    )
    return X


def _normalize_total_dask(X: DaskArray, target_sum: int) -> DaskArray:
    if isinstance(X._meta, sparse.csr_matrix):
        from ._kernels._norm_kernel import _mul_csr

        mul_kernel = _mul_csr(X.dtype)
        mul_kernel.compile()

        def __mul(X_part):
            mul_kernel(
                (math.ceil(X_part.shape[0] / 32),),
                (32,),
                (X_part.indptr, X_part.data, X_part.shape[0], int(target_sum)),
            )
            return X_part

        X = X.map_blocks(__mul, meta=_meta_sparse(X.dtype))
    elif isinstance(X._meta, cp.ndarray):
        from ._kernels._norm_kernel import _mul_dense

        mul_kernel = _mul_dense(X.dtype)
        mul_kernel.compile()

        def __mul(X_part):
            mul_kernel(
                (math.ceil(X_part.shape[0] / 128),),
                (128,),
                (X_part, X_part.shape[0], X_part.shape[1], int(target_sum)),
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
    from ._kernels._norm_kernel import _get_sparse_sum_major

    counts_per_cell = cp.zeros(X.shape[0], dtype=X.dtype)
    sum_kernel = _get_sparse_sum_major(X.dtype)
    sum_kernel(
        (X.shape[0],),
        (64,),
        (X.indptr, X.data, counts_per_cell, X.shape[0]),
    )
    counts_per_cell = counts_per_cell[counts_per_cell > 0]
    target_sum = cp.median(counts_per_cell)
    return target_sum


def _get_target_sum_dask(X: DaskArray) -> int:
    if isinstance(X._meta, sparse.csr_matrix):
        from ._kernels._norm_kernel import _get_sparse_sum_major

        sum_kernel = _get_sparse_sum_major(X.dtype)
        sum_kernel.compile()

        def __sum(X_part):
            counts_per_cell = cp.zeros(X_part.shape[0], dtype=X_part.dtype)
            sum_kernel(
                (X.shape[0],),
                (64,),
                (X_part.indptr, X_part.data, counts_per_cell, X_part.shape[0]),
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


def _calc_log1p(X: ArrayTypesDask, base: float | None = None) -> ArrayTypesDask:
    if isinstance(X, DaskArray):
        meta = _meta_sparse if isinstance(X._meta, csr_matrix) else _meta_dense
        X = X.map_blocks(partial(_calc_log1p, base=base), meta=meta(X.dtype))
    else:
        X = X.copy()
        if sparse.issparse(X):
            X = X.log1p()
            if base is not None:
                X.data /= cp.log(base)
        else:
            X = cp.log1p(X)
            if base is not None:
                X /= cp.log(base)
    return X


def log1p(
    adata: AnnData,
    *,
    base: float | None = None,
    layer: str | None = None,
    obsm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, spmatrix, cp.ndarray, None]:  # noqa: UP007
    """\
    Logarithmize the data matrix.

    Computes :math:`X = \\log(X + 1)`, where :math:`log` denotes the natural logarithm
    unless a different `base` is given.

    Parameters
    ----------
        adata
            AnnData object
        base
            Base of the logarithm. Natural logarithm is used by default.
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
    The resulting matrix after applying the logarithm of one plus the input matrix. \
    If `copy` is set to True, returns the modified AnnData. Otherwise, updates the `adata` object \
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

    X = _calc_log1p(X, base=base)
    adata.uns["log1p"] = {"base": base}
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
    Applies analytic Pearson residual normalization :cite:p:`Lause2021`.
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
        clip = cp.sqrt(n, dtype=X.dtype)
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")
    theta = cp.array([1 / theta], dtype=X.dtype)
    clip = cp.array([clip], dtype=X.dtype)
    sums_cells = cp.zeros(X.shape[0], dtype=X.dtype)
    sums_genes = cp.zeros(X.shape[1], dtype=X.dtype)

    if sparse.issparse(X):
        residuals = cp.zeros(X.shape, dtype=X.dtype)
        if sparse.isspmatrix_csc(X):
            from ._kernels._pr_kernels import _sparse_norm_res_csc, _sparse_sum_csc

            block = (8,)
            grid = (int(math.ceil(X.shape[1] / block[0])),)
            sum_csc = _sparse_sum_csc(X.dtype)
            sum_csc(
                grid,
                block,
                (X.indptr, X.indices, X.data, sums_genes, sums_cells, X.shape[1]),
            )
            sum_total = 1 / sums_genes.sum().squeeze()
            norm_res = _sparse_norm_res_csc(X.dtype)
            norm_res(
                grid,
                block,
                (
                    X.indptr,
                    X.indices,
                    X.data,
                    sums_cells,
                    sums_genes,
                    residuals,
                    sum_total,
                    clip,
                    theta,
                    X.shape[0],
                    X.shape[1],
                ),
            )
        elif sparse.isspmatrix_csr(X):
            from ._kernels._pr_kernels import _sparse_norm_res_csr, _sparse_sum_csr

            block = (8,)
            grid = (int(math.ceil(X.shape[0] / block[0])),)
            sum_csr = _sparse_sum_csr(X.dtype)
            sum_csr(
                grid,
                block,
                (X.indptr, X.indices, X.data, sums_genes, sums_cells, X.shape[0]),
            )
            sum_total = 1 / sums_genes.sum().squeeze()
            norm_res = _sparse_norm_res_csr(X.dtype)
            norm_res(
                grid,
                block,
                (
                    X.indptr,
                    X.indices,
                    X.data,
                    sums_cells,
                    sums_genes,
                    residuals,
                    sum_total,
                    clip,
                    theta,
                    X.shape[0],
                    X.shape[1],
                ),
            )
        else:
            raise ValueError(
                "Please transform you sparse matrix into CSR or CSC format."
            )
    else:
        from ._kernels._pr_kernels import _norm_res_dense, _sum_dense

        residuals = cp.zeros(X.shape, dtype=X.dtype)
        block = (8, 8)
        grid = (
            math.ceil(residuals.shape[0] / block[0]),
            math.ceil(residuals.shape[1] / block[1]),
        )
        sum_dense = _sum_dense(X.dtype)
        sum_dense(
            grid,
            block,
            (X, sums_cells, sums_genes, residuals.shape[0], residuals.shape[1]),
        )
        sum_total = 1 / sums_genes.sum().squeeze()
        norm_res = _norm_res_dense(X.dtype)
        norm_res(
            grid,
            block,
            (
                X,
                residuals,
                sums_cells,
                sums_genes,
                sum_total,
                clip,
                theta,
                residuals.shape[0],
                residuals.shape[1],
            ),
        )

    if inplace is True:
        adata.uns["pearson_residuals_normalization"] = settings_dict
        _set_obs_rep(adata, residuals, layer=layer)
    else:
        return residuals
