from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_ovo_cuda as _wc

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes

MIN_GROUP_SIZE_WARNING = 25
STREAMING_SUB_BATCH = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_chunk_size(chunk_size: int | None, n_total_genes: int) -> int:
    if chunk_size is None:
        return max(1, n_total_genes)
    chunk_width = int(chunk_size)
    if chunk_width <= 0:
        msg = "`chunk_size` must be a positive integer."
        raise ValueError(msg)
    return min(chunk_width, max(1, n_total_genes))


def _fill_basic_stats_from_accumulators(
    rg: _RankGenes,
    group_sums: cp.ndarray,
    group_sq_sums: cp.ndarray,
    group_nnz: cp.ndarray,
    group_sizes: np.ndarray,
    *,
    n_cells: int,
    compute_vars: bool = True,
    total_sums: cp.ndarray | None = None,
    total_sq_sums: cp.ndarray | None = None,
    total_nnz: cp.ndarray | None = None,
) -> None:
    """Populate rg.means/vars/pts (+ *_rest) from streamed accumulators.

    Mirrors the Aggregate-based path in :meth:`_RankGenes._basic_stats`
    but consumes per-group sums/sum-of-squares/nnz that the host-streaming
    kernels write directly into caller-provided CuPy buffers.  The math
    runs on GPU and only the derived (means/vars/pts) arrays are
    transferred to host, so the full matrix never round-trips.
    """
    n = cp.asarray(group_sizes, dtype=cp.float64)[:, None]
    means = group_sums / n

    rg.means = cp.asnumpy(means)
    if compute_vars:
        group_ss = group_sq_sums - n * means**2
        vars_ = cp.maximum(group_ss / cp.maximum(n - 1, 1), 0)
        rg.vars = cp.asnumpy(vars_)
    else:
        rg.vars = np.zeros_like(rg.means)
    rg.pts = cp.asnumpy(group_nnz / n) if rg.comp_pts else None

    if rg.ireference is None:
        n_rest = cp.float64(n_cells) - n
        if total_sums is None:
            total_sums = group_sums.sum(axis=0, keepdims=True)
        rest_sums = total_sums - group_sums
        rest_means = rest_sums / n_rest
        rg.means_rest = cp.asnumpy(rest_means)
        if compute_vars:
            if total_sq_sums is None:
                total_sq_sums = group_sq_sums.sum(axis=0, keepdims=True)
            rest_ss = (total_sq_sums - group_sq_sums) - n_rest * rest_means**2
            rg.vars_rest = cp.asnumpy(
                cp.maximum(rest_ss / cp.maximum(n_rest - 1, 1), 0)
            )
        else:
            rg.vars_rest = np.zeros_like(rg.means_rest)
        if rg.comp_pts:
            if total_nnz is None:
                total_nnz = group_nnz.sum(axis=0, keepdims=True)
            rg.pts_rest = cp.asnumpy((total_nnz - group_nnz) / n_rest)
        else:
            rg.pts_rest = None
    else:
        rg.means_rest = None
        rg.vars_rest = None
        rg.pts_rest = None

    rg._compute_stats_in_chunks = False


def _fill_ovo_stats_from_accumulators(
    rg: _RankGenes,
    group_sums_slots: cp.ndarray,
    group_sq_sums_slots: cp.ndarray,
    group_nnz_slots: cp.ndarray,
    *,
    group_sizes: NDArray,
    test_group_indices: list[int],
    n_ref: int,
    compute_vars: bool = True,
) -> None:
    """Populate rg.means/vars/pts from OVO stats slots.

    Slot ordering: 0..n_test-1 are test groups (in ``test_group_indices``
    order); slot n_test is the reference group.  Stats arrays arrive as
    CuPy buffers; the math runs on GPU and only the per-group rows are
    transferred to host as they're assigned onto rg.
    """
    n_test = len(test_group_indices)
    n_genes = int(group_sums_slots.shape[1])
    n_groups = len(rg.groups_order)

    rg.means = np.zeros((n_groups, n_genes), dtype=np.float64)
    rg.vars = np.zeros((n_groups, n_genes), dtype=np.float64)
    rg.pts = np.zeros((n_groups, n_genes), dtype=np.float64) if rg.comp_pts else None

    def _fill(slot: int, size: int, gi: int) -> None:
        if size <= 0:
            return
        sums = group_sums_slots[slot]
        mean = sums / size
        rg.means[gi] = cp.asnumpy(mean)
        if compute_vars and size > 1:
            sq = group_sq_sums_slots[slot]
            ss = sq - size * mean**2
            rg.vars[gi] = cp.asnumpy(cp.maximum(ss / max(size - 1, 1), 0))
        if rg.comp_pts:
            rg.pts[gi] = cp.asnumpy(group_nnz_slots[slot] / size)

    for i, gi in enumerate(test_group_indices):
        _fill(i, int(group_sizes[gi]), gi)
    _fill(n_test, int(n_ref), rg.ireference)

    rg.means_rest = None
    rg.vars_rest = None
    rg.pts_rest = None
    rg._compute_stats_in_chunks = False


def _to_gpu_native(X, n_rows: int, n_cols: int):
    """Move *X* to GPU, preserving its format (CSR / CSC / dense)."""
    # Already on GPU
    if isinstance(X, cp.ndarray):
        return X
    if cpsp.issparse(X):
        return X

    # Host sparse → GPU sparse, same format.  Wilcoxon kernels are native CSR
    # or CSC only; do not hide a whole-matrix sparse format conversion here.
    # Downcast indices to int32 on host before transfer (column indices
    # always fit in int32; scipy may use int64 when nnz > 2^31).
    if isinstance(X, sp.spmatrix | sp.sparray):
        if X.format == "csc":
            return cpsp.csc_matrix(
                (
                    cp.asarray(X.data),
                    cp.asarray(X.indices.astype(np.int32, copy=False)),
                    cp.asarray(X.indptr),
                ),
                shape=(n_rows, n_cols),
            )
        if X.format != "csr":
            raise TypeError(
                "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
                f"full-matrix conversion from {X.format!r}."
            )
        csr = X
        return cpsp.csr_matrix(
            (
                cp.asarray(csr.data),
                cp.asarray(csr.indices.astype(np.int32, copy=False)),
                cp.asarray(csr.indptr),
            ),
            shape=(n_rows, n_cols),
        )

    # Host dense → GPU dense
    if isinstance(X, np.ndarray):
        return cp.asarray(X)

    raise TypeError(f"Unsupported matrix type: {type(X)}")


def _host_sparse_fn_and_arrays(module, base_name: str, X, *, support_idx64: bool):
    """Select host sparse binding and dtype-normalized arrays."""
    is_f64 = X.data.dtype == np.float64
    is_idx64 = support_idx64 and X.indices.dtype == np.int64
    is_i64 = X.indptr.dtype == np.int64
    suffix = ""
    if is_f64:
        suffix += "_f64"
    if is_idx64:
        suffix += "_idx64"
    if is_i64:
        suffix += "_i64"
    fn = getattr(module, base_name + suffix)
    data_arr = X.data if is_f64 else X.data.astype(np.float32, copy=False)
    indices_arr = X.indices if is_idx64 else X.indices.astype(np.int32, copy=False)
    return fn, data_arr, indices_arr


def _column_totals_for_host_matrix(
    X, *, compute_sq_sums: bool, compute_nnz: bool
) -> tuple[cp.ndarray, cp.ndarray | None, cp.ndarray | None]:
    """Compute all-cell column totals without changing sparse format."""
    n_cols = X.shape[1]

    if isinstance(X, sp.spmatrix | sp.sparray):
        data = np.asarray(X.data)
        values = data.astype(np.float64, copy=False)

        if X.format == "csc":
            indptr = np.asarray(X.indptr)
            counts = np.diff(indptr)
            nonempty = counts > 0
            starts = indptr[:-1][nonempty]

            sums = np.zeros(n_cols, dtype=np.float64)
            if starts.size:
                sums[nonempty] = np.add.reduceat(values, starts)

            sq_sums = None
            if compute_sq_sums:
                sq_sums = np.zeros(n_cols, dtype=np.float64)
                if starts.size:
                    sq_sums[nonempty] = np.add.reduceat(values * values, starts)

            nnz = None
            if compute_nnz:
                nnz = np.zeros(n_cols, dtype=np.float64)
                if starts.size:
                    nnz[nonempty] = np.add.reduceat(
                        (data != 0).astype(np.float64, copy=False), starts
                    )
        elif X.format == "csr":
            indices = np.asarray(X.indices, dtype=np.intp)
            sums = np.bincount(indices, weights=values, minlength=n_cols).astype(
                np.float64, copy=False
            )

            sq_sums = (
                np.bincount(indices, weights=values * values, minlength=n_cols).astype(
                    np.float64, copy=False
                )
                if compute_sq_sums
                else None
            )
            nnz = (
                np.bincount(
                    indices,
                    weights=(data != 0).astype(np.float64, copy=False),
                    minlength=n_cols,
                ).astype(np.float64, copy=False)
                if compute_nnz
                else None
            )
        else:
            raise TypeError(
                "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
                f"full-matrix conversion from {X.format!r}."
            )
    elif isinstance(X, np.ndarray):
        sums = np.asarray(X.sum(axis=0, dtype=np.float64), dtype=np.float64)
        sq_sums = (
            np.asarray(np.square(X, dtype=np.float64).sum(axis=0), dtype=np.float64)
            if compute_sq_sums
            else None
        )
        nnz = (
            np.asarray(np.count_nonzero(X, axis=0), dtype=np.float64)
            if compute_nnz
            else None
        )
    else:
        raise TypeError(f"Unsupported host matrix type: {type(X)}")

    total_sums = cp.asarray(sums.reshape(1, n_cols), dtype=cp.float64)
    total_sq_sums = (
        cp.asarray(sq_sums.reshape(1, n_cols), dtype=cp.float64)
        if sq_sums is not None
        else None
    )
    total_nnz = (
        cp.asarray(nnz.reshape(1, n_cols), dtype=cp.float64)
        if nnz is not None
        else None
    )
    return total_sums, total_sq_sums, total_nnz


def _host_ovr_totals_if_needed(
    X,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    compute_sq_sums: bool,
    compute_nnz: bool,
) -> tuple[cp.ndarray | None, cp.ndarray | None, cp.ndarray | None]:
    if not np.any(group_codes == n_groups):
        return None, None, None
    return _column_totals_for_host_matrix(
        X, compute_sq_sums=compute_sq_sums, compute_nnz=compute_nnz
    )


def _extract_dense_block(
    X,
    row_ids: cp.ndarray | None,
    start: int,
    stop: int,
    *,
    csr_arrays: tuple[cp.ndarray, cp.ndarray, cp.ndarray] | None = None,
) -> cp.ndarray:
    """Extract ``X[row_ids, start:stop]`` as dense F-order on GPU.

    CSR kernel path: outputs same dtype as CSR data (float32 or float64).
    Other paths: preserve input dtype.
    """
    if csr_arrays is not None:
        data, indices, indptr = csr_arrays
        if row_ids is None:
            n_target = int(indptr.shape[0] - 1)
            row_ids = cp.arange(n_target, dtype=cp.int32)
        n_target = row_ids.shape[0]
        n_cols = stop - start
        out = cp.zeros((n_target, n_cols), dtype=data.dtype, order="F")
        if n_target > 0 and n_cols > 0:
            stream = cp.cuda.get_current_stream().ptr
            if data.dtype == cp.float32:
                _wc.csr_extract_dense_f32(
                    data,
                    indices,
                    indptr,
                    row_ids,
                    out,
                    n_target=n_target,
                    col_start=start,
                    col_stop=stop,
                    stream=stream,
                )
            else:
                _wc.csr_extract_dense(
                    data,
                    indices,
                    indptr,
                    row_ids,
                    out,
                    n_target=n_target,
                    col_start=start,
                    col_stop=stop,
                    stream=stream,
                )
        return out

    if isinstance(X, np.ndarray):
        if row_ids is not None:
            return cp.asarray(X[cp.asnumpy(row_ids), start:stop], order="F")
        return cp.asarray(X[:, start:stop], order="F")

    if isinstance(X, cp.ndarray):
        chunk = X[row_ids, start:stop] if row_ids is not None else X[:, start:stop]
        return cp.asfortranarray(chunk)

    if isinstance(X, sp.spmatrix | sp.sparray):
        if row_ids is not None:
            idx = cp.asnumpy(row_ids)
            chunk = X[idx][:, start:stop].toarray()
        else:
            chunk = X[:, start:stop].toarray()
        return cp.asarray(chunk, order="F")

    if cpsp.issparse(X):
        if row_ids is not None:
            chunk = X[row_ids][:, start:stop].toarray()
        else:
            chunk = X[:, start:stop].toarray()
        return cp.asfortranarray(chunk)

    raise TypeError(f"Unsupported matrix type: {type(X)}")


def _segmented_sort_columns(
    data: cp.ndarray,
    offsets_host: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_groups: int,
) -> cp.ndarray:
    """Sort each group segment within each column using CUB radix sort.

    Sorts in float32 for half the bandwidth. Returns float32 F-order.
    """
    n_items = n_rows * n_cols
    n_segments = n_cols * n_groups

    col_bases = np.arange(n_cols, dtype=np.int32) * n_rows
    seg_starts = col_bases[:, None] + offsets_host[None, :n_groups]
    seg_arr = np.empty(n_segments + 1, dtype=np.int32)
    seg_arr[:n_segments] = seg_starts.ravel()
    seg_arr[n_segments] = n_items
    seg_offsets_gpu = cp.asarray(seg_arr)

    temp_bytes = _wc.get_seg_sort_temp_bytes(n_items=n_items, n_segments=n_segments)
    cub_temp = cp.empty(temp_bytes, dtype=cp.uint8)

    # data is F-order; ravel("F") gives a flat C-contiguous view (no copy)
    keys_in = data.astype(cp.float32, copy=False).ravel(order="F")
    keys_out = cp.empty_like(keys_in)

    _wc.segmented_sort(
        keys_in,
        keys_out,
        seg_offsets_gpu,
        cub_temp,
        n_items=n_items,
        n_segments=n_segments,
        stream=cp.cuda.get_current_stream().ptr,
    )

    return cp.ndarray(
        (n_rows, n_cols), dtype=cp.float32, memptr=keys_out.data, order="F"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool = False,
    chunk_size: int | None = None,
) -> list[tuple[int, NDArray, NDArray]]:
    """Compute Wilcoxon rank-sum test statistics."""
    X = rg.X
    n_cells, n_total_genes = X.shape
    group_sizes = rg.group_sizes

    # Stats via Aggregate for both OVR and OVO — decoupled from sort.
    # Aggregate reads original dtype for precision, accumulates in float64.
    rg._basic_stats()

    if rg.ireference is not None:
        return _wilcoxon_with_reference(
            rg,
            X,
            n_total_genes,
            group_sizes,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            chunk_size=chunk_size,
        )
    return _wilcoxon_vs_rest(
        rg,
        X,
        n_cells,
        n_total_genes,
        group_sizes,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
    )


# ---------------------------------------------------------------------------
# One-vs-rest
# ---------------------------------------------------------------------------


def _wilcoxon_vs_rest(
    rg: _RankGenes,
    X,
    n_cells: int,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
) -> list[tuple[int, NDArray, NDArray]]:
    """Wilcoxon test: each group vs rest of cells.

    Dispatches to CSR, CSC, or dense streaming kernel based on input format.
    No unnecessary format conversions.
    """
    from rapids_singlecell._cuda import _wilcoxon_ovr_cuda as _ovr

    n_groups = len(rg.groups_order)

    for name, size in zip(rg.groups_order, group_sizes, strict=True):
        rest = n_cells - size
        if size <= MIN_GROUP_SIZE_WARNING or rest <= MIN_GROUP_SIZE_WARNING:
            warnings.warn(
                f"Group {name} has size {size} (rest {rest}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=4,
            )

    group_codes = rg.group_codes.astype(np.int32, copy=False)
    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev

    # Determine host-streaming eligibility BEFORE transferring
    host_csc = isinstance(X, sp.spmatrix | sp.sparray) and X.format == "csc"
    host_csr = isinstance(X, sp.spmatrix | sp.sparray) and X.format == "csr"
    host_dense = isinstance(X, np.ndarray)

    if host_csc or host_csr or host_dense:
        # Host-streaming: sort+rank stays on host→GPU per sub-batch.  The
        # kernel also emits per-group sum, sum-of-squares, and nonzero
        # counts into caller-provided CuPy buffers, so means/vars/pts can
        # be derived without uploading the full matrix.  Outputs live on
        # the GPU and feed directly into the z-score / p-value math below.
        compute_vars = False
        compute_nnz = rg.comp_pts
        rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
        tie_corr = cp.ones(n_total_genes, dtype=cp.float64)
        group_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
        group_sq_sums = cp.empty(
            (n_groups, n_total_genes) if compute_vars else (1, 1),
            dtype=cp.float64,
        )
        group_nnz = cp.empty(
            (n_groups, n_total_genes) if compute_nnz else (1, 1),
            dtype=cp.float64,
        )

        if host_csc:
            group_sizes_np = group_sizes.astype(np.float64, copy=False)
            # Native host dtype is preserved and uploaded once per sub-batch;
            # a pre-sort kernel casts to float32 for the sort keys while
            # accumulating stats in float64 from the original values.
            _csc_host_fn, data_arr, indices_arr = _host_sparse_fn_and_arrays(
                _ovr, "ovr_sparse_csc_host", X, support_idx64=False
            )
            _csc_host_fn(
                data_arr,
                indices_arr,
                X.indptr,
                group_codes,
                group_sizes_np,
                rank_sums,
                tie_corr,
                group_sums,
                group_sq_sums,
                group_nnz,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                compute_sq_sums=compute_vars,
                compute_nnz=compute_nnz,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif host_csr:
            group_sizes_np = group_sizes.astype(np.float64, copy=False)
            csr = X
            if not csr.has_sorted_indices:
                csr = csr.copy()
                csr.sort_indices()
            _csr_host_fn, data_arr, indices_arr = _host_sparse_fn_and_arrays(
                _ovr, "ovr_sparse_csr_host", csr, support_idx64=True
            )
            _csr_host_fn(
                data_arr,
                indices_arr,
                csr.indptr,
                group_codes,
                group_sizes_np,
                rank_sums,
                tie_corr,
                group_sums,
                group_sq_sums,
                group_nnz,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                compute_sq_sums=compute_vars,
                compute_nnz=compute_nnz,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        else:
            is_f64 = X.dtype == np.float64
            _dense_host_fn = (
                _ovr.ovr_streaming_dense_host_f64
                if is_f64
                else _ovr.ovr_streaming_dense_host
            )
            block = X if is_f64 else X.astype(np.float32, copy=False)
            _dense_host_fn(
                np.asfortranarray(block),
                group_codes,
                rank_sums,
                tie_corr,
                group_sums,
                group_sq_sums,
                group_nnz,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                compute_sq_sums=compute_vars,
                compute_nnz=compute_nnz,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )

        if rg._compute_stats_in_chunks:
            total_sums, total_sq_sums, total_nnz = _host_ovr_totals_if_needed(
                X,
                group_codes,
                n_groups,
                compute_sq_sums=compute_vars,
                compute_nnz=compute_nnz,
            )
            _fill_basic_stats_from_accumulators(
                rg,
                group_sums,
                group_sq_sums,
                group_nnz,
                group_sizes.astype(np.float64, copy=False),
                n_cells=n_cells,
                compute_vars=compute_vars,
                total_sums=total_sums,
                total_sq_sums=total_sq_sums,
                total_nnz=total_nnz,
            )
    else:
        # GPU data → use native GPU kernels.
        X_gpu = _to_gpu_native(X, n_cells, n_total_genes)

        if rg._compute_stats_in_chunks:
            rg.X = X_gpu
            rg._compute_stats_in_chunks = False
            rg._basic_stats()

        group_codes_gpu = cp.asarray(group_codes)
        rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
        tie_corr = cp.ones(n_total_genes, dtype=cp.float64)

        if cpsp.isspmatrix_csc(X_gpu):
            # Sparse-aware path: sort only stored nonzeros,
            # handle zeros analytically.
            csc_data = X_gpu.data.astype(cp.float32, copy=False)
            csc_indices = X_gpu.indices.astype(cp.int32, copy=False)
            csc_indptr = X_gpu.indptr.astype(cp.int32, copy=False)
            cp.cuda.get_current_stream().synchronize()
            _ovr.ovr_sparse_csc(
                csc_data,
                csc_indices,
                csc_indptr,
                group_codes_gpu,
                group_sizes_dev,
                rank_sums,
                tie_corr,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif cpsp.isspmatrix_csr(X_gpu):
            csr_gpu = X_gpu
            if not csr_gpu.has_sorted_indices:
                csr_gpu = csr_gpu.copy()
                csr_gpu.sort_indices()
            csr_data = csr_gpu.data.astype(cp.float32, copy=False)
            csr_indices = csr_gpu.indices.astype(cp.int32, copy=False)
            csr_indptr = csr_gpu.indptr.astype(cp.int32, copy=False)
            cp.cuda.get_current_stream().synchronize()
            _ovr.ovr_sparse_csr(
                csr_data,
                csr_indices,
                csr_indptr,
                group_codes_gpu,
                group_sizes_dev,
                rank_sums,
                tie_corr,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        else:
            dense_f32 = cp.asfortranarray(X_gpu.astype(cp.float32, copy=False))
            cp.cuda.get_current_stream().synchronize()
            _ovr.ovr_streaming(
                dense_f32,
                group_codes_gpu,
                rank_sums,
                tie_corr,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )

    # Z-scores + p-values (vectorised)
    expected = group_sizes_dev[:, None] * (n_cells + 1) / 2.0
    variance = tie_corr[None, :] * group_sizes_dev[:, None] * rest_sizes[:, None]
    variance *= (n_cells + 1) / 12.0
    std = cp.sqrt(variance)
    diff = rank_sums - expected
    if use_continuity:
        diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
    z = diff / std
    cp.nan_to_num(z, copy=False)
    p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))

    all_z = z.astype(cp.float32).get()
    all_p = p_values.get()

    return [(gi, all_z[gi], all_p[gi]) for gi in range(n_groups)]


# ---------------------------------------------------------------------------
# One-vs-reference
# ---------------------------------------------------------------------------


def _wilcoxon_with_reference(
    rg: _RankGenes,
    X,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
) -> list[tuple[int, NDArray, NDArray]]:
    """Wilcoxon test: each group vs a specific reference group.

    All test groups are processed in a single batched streaming kernel,
    eliminating per-group kernel launch overhead.
    """

    n_cells = X.shape[0]
    n_groups = len(rg.groups_order)
    ireference = rg.ireference
    n_ref = int(group_sizes[ireference])
    codes = rg.group_codes

    # ---- build row-index arrays via CSR-style cat offsets (O(n)) ----
    # scipy's coo→csr conversion is a vectorised counting sort — ~15× faster
    # than np.argsort(stable) on this shape (1.8 ms vs 27 ms for 534 k cells).
    # indptr[g] .. indptr[g+1] bracket the rows of group g in indices.
    n_rows_incl_sentinel = n_groups + 1  # last slot holds "unselected" rows
    _csr = sp.coo_matrix(
        (
            np.ones(n_cells, dtype=np.int8),
            (codes, np.arange(n_cells, dtype=np.int32)),
        ),
        shape=(n_rows_incl_sentinel, n_cells),
    ).tocsr()
    offsets_full = _csr.indptr  # int32, length n_groups + 2
    sorted_rows = _csr.indices  # int32, length n_cells

    test_group_indices: list[int] = [gi for gi in range(n_groups) if gi != ireference]
    if not test_group_indices:
        return []
    test_group_indices_np = np.asarray(test_group_indices, dtype=np.intp)

    offsets = [0]
    all_grp_rows_parts: list[np.ndarray] = []
    for gi in test_group_indices:
        g_start = int(offsets_full[gi])
        g_end = int(offsets_full[gi + 1])
        all_grp_rows_parts.append(sorted_rows[g_start:g_end])
        offsets.append(offsets[-1] + (g_end - g_start))

    all_grp_row_ids_np = np.concatenate(all_grp_rows_parts)
    n_test = len(test_group_indices)
    n_all_grp = len(all_grp_row_ids_np)
    ref_row_ids_np = sorted_rows[
        int(offsets_full[ireference]) : int(offsets_full[ireference + 1])
    ]

    # ---- warn for small groups (single aggregated warning for the batch
    # rather than one per group — emitting a warning per group is O(n_test)
    # Python overhead that dominates on workloads with thousands of groups).
    small_test = [
        str(rg.groups_order[gi])
        for gi in test_group_indices
        if int(group_sizes[gi]) <= MIN_GROUP_SIZE_WARNING
    ]
    ref_small = n_ref <= MIN_GROUP_SIZE_WARNING
    if small_test or ref_small:
        parts = []
        if small_test:
            parts.append(
                f"{len(small_test)} test group(s) have size "
                f"<= {MIN_GROUP_SIZE_WARNING} (first few: "
                f"{', '.join(small_test[:5])}{'...' if len(small_test) > 5 else ''})"
            )
        if ref_small:
            parts.append(f"reference has size {n_ref}")
        warnings.warn(
            f"Small groups detected: {'; '.join(parts)}. normal "
            "approximation of the Wilcoxon statistic may be inaccurate.",
            RuntimeWarning,
            stacklevel=4,
        )

    test_sizes = cp.asarray(
        group_sizes[test_group_indices_np].astype(np.float64, copy=False)
    )

    offsets_np = np.asarray(offsets, dtype=np.int32)

    # ---- host-streaming paths: skip bulk transfer ----
    host_sparse = isinstance(X, sp.spmatrix | sp.sparray)
    host_dense = isinstance(X, np.ndarray)

    # ---- build row maps only for paths that need original-row lookup ----
    ref_row_map_np = grp_row_map_np = None
    if (host_sparse and X.format == "csc") or (not host_sparse and not host_dense):
        ref_row_map_np = np.full(n_cells, -1, dtype=np.int32)
        ref_row_map_np[ref_row_ids_np] = np.arange(n_ref, dtype=np.int32)
        grp_row_map_np = np.full(n_cells, -1, dtype=np.int32)
        grp_row_map_np[all_grp_row_ids_np] = np.arange(n_all_grp, dtype=np.int32)

    if host_sparse or host_dense:
        # Output buffers live on the GPU (caller-provided CuPy memory);
        # kernels write directly into them, and rank_sums / tie_corr
        # feed the z-score math below without any H2D → H2D round-trip.
        rank_sums = cp.empty((n_test, n_total_genes), dtype=cp.float64)
        tie_corr_arr = cp.ones((n_test, n_total_genes), dtype=cp.float64)

        # Stats slots: 0..n_test-1 = test groups, slot n_test = reference.
        # Unselected cells carry the sentinel (n_groups_stats) which the
        # kernel skips.
        n_groups_stats = n_test + 1
        compute_vars = False
        compute_nnz = rg.comp_pts
        stats_code_lookup = np.full(n_groups + 1, n_groups_stats, dtype=np.int32)
        stats_code_lookup[test_group_indices_np] = np.arange(n_test, dtype=np.int32)
        stats_code_lookup[ireference] = n_test
        stats_codes_np = stats_code_lookup[codes]

        group_sums = cp.empty((n_groups_stats, n_total_genes), dtype=cp.float64)
        group_sq_sums = cp.empty(
            (n_groups_stats, n_total_genes) if compute_vars else (1,),
            dtype=cp.float64,
        )
        group_nnz = cp.empty(
            (n_groups_stats, n_total_genes) if compute_nnz else (1,),
            dtype=cp.float64,
        )

        if host_sparse and X.format == "csc":
            _csc_host_fn, data_arr, indices_arr = _host_sparse_fn_and_arrays(
                _wc, "ovo_streaming_csc_host", X, support_idx64=True
            )
            _csc_host_fn(
                data_arr,
                indices_arr,
                X.indptr,
                ref_row_map_np,
                grp_row_map_np,
                offsets_np,
                stats_codes_np,
                rank_sums,
                tie_corr_arr,
                group_sums,
                group_sq_sums,
                group_nnz,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_test,
                n_groups_stats=n_groups_stats,
                compute_tie_corr=tie_correct,
                compute_sq_sums=compute_vars,
                compute_nnz=compute_nnz,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif host_sparse and X.format == "csr":
            csr = X
            if not csr.has_sorted_indices:
                csr = csr.copy()
                csr.sort_indices()

            # Zero-copy mapped: pin full CSR, upload indptr + row_ids, GPU
            # kernels gather per-pack rows via UVA reads.
            _csr_host_fn, data_arr, indices_arr = _host_sparse_fn_and_arrays(
                _wc, "ovo_streaming_csr_host", csr, support_idx64=True
            )
            _csr_host_fn(
                data_arr,
                indices_arr,
                csr.indptr,
                ref_row_ids_np.astype(np.int32, copy=False),
                all_grp_row_ids_np.astype(np.int32, copy=False),
                offsets_np,
                rank_sums,
                tie_corr_arr,
                group_sums,
                group_sq_sums,
                group_nnz,
                n_full_rows=n_cells,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_cols=n_total_genes,
                n_test=n_test,
                n_groups_stats=n_groups_stats,
                compute_tie_corr=tie_correct,
                compute_sq_sums=compute_vars,
                compute_nnz=compute_nnz,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif host_sparse:
            raise TypeError(
                "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
                f"full-matrix conversion from {X.format!r}."
            )
        else:
            is_f64 = X.dtype == np.float64
            _dense_host_fn = (
                _wc.ovo_streaming_dense_host_f64
                if is_f64
                else _wc.ovo_streaming_dense_host
            )
            block = X if is_f64 else X.astype(np.float32, copy=False)
            _dense_host_fn(
                np.asfortranarray(block),
                ref_row_ids_np.astype(np.int32, copy=False),
                all_grp_row_ids_np.astype(np.int32, copy=False),
                offsets_np,
                stats_codes_np,
                rank_sums,
                tie_corr_arr,
                group_sums,
                group_sq_sums,
                group_nnz,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_test,
                n_groups_stats=n_groups_stats,
                compute_tie_corr=tie_correct,
                compute_sq_sums=compute_vars,
                compute_nnz=compute_nnz,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )

        if rg._compute_stats_in_chunks:
            _fill_ovo_stats_from_accumulators(
                rg,
                group_sums,
                group_sq_sums,
                group_nnz,
                group_sizes=group_sizes,
                test_group_indices=test_group_indices,
                n_ref=n_ref,
                compute_vars=compute_vars,
            )

    else:
        # ---- GPU path: transfer once, then dispatch ----
        X_gpu = _to_gpu_native(X, n_cells, n_total_genes)
        grp_offsets_gpu = cp.asarray(offsets_np, dtype=cp.int32)

        if rg._compute_stats_in_chunks:
            rg.X = X_gpu
            rg._compute_stats_in_chunks = False
            rg._basic_stats()

        ref_row_ids_gpu = cp.asarray(ref_row_ids_np, dtype=cp.int32)
        all_grp_row_ids_gpu = cp.asarray(all_grp_row_ids_np, dtype=cp.int32)

        rank_sums = cp.empty((n_test, n_total_genes), dtype=cp.float64)
        tie_corr_arr = cp.empty((n_test, n_total_genes), dtype=cp.float64)

        if cpsp.isspmatrix_csc(X_gpu):
            ref_row_map = cp.asarray(ref_row_map_np)
            grp_row_map = cp.asarray(grp_row_map_np)
            csc_data = X_gpu.data.astype(cp.float32, copy=False)
            csc_indices = X_gpu.indices.astype(cp.int32, copy=False)
            csc_indptr = X_gpu.indptr.astype(cp.int32, copy=False)
            cp.cuda.get_current_stream().synchronize()
            _wc.ovo_streaming_csc(
                csc_data,
                csc_indices,
                csc_indptr,
                ref_row_map,
                grp_row_map,
                grp_offsets_gpu,
                rank_sums,
                tie_corr_arr,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_cols=n_total_genes,
                n_groups=n_test,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif cpsp.isspmatrix_csr(X_gpu):
            # CSR-native: extract ref/grp rows directly
            csr_gpu = X_gpu
            if not csr_gpu.has_sorted_indices:
                csr_gpu = csr_gpu.copy()
                csr_gpu.sort_indices()
            csr_data = csr_gpu.data.astype(cp.float32, copy=False)
            csr_indices = csr_gpu.indices.astype(cp.int32, copy=False)
            csr_indptr = csr_gpu.indptr.astype(cp.int32, copy=False)
            cp.cuda.get_current_stream().synchronize()
            _wc.ovo_streaming_csr(
                csr_data,
                csr_indices,
                csr_indptr,
                ref_row_ids_gpu,
                all_grp_row_ids_gpu,
                grp_offsets_gpu,
                rank_sums,
                tie_corr_arr,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_cols=n_total_genes,
                n_groups=n_test,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif cpsp.issparse(X_gpu):
            raise TypeError(
                "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
                "full-matrix GPU sparse conversion."
            )
        else:
            # Dense device data is already resident, but extracting all genes
            # for all reference/test rows can still blow up memory.  Preserve
            # the public chunk_size escape hatch by materializing bounded
            # column blocks and stitching the CUDA outputs together.
            dense_chunk = _resolve_chunk_size(chunk_size, n_total_genes)
            for start in range(0, n_total_genes, dense_chunk):
                stop = min(start + dense_chunk, n_total_genes)
                sb_cols = stop - start
                ref_block = _extract_dense_block(X_gpu, ref_row_ids_gpu, start, stop)
                grp_block = _extract_dense_block(
                    X_gpu, all_grp_row_ids_gpu, start, stop
                )
                ref_sorted = _segmented_sort_columns(
                    ref_block,
                    np.array([0, n_ref], dtype=np.int32),
                    n_ref,
                    sb_cols,
                    1,
                )
                grp_f32 = cp.asfortranarray(grp_block.astype(cp.float32, copy=False))
                sub_rank_sums = cp.empty((n_test, sb_cols), dtype=cp.float64)
                sub_tie_corr = cp.empty((n_test, sb_cols), dtype=cp.float64)
                cp.cuda.get_current_stream().synchronize()
                _wc.ovo_streaming(
                    ref_sorted,
                    grp_f32,
                    grp_offsets_gpu,
                    sub_rank_sums,
                    sub_tie_corr,
                    n_ref=n_ref,
                    n_all_grp=n_all_grp,
                    n_cols=sb_cols,
                    n_groups=n_test,
                    compute_tie_corr=tie_correct,
                    sub_batch_cols=STREAMING_SUB_BATCH,
                )
                rank_sums[:, start:stop] = sub_rank_sums
                tie_corr_arr[:, start:stop] = sub_tie_corr

    # ---- z-scores & p-values (vectorised) ----
    n_combined = test_sizes + n_ref
    expected = test_sizes * (n_combined + 1) / 2.0
    variance = test_sizes * n_ref * (n_combined + 1) / 12.0
    if tie_correct:
        variance = variance[:, None] * tie_corr_arr
    else:
        variance = variance[:, None]

    diff = rank_sums - expected[:, None]
    if use_continuity:
        diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
    z = diff / cp.sqrt(variance)
    cp.nan_to_num(z, copy=False)
    p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))

    # Downcast scores to float32 on GPU (the scanpy output dtype); keeps
    # p-values in float64 for downstream BH correction precision.  Moving
    # the cast off CPU saves ~150 ms per stat per call on wide workloads.
    all_z = z.astype(cp.float32).get()
    all_p = p_values.get()

    return [(gi, all_z[ti], all_p[ti]) for ti, gi in enumerate(test_group_indices)]
