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


def _fill_basic_stats_from_accumulators(
    rg: _RankGenes,
    group_sums: cp.ndarray,
    group_sq_sums: cp.ndarray,
    group_nnz: cp.ndarray,
    group_sizes: np.ndarray,
    *,
    n_cells: int,
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
    group_ss = group_sq_sums - n * means**2
    vars_ = cp.maximum(group_ss / cp.maximum(n - 1, 1), 0)

    rg.means = cp.asnumpy(means)
    rg.vars = cp.asnumpy(vars_)
    rg.pts = cp.asnumpy(group_nnz / n) if rg.comp_pts else None

    if rg.ireference is None:
        n_rest = cp.float64(n_cells) - n
        total_sum = group_sums.sum(axis=0, keepdims=True)
        total_sq_sum = group_sq_sums.sum(axis=0, keepdims=True)
        rest_sums = total_sum - group_sums
        rest_means = rest_sums / n_rest
        rest_ss = (total_sq_sum - group_sq_sums) - n_rest * rest_means**2
        rg.means_rest = cp.asnumpy(rest_means)
        rg.vars_rest = cp.asnumpy(cp.maximum(rest_ss / cp.maximum(n_rest - 1, 1), 0))
        if rg.comp_pts:
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
        sq = group_sq_sums_slots[slot]
        mean = sums / size
        rg.means[gi] = cp.asnumpy(mean)
        if size > 1:
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

    # Host sparse → GPU sparse, same format.
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
        csr = X.tocsr() if X.format != "csr" else X
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
    host_dense = isinstance(X, np.ndarray)

    if host_csc or host_dense:
        # Host-streaming: sort+rank stays on host→GPU per sub-batch.  The
        # kernel also emits per-group sum, sum-of-squares, and nonzero
        # counts into caller-provided CuPy buffers, so means/vars/pts can
        # be derived without uploading the full matrix.  Outputs live on
        # the GPU and feed directly into the z-score / p-value math below.
        rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
        tie_corr = cp.ones(n_total_genes, dtype=cp.float64)
        group_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
        group_sq_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
        group_nnz = cp.empty((n_groups, n_total_genes), dtype=cp.float64)

        if host_csc:
            group_sizes_np = group_sizes.astype(np.float64, copy=False)
            # Native host dtype is preserved and uploaded once per sub-batch;
            # a pre-sort kernel casts to float32 for the sort keys while
            # accumulating stats in float64 from the original values.
            is_f64 = X.data.dtype == np.float64
            is_i64 = X.indptr.dtype == np.int64
            if is_f64 and is_i64:
                _csc_host_fn = _ovr.ovr_sparse_csc_host_f64_i64
            elif is_f64:
                _csc_host_fn = _ovr.ovr_sparse_csc_host_f64
            elif is_i64:
                _csc_host_fn = _ovr.ovr_sparse_csc_host_i64
            else:
                _csc_host_fn = _ovr.ovr_sparse_csc_host
            data_arr = X.data if is_f64 else X.data.astype(np.float32, copy=False)
            _csc_host_fn(
                data_arr,
                X.indices.astype(np.int32, copy=False),
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
                sub_batch_cols=STREAMING_SUB_BATCH,
            )

        if rg._compute_stats_in_chunks:
            _fill_basic_stats_from_accumulators(
                rg,
                group_sums,
                group_sq_sums,
                group_nnz,
                group_sizes.astype(np.float64, copy=False),
                n_cells=n_cells,
            )
    else:
        # GPU data or host CSR → transfer to GPU, use GPU kernels
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
            _ovr.ovr_sparse_csc(
                X_gpu.data.astype(cp.float32, copy=False),
                X_gpu.indices.astype(cp.int32, copy=False),
                X_gpu.indptr.astype(cp.int32, copy=False),
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
            _ovr.ovr_sparse_csr(
                X_gpu.data.astype(cp.float32, copy=False),
                X_gpu.indices.astype(cp.int32, copy=False),
                X_gpu.indptr.astype(cp.int32, copy=False),
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

    all_z = z.get()
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

    # ---- build row-index arrays ----
    test_group_indices: list[int] = []
    all_grp_rows: list[np.ndarray] = []
    offsets = [0]
    for gi in range(n_groups):
        if gi == ireference:
            continue
        rows = np.where(codes == gi)[0]
        test_group_indices.append(gi)
        all_grp_rows.append(rows)
        offsets.append(offsets[-1] + len(rows))

    if not test_group_indices:
        return []

    all_grp_row_ids_np = np.concatenate(all_grp_rows)
    grp_offsets_gpu = cp.asarray(offsets, dtype=cp.int32)
    n_test = len(test_group_indices)
    n_all_grp = len(all_grp_row_ids_np)
    ref_row_ids_np = np.where(codes == ireference)[0]

    # ---- warn for small groups ----
    for gi in test_group_indices:
        n_group = int(group_sizes[gi])
        if n_group <= MIN_GROUP_SIZE_WARNING or n_ref <= MIN_GROUP_SIZE_WARNING:
            warnings.warn(
                f"Group {rg.groups_order[gi]} has size {n_group} "
                f"(reference {n_ref}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=4,
            )

    test_sizes = cp.asarray(
        [group_sizes[gi] for gi in test_group_indices], dtype=cp.float64
    )

    # ---- build row maps (numpy, for both host and GPU CSC paths) ----
    ref_row_map_np = np.full(n_cells, -1, dtype=np.int32)
    ref_row_map_np[ref_row_ids_np] = np.arange(n_ref, dtype=np.int32)
    grp_row_map_np = np.full(n_cells, -1, dtype=np.int32)
    grp_row_map_np[all_grp_row_ids_np] = np.arange(n_all_grp, dtype=np.int32)
    offsets_np = np.asarray(offsets, dtype=np.int32)

    # ---- host-streaming paths: skip bulk transfer ----
    host_sparse = isinstance(X, sp.spmatrix | sp.sparray)
    host_dense = isinstance(X, np.ndarray)
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
        stats_codes_np = np.full(n_cells, n_groups_stats, dtype=np.int32)
        for i, gi in enumerate(test_group_indices):
            stats_codes_np[codes == gi] = i
        stats_codes_np[codes == ireference] = n_test

        group_sums = cp.empty((n_groups_stats, n_total_genes), dtype=cp.float64)
        group_sq_sums = cp.empty((n_groups_stats, n_total_genes), dtype=cp.float64)
        group_nnz = cp.empty((n_groups_stats, n_total_genes), dtype=cp.float64)

        if host_sparse and X.format == "csc":
            is_f64 = X.data.dtype == np.float64
            is_i64 = X.indptr.dtype == np.int64
            if is_f64 and is_i64:
                _csc_host_fn = _wc.ovo_streaming_csc_host_f64_i64
            elif is_f64:
                _csc_host_fn = _wc.ovo_streaming_csc_host_f64
            elif is_i64:
                _csc_host_fn = _wc.ovo_streaming_csc_host_i64
            else:
                _csc_host_fn = _wc.ovo_streaming_csc_host
            data_arr = X.data if is_f64 else X.data.astype(np.float32, copy=False)
            _csc_host_fn(
                data_arr,
                X.indices.astype(np.int32, copy=False),
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
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif host_sparse:
            csr = X.tocsr() if X.format != "csr" else X
            is_f64 = csr.data.dtype == np.float64
            is_i64 = csr.indptr.dtype == np.int64
            if is_f64 and is_i64:
                _csr_host_fn = _wc.ovo_streaming_csr_host_f64_i64
            elif is_f64:
                _csr_host_fn = _wc.ovo_streaming_csr_host_f64
            elif is_i64:
                _csr_host_fn = _wc.ovo_streaming_csr_host_i64
            else:
                _csr_host_fn = _wc.ovo_streaming_csr_host
            data_arr = csr.data if is_f64 else csr.data.astype(np.float32, copy=False)
            _csr_host_fn(
                data_arr,
                csr.indices.astype(np.int32, copy=False),
                csr.indptr,
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
                nnz=csr.nnz,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
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
            )

    else:
        # ---- GPU path: transfer once, then dispatch ----
        X_gpu = _to_gpu_native(X, n_cells, n_total_genes)

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
            _wc.ovo_streaming_csc(
                X_gpu.data.astype(cp.float32, copy=False),
                X_gpu.indices.astype(cp.int32, copy=False),
                X_gpu.indptr.astype(cp.int32, copy=False),
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
        elif cpsp.issparse(X_gpu):
            # CSR-native: extract ref/grp rows directly
            csr_gpu = X_gpu.tocsr() if not cpsp.isspmatrix_csr(X_gpu) else X_gpu
            _wc.ovo_streaming_csr(
                csr_gpu.data.astype(cp.float32, copy=False),
                csr_gpu.indices.astype(cp.int32, copy=False),
                csr_gpu.indptr.astype(cp.int32, copy=False),
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
        else:
            # Dense: extract blocks, sort, stream
            ref_block = _extract_dense_block(X_gpu, ref_row_ids_gpu, 0, n_total_genes)
            grp_block = _extract_dense_block(
                X_gpu, all_grp_row_ids_gpu, 0, n_total_genes
            )
            ref_sorted = _segmented_sort_columns(
                ref_block,
                np.array([0, n_ref], dtype=np.int32),
                n_ref,
                n_total_genes,
                1,
            )
            grp_f32 = cp.asfortranarray(grp_block.astype(cp.float32, copy=False))
            _wc.ovo_streaming(
                ref_sorted,
                grp_f32,
                grp_offsets_gpu,
                rank_sums,
                tie_corr_arr,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_cols=n_total_genes,
                n_groups=n_test,
                compute_tie_corr=tie_correct,
            )

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

    all_z = z.get()
    all_p = p_values.get()

    return [(gi, all_z[ti], all_p[ti]) for ti, gi in enumerate(test_group_indices)]
