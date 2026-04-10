from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes

MIN_GROUP_SIZE_WARNING = 25
STREAMING_SUB_BATCH = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_gpu_native(X, n_rows: int, n_cols: int):
    """Move *X* to GPU, preserving its format (CSR / CSC / dense)."""
    # Already on GPU
    if isinstance(X, cp.ndarray):
        return X
    if cpsp.issparse(X):
        return X

    # Host sparse → GPU sparse, same format
    if isinstance(X, sp.spmatrix | sp.sparray):
        if sp.issparse(X) and X.format == "csc":
            csc = X if X.format == "csc" else X.tocsc()
            return cpsp.csc_matrix(
                (cp.asarray(csc.data), cp.asarray(csc.indices), cp.asarray(csc.indptr)),
                shape=(n_rows, n_cols),
            )
        csr = X.tocsr() if X.format != "csr" else X
        return cpsp.csr_matrix(
            (cp.asarray(csr.data), cp.asarray(csr.indices), cp.asarray(csr.indptr)),
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
    """Extract ``X[row_ids, start:stop]`` as dense F-order on GPU (native dtype).

    The CSR kernel path outputs float64 (kernel writes double*).
    All other paths preserve the input dtype.
    """
    if csr_arrays is not None:
        data, indices, indptr = csr_arrays
        if row_ids is None:
            n_target = int(indptr.shape[0] - 1)
            row_ids = cp.arange(n_target, dtype=cp.int32)
        n_target = row_ids.shape[0]
        n_cols = stop - start
        out = cp.zeros((n_target, n_cols), dtype=cp.float64, order="F")
        if n_target > 0 and n_cols > 0:
            _wc.csr_extract_dense(
                data,
                indices,
                indptr,
                row_ids,
                out,
                n_target=n_target,
                col_start=start,
                col_stop=stop,
                stream=cp.cuda.get_current_stream().ptr,
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

    keys_in = cp.ascontiguousarray(data.astype(cp.float32).ravel(order="F"))
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

    if rg.ireference is not None:
        rg._init_stats_arrays(n_total_genes)
        return _wilcoxon_with_reference(
            rg,
            X,
            n_total_genes,
            group_sizes,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            chunk_size=chunk_size,
        )
    rg._basic_stats()
    return _wilcoxon_vs_rest(
        rg,
        X,
        n_cells,
        n_total_genes,
        group_sizes,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        chunk_size=chunk_size,
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
    chunk_size: int | None,
) -> list[tuple[int, NDArray, NDArray]]:
    """Wilcoxon test: each group vs rest of cells.

    Dispatches to CSR, CSC, or dense streaming kernel based on input format.
    No unnecessary format conversions.
    """
    from rapids_singlecell._cuda import _wilcoxon_streaming_cuda as _ws

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
        # Host-streaming: sort+rank stays on host→GPU per sub-batch.
        # Stats still need Aggregate on GPU — cheap one-time transfer.
        # _basic_stats was already called by wilcoxon() which set
        # _compute_stats_in_chunks=True for host data.  Transfer a
        # lightweight GPU copy just for Aggregate, then discard it.
        if rg._compute_stats_in_chunks:
            X_gpu_tmp = _to_gpu_native(X, n_cells, n_total_genes)
            rg.X = X_gpu_tmp
            rg._compute_stats_in_chunks = False
            rg._basic_stats()
            del X_gpu_tmp

        rank_sums_np = np.empty((n_groups, n_total_genes), dtype=np.float64)
        tie_corr_np = np.ones(n_total_genes, dtype=np.float64)

        if host_csc:
            _ws.ovr_streaming_csc_host(
                X.data.astype(np.float32, copy=False),
                X.indices.astype(np.int32, copy=False),
                X.indptr.astype(np.int32, copy=False),
                group_codes,
                rank_sums_np,
                tie_corr_np,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        else:
            _ws.ovr_streaming_dense_host(
                np.asfortranarray(X.astype(np.float32, copy=False)),
                group_codes,
                rank_sums_np,
                tie_corr_np,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )

        rank_sums = cp.asarray(rank_sums_np)
        tie_corr = cp.asarray(tie_corr_np)
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

        if cpsp.isspmatrix_csr(X_gpu):
            _ws.ovr_streaming_csr(
                X_gpu.data.astype(cp.float32, copy=False),
                X_gpu.indices.astype(cp.int32, copy=False),
                X_gpu.indptr.astype(cp.int32, copy=False),
                group_codes_gpu,
                rank_sums,
                tie_corr,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif cpsp.isspmatrix_csc(X_gpu):
            _ws.ovr_streaming_csc(
                X_gpu.data.astype(cp.float32, copy=False),
                X_gpu.indices.astype(cp.int32, copy=False),
                X_gpu.indptr.astype(cp.int32, copy=False),
                group_codes_gpu,
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
            _ws.ovr_streaming(
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
    chunk_size: int | None,
) -> list[tuple[int, NDArray, NDArray]]:
    """Wilcoxon test: each group vs a specific reference group.

    All test groups are processed in a single batched streaming kernel,
    eliminating per-group kernel launch overhead.
    """
    from rapids_singlecell._cuda import _wilcoxon_streaming_cuda as _ws

    n_cells = X.shape[0]
    n_groups = len(rg.groups_order)
    ireference = rg.ireference
    n_ref = int(group_sizes[ireference])
    codes = rg.group_codes

    # ---- build row-index arrays ----
    ref_row_ids = cp.asarray(np.where(codes == ireference)[0], dtype=cp.int32)

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

    all_grp_row_ids = cp.asarray(np.concatenate(all_grp_rows), dtype=cp.int32)
    grp_offsets_gpu = cp.asarray(offsets, dtype=cp.int32)
    n_test = len(test_group_indices)

    # ---- move data to GPU ----
    X_gpu = _to_gpu_native(X, n_cells, n_total_genes)

    # For row extraction, CSR kernel is optimal.  Dense uses cupy indexing.
    csr_arrays = None
    if cpsp.issparse(X_gpu):
        csr_gpu = X_gpu.tocsr() if not cpsp.isspmatrix_csr(X_gpu) else X_gpu
        csr_arrays = (
            csr_gpu.data.astype(cp.float64, copy=False),
            csr_gpu.indices.astype(cp.int32, copy=False),
            csr_gpu.indptr.astype(cp.int32, copy=False),
        )

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

    # ---- extract ref + grp blocks (one-shot, all genes) ----
    ref_block = _extract_dense_block(
        X_gpu, ref_row_ids, 0, n_total_genes, csr_arrays=csr_arrays
    )
    grp_block = _extract_dense_block(
        X_gpu, all_grp_row_ids, 0, n_total_genes, csr_arrays=csr_arrays
    )
    n_all_grp = grp_block.shape[0]

    # ---- stats via fused kernel ----
    _compute_grouped_stats(
        rg,
        ireference,
        ref_block,
        n_ref,
        test_group_indices=test_group_indices,
        grp_block=grp_block,
        grp_offsets_gpu=grp_offsets_gpu,
        n_test=n_test,
        n_cols=n_total_genes,
        start=0,
        stop=n_total_genes,
    )

    # ---- sort reference once ----
    ref_sorted = _segmented_sort_columns(
        ref_block, np.array([0, n_ref], dtype=np.int32), n_ref, n_total_genes, 1
    )

    # ---- streaming OVO: sort groups + binary search rank sums ----
    grp_f32 = cp.asfortranarray(grp_block.astype(cp.float32))
    rank_sums = cp.empty((n_test, n_total_genes), dtype=cp.float64)
    tie_corr_arr = cp.empty((n_test, n_total_genes), dtype=cp.float64)

    _ws.ovo_streaming(
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
        variance = cp.broadcast_to(variance[:, None], (n_test, n_total_genes)).copy()

    diff = rank_sums - expected[:, None]
    if use_continuity:
        diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
    z = diff / cp.sqrt(variance)
    cp.nan_to_num(z, copy=False)
    p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))

    all_z = z.get()
    all_p = p_values.get()

    return [(gi, all_z[ti], all_p[ti]) for ti, gi in enumerate(test_group_indices)]


def _compute_grouped_stats(
    rg: _RankGenes,
    ireference: int,
    ref_block: cp.ndarray,
    n_ref: int,
    *,
    test_group_indices: list[int],
    grp_block: cp.ndarray,
    grp_offsets_gpu: cp.ndarray,
    n_test: int,
    n_cols: int,
    start: int,
    stop: int,
) -> None:
    """Compute mean/var/pts for ref + all test groups via fused C++ kernel."""
    s = slice(start, stop)
    stream = cp.cuda.get_current_stream().ptr

    # Reference stats (single "group")
    ref_offsets = cp.array([0, n_ref], dtype=cp.int32)
    ref_sums = cp.empty((1, n_cols), dtype=cp.float64)
    ref_sq = cp.empty((1, n_cols), dtype=cp.float64)
    ref_nnz = cp.empty((1, n_cols), dtype=cp.float64)
    _wc.grouped_stats(
        ref_block,
        ref_offsets,
        ref_sums,
        ref_sq,
        ref_nnz,
        n_all_rows=n_ref,
        n_cols=n_cols,
        n_groups=1,
        compute_nnz=rg.comp_pts,
        stream=stream,
    )

    rg.means[ireference, s] = cp.asnumpy(ref_sums[0] / n_ref)
    if n_ref > 1:
        var = (ref_sq[0] - ref_sums[0] ** 2 / n_ref) / (n_ref - 1)
        rg.vars[ireference, s] = cp.asnumpy(cp.maximum(var, 0))
    if rg.comp_pts:
        rg.pts[ireference, s] = cp.asnumpy(ref_nnz[0] / n_ref)

    # All test groups in one kernel launch
    n_all_grp = grp_block.shape[0]
    grp_sums = cp.empty((n_test, n_cols), dtype=cp.float64)
    grp_sq = cp.empty((n_test, n_cols), dtype=cp.float64)
    grp_nnz = cp.empty((n_test, n_cols), dtype=cp.float64)
    _wc.grouped_stats(
        grp_block,
        grp_offsets_gpu,
        grp_sums,
        grp_sq,
        grp_nnz,
        n_all_rows=n_all_grp,
        n_cols=n_cols,
        n_groups=n_test,
        compute_nnz=rg.comp_pts,
        stream=stream,
    )

    # Vectorised mean/var computation on GPU, single D2H transfer
    sizes = cp.asarray(
        [rg.group_sizes[gi] for gi in test_group_indices], dtype=cp.float64
    )[:, None]
    means = grp_sums / sizes
    vars_ = cp.maximum((grp_sq - grp_sums**2 / sizes) / cp.maximum(sizes - 1, 1), 0)

    means_host = cp.asnumpy(means)
    vars_host = cp.asnumpy(vars_)
    for ti, gi in enumerate(test_group_indices):
        rg.means[gi, s] = means_host[ti]
        rg.vars[gi, s] = vars_host[ti]

    if rg.comp_pts:
        pts_host = cp.asnumpy(grp_nnz / sizes)
        for ti, gi in enumerate(test_group_indices):
            rg.pts[gi, s] = pts_host[ti]
