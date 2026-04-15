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
        if sp.issparse(X) and X.format == "csc":
            csc = X if X.format == "csc" else X.tocsc()
            return cpsp.csc_matrix(
                (
                    cp.asarray(csc.data),
                    cp.asarray(csc.indices.astype(np.int32, copy=False)),
                    cp.asarray(csc.indptr),
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
            group_sizes_np = group_sizes.astype(np.float64, copy=False)
            _ovr.ovr_sparse_csc_host(
                X.data.astype(np.float32, copy=False),
                X.indices.astype(np.int32, copy=False),
                X.indptr.astype(np.int32, copy=False),
                group_codes,
                group_sizes_np,
                rank_sums_np,
                tie_corr_np,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        else:
            _ovr.ovr_streaming_dense_host(
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
        if rg._compute_stats_in_chunks:
            X_gpu_tmp = _to_gpu_native(X, n_cells, n_total_genes)
            rg.X = X_gpu_tmp
            rg._compute_stats_in_chunks = False
            rg._basic_stats()
            del X_gpu_tmp

        rank_sums_np = np.empty((n_test, n_total_genes), dtype=np.float64)
        tie_corr_np = np.ones((n_test, n_total_genes), dtype=np.float64)

        if host_sparse and X.format == "csc":
            _wc.ovo_streaming_csc_host(
                X.data.astype(np.float32, copy=False),
                X.indices.astype(np.int32, copy=False),
                X.indptr.astype(np.int32, copy=False),
                ref_row_map_np,
                grp_row_map_np,
                offsets_np,
                rank_sums_np,
                tie_corr_np,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_test,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        elif host_sparse:
            csr = X.tocsr() if X.format != "csr" else X
            _wc.ovo_streaming_csr_host(
                csr.data.astype(np.float32, copy=False),
                csr.indices.astype(np.int32, copy=False),
                csr.indptr.astype(np.int32, copy=False),
                ref_row_ids_np.astype(np.int32, copy=False),
                all_grp_row_ids_np.astype(np.int32, copy=False),
                offsets_np,
                rank_sums_np,
                tie_corr_np,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_test,
                nnz=csr.nnz,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )
        else:
            _wc.ovo_streaming_dense_host(
                np.asfortranarray(X.astype(np.float32, copy=False)),
                ref_row_ids_np.astype(np.int32, copy=False),
                all_grp_row_ids_np.astype(np.int32, copy=False),
                offsets_np,
                rank_sums_np,
                tie_corr_np,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_test,
                compute_tie_corr=tie_correct,
                sub_batch_cols=STREAMING_SUB_BATCH,
            )

        rank_sums = cp.asarray(rank_sums_np)
        tie_corr_arr = cp.asarray(tie_corr_np)

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
