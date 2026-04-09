from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc

from ._utils import _choose_chunk_size

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes

MIN_GROUP_SIZE_WARNING = 25

# ---------------------------------------------------------------------------
# CuPy RawKernels for sort-once OVO
# ---------------------------------------------------------------------------

_RANK_SUMS_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void rank_sums_from_sorted(
    const double* __restrict__ ref_sorted,   // (n_ref, n_cols) F-order
    const double* __restrict__ grp_sorted,   // (n_grp, n_cols) F-order
    double* __restrict__ rank_sums,          // (n_cols,)
    const int n_ref,
    const int n_grp,
    const int n_cols
) {
    /*  One block per gene (column).
        Threads cooperatively process group elements.
        For each group element, binary-search the sorted reference
        and the sorted group to compute the average rank in the
        combined (group + reference) set.
    */
    int col = blockIdx.x;
    if (col >= n_cols) return;

    const double* ref = ref_sorted + (long long)col * n_ref;
    const double* grp = grp_sorted + (long long)col * n_grp;

    double local_sum = 0.0;

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        double v = grp[i];

        // --- count of ref values < v ---
        int lo = 0, hi = n_ref;
        while (lo < hi) { int m = (lo + hi) >> 1; if (ref[m] < v) lo = m + 1; else hi = m; }
        int n_lt_ref = lo;

        // --- count of ref values <= v ---
        lo = n_lt_ref; hi = n_ref;
        while (lo < hi) { int m = (lo + hi) >> 1; if (ref[m] <= v) lo = m + 1; else hi = m; }
        int n_eq_ref = lo - n_lt_ref;

        // --- count of grp values < v ---
        lo = 0; hi = n_grp;
        while (lo < hi) { int m = (lo + hi) >> 1; if (grp[m] < v) lo = m + 1; else hi = m; }
        int n_lt_grp = lo;

        // --- count of grp values <= v ---
        lo = n_lt_grp; hi = n_grp;
        while (lo < hi) { int m = (lo + hi) >> 1; if (grp[m] <= v) lo = m + 1; else hi = m; }
        int n_eq_grp = lo - n_lt_grp;

        int n_lt = n_lt_ref + n_lt_grp;
        int n_eq = n_eq_ref + n_eq_grp;
        double avg_rank = (double)n_lt + ((double)n_eq + 1.0) / 2.0;
        local_sum += avg_rank;
    }

    // --- warp-level reduction ---
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    __shared__ double warp_sums[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < ((blockDim.x + 31) >> 5))
                     ? warp_sums[threadIdx.x] : 0.0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (threadIdx.x == 0) rank_sums[col] = val;
    }
}
""",
    "rank_sums_from_sorted",
    options=("--use_fast_math",),
)


_TIE_CORR_MERGE_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void tie_correction_merged(
    const double* __restrict__ ref_sorted,
    const double* __restrict__ grp_sorted,
    double* __restrict__ correction,
    const int n_ref,
    const int n_grp,
    const int n_cols
) {
    /*  One block per gene column.  Thread 0 merges the two sorted
        arrays and accumulates the tie-correction term
        sum(t^3 - t) over all tie groups of size t.
    */
    int col = blockIdx.x;
    if (col >= n_cols || threadIdx.x != 0) return;

    const double* ref = ref_sorted + (long long)col * n_ref;
    const double* grp = grp_sorted + (long long)col * n_grp;

    int i = 0, j = 0;
    double tie_sum = 0.0;

    while (i < n_ref || j < n_grp) {
        double v;
        if      (j >= n_grp)      v = ref[i];
        else if (i >= n_ref)      v = grp[j];
        else                      v = (ref[i] <= grp[j]) ? ref[i] : grp[j];

        int count = 0;
        while (i < n_ref && ref[i] == v) { ++i; ++count; }
        while (j < n_grp && grp[j] == v) { ++j; ++count; }

        if (count > 1) {
            double t = (double)count;
            tie_sum += t * t * t - t;
        }
    }

    int n = n_ref + n_grp;
    double dn = (double)n;
    double denom = dn * dn * dn - dn;
    correction[col] = (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
}
""",
    "tie_correction_merged",
    options=("--use_fast_math",),
)


_CSR_EXTRACT_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void csr_extract_dense(
    const double*    __restrict__ data,
    const int*       __restrict__ indices,
    const long long* __restrict__ indptr,
    const int*       __restrict__ row_ids,
    double*          __restrict__ out,       // F-order (n_target, n_cols)
    const int n_target,
    const int col_start,
    const int col_stop,
    const int n_cols                         // = col_stop - col_start
) {
    /*  One thread per target row.
        Binary-search the CSR index array for col_start, then
        linear-scan through [col_start, col_stop) writing to
        the dense output in column-major order.
    */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_target) return;

    int row = row_ids[tid];
    long long rs = indptr[row];
    long long re = indptr[row + 1];

    // binary search for col_start
    long long lo = rs, hi = re;
    while (lo < hi) {
        long long m = (lo + hi) >> 1;
        if (indices[m] < col_start) lo = m + 1; else hi = m;
    }

    for (long long p = lo; p < re; ++p) {
        int c = indices[p];
        if (c >= col_stop) break;
        int lc = c - col_start;
        out[(long long)lc * n_target + tid] = data[p];
    }
}
""",
    "csr_extract_dense",
    options=("--use_fast_math",),
)


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

WARP_SIZE = 32
MAX_THREADS = 512


def _round_up_to_warp(n: int) -> int:
    return min(MAX_THREADS, ((n + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE)


def _rank_sums_searchsorted(
    ref_sorted: cp.ndarray,
    grp_sorted: cp.ndarray,
) -> cp.ndarray:
    """Rank sums for *grp* via binary search in pre-sorted *ref*.

    Both must be F-order float64 ``(n_rows, n_cols)``.
    """
    n_ref, n_cols = ref_sorted.shape
    n_grp = grp_sorted.shape[0]
    rank_sums = cp.empty(n_cols, dtype=cp.float64)
    threads = _round_up_to_warp(min(n_grp, MAX_THREADS))
    _RANK_SUMS_KERNEL(
        (n_cols,),
        (threads,),
        (
            ref_sorted,
            grp_sorted,
            rank_sums,
            np.int32(n_ref),
            np.int32(n_grp),
            np.int32(n_cols),
        ),
        stream=cp.cuda.get_current_stream(),
    )
    return rank_sums


def _tie_correction_merged(
    ref_sorted: cp.ndarray,
    grp_sorted: cp.ndarray,
) -> cp.ndarray:
    """Tie-correction factor via merge of two sorted F-order arrays."""
    n_ref, n_cols = ref_sorted.shape
    n_grp = grp_sorted.shape[0]
    correction = cp.empty(n_cols, dtype=cp.float64)
    _TIE_CORR_MERGE_KERNEL(
        (n_cols,),
        (1,),
        (
            ref_sorted,
            grp_sorted,
            correction,
            np.int32(n_ref),
            np.int32(n_grp),
            np.int32(n_cols),
        ),
        stream=cp.cuda.get_current_stream(),
    )
    return correction


def _extract_dense_block_csr_gpu(
    data: cp.ndarray,
    indices: cp.ndarray,
    indptr: cp.ndarray,
    row_ids: cp.ndarray,
    *,
    col_start: int,
    col_stop: int,
) -> cp.ndarray:
    """Extract a dense F-order float64 block from GPU CSR arrays."""
    n_target = row_ids.shape[0]
    n_cols = col_stop - col_start
    out = cp.zeros((n_target, n_cols), dtype=cp.float64, order="F")
    if n_target == 0 or n_cols == 0:
        return out
    threads = _round_up_to_warp(min(n_target, MAX_THREADS))
    blocks = (n_target + threads - 1) // threads
    _CSR_EXTRACT_KERNEL(
        (blocks,),
        (threads,),
        (
            data,
            indices,
            indptr,
            row_ids,
            out,
            np.int32(n_target),
            np.int32(col_start),
            np.int32(col_stop),
            np.int32(n_cols),
        ),
        stream=cp.cuda.get_current_stream(),
    )
    return out


def _to_gpu_csr_arrays(X) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Return (data, indices, indptr) as float64/int32/int64 on GPU."""
    if isinstance(X, cpsp.csr_matrix):
        csr = X
    elif isinstance(X, cpsp.csc_matrix):
        csr = X.tocsr()
    elif isinstance(X, sp.spmatrix | sp.sparray):
        if X.format != "csr":
            X = X.tocsr()
        csr = cpsp.csr_matrix(X)
    else:
        raise TypeError(f"Expected sparse matrix, got {type(X)}")
    return (
        csr.data.astype(cp.float64, copy=False),
        csr.indices.astype(cp.int32, copy=False),
        csr.indptr.astype(cp.int64, copy=False),
    )


def _extract_dense_block(
    X,
    row_ids: cp.ndarray | None,
    start: int,
    stop: int,
    *,
    csr_arrays: tuple[cp.ndarray, cp.ndarray, cp.ndarray] | None = None,
) -> cp.ndarray:
    """Extract ``X[row_ids, start:stop]`` as dense F-order float64 on GPU."""
    if csr_arrays is not None:
        data, indices, indptr = csr_arrays
        if row_ids is None:
            n_target = int(indptr.shape[0] - 1)
            row_ids = cp.arange(n_target, dtype=cp.int32)
        return _extract_dense_block_csr_gpu(
            data, indices, indptr, row_ids, col_start=start, col_stop=stop
        )

    if isinstance(X, np.ndarray):
        if row_ids is not None:
            return cp.asarray(
                X[cp.asnumpy(row_ids), start:stop], dtype=cp.float64, order="F"
            )
        return cp.asarray(X[:, start:stop], dtype=cp.float64, order="F")

    if isinstance(X, cp.ndarray):
        chunk = X[row_ids, start:stop] if row_ids is not None else X[:, start:stop]
        return cp.asfortranarray(chunk.astype(cp.float64, copy=False))

    if isinstance(X, sp.spmatrix | sp.sparray):
        if row_ids is not None:
            idx = cp.asnumpy(row_ids)
            chunk = X[idx][:, start:stop].toarray()
        else:
            chunk = X[:, start:stop].toarray()
        return cp.asarray(chunk, dtype=cp.float64, order="F")

    if cpsp.issparse(X):
        if row_ids is not None:
            chunk = X[row_ids][:, start:stop].toarray()
        else:
            chunk = X[:, start:stop].toarray()
        return cp.asfortranarray(chunk.astype(cp.float64, copy=False))

    raise TypeError(f"Unsupported matrix type: {type(X)}")


# ---------------------------------------------------------------------------
# Existing kernels (OVR path)
# ---------------------------------------------------------------------------


def _average_ranks(
    matrix: cp.ndarray, *, return_sorted: bool = False
) -> cp.ndarray | tuple[cp.ndarray, cp.ndarray]:
    """Compute average ranks for each column using GPU kernel."""
    n_rows, n_cols = matrix.shape
    sorter = cp.argsort(matrix, axis=0)
    sorted_vals = cp.take_along_axis(matrix, sorter, axis=0)
    sorted_vals = cp.asfortranarray(sorted_vals)
    sorter = cp.asfortranarray(sorter.astype(cp.int32))
    stream = cp.cuda.get_current_stream().ptr
    _wc.average_rank(
        sorted_vals, sorter, matrix, n_rows=n_rows, n_cols=n_cols, stream=stream
    )
    if return_sorted:
        return matrix, sorted_vals
    return matrix


def _tie_correction(sorted_vals: cp.ndarray) -> cp.ndarray:
    """Tie correction factor from pre-sorted values (F-order)."""
    n_rows, n_cols = sorted_vals.shape
    correction = cp.ones(n_cols, dtype=cp.float64)
    if n_rows < 2:
        return correction
    sorted_vals = cp.asfortranarray(sorted_vals)
    stream = cp.cuda.get_current_stream().ptr
    _wc.tie_correction(
        sorted_vals, correction, n_rows=n_rows, n_cols=n_cols, stream=stream
    )
    return correction


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
    rg._basic_stats()
    X = rg.X
    n_cells, n_total_genes = rg.X.shape
    group_sizes = rg.group_sizes

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
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# One-vs-rest  (unchanged from main)
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
    """Wilcoxon test: each group vs rest of cells."""
    from rapids_singlecell._utils._csr_to_csc import _fast_csr_to_csc

    from ._utils import _get_column_block

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

    codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int64)
    group_matrix = cp.zeros((n_cells, n_groups), dtype=cp.float64)
    valid_idx = cp.where(codes_gpu < n_groups)[0]
    group_matrix[valid_idx, codes_gpu[valid_idx]] = 1.0

    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev

    chunk_width = _choose_chunk_size(chunk_size)

    all_scores: dict[int, list] = {i: [] for i in range(n_groups)}
    all_pvals: dict[int, list] = {i: [] for i in range(n_groups)}

    if isinstance(X, sp.spmatrix | sp.sparray):
        X = _fast_csr_to_csc(X) if X.format == "csr" else X.tocsc()

    for start in range(0, n_total_genes, chunk_width):
        stop = min(start + chunk_width, n_total_genes)

        block = _get_column_block(X, start, stop)

        rg._accumulate_chunk_stats_vs_rest(
            block,
            start,
            stop,
            group_matrix=group_matrix,
            group_sizes_dev=group_sizes_dev,
            n_cells=n_cells,
        )

        if tie_correct:
            ranks, sorted_vals = _average_ranks(block, return_sorted=True)
            tie_corr = _tie_correction(sorted_vals)
        else:
            ranks = _average_ranks(block)
            tie_corr = cp.ones(ranks.shape[1], dtype=cp.float64)

        rank_sums = group_matrix.T @ ranks
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

        z_host = z.get()
        p_host = p_values.get()

        for idx in range(n_groups):
            all_scores[idx].append(z_host[idx])
            all_pvals[idx].append(p_host[idx])

    return [
        (gi, np.concatenate(all_scores[gi]), np.concatenate(all_pvals[gi]))
        for gi in range(n_groups)
    ]


# ---------------------------------------------------------------------------
# One-vs-reference  (sort-once optimisation inspired by illico)
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

    Key optimisations over the naive per-group approach:

    * **No CSR->CSC conversion** -- data is read directly from CSR via a
      binary-search extraction kernel.
    * **Reference sorted once per gene chunk** -- the (typically large)
      reference group is extracted and column-sorted once, then reused
      for every test-group comparison.
    * **Rank sums via binary search** -- instead of concatenating and
      re-sorting reference + group for every pair, a GPU kernel computes
      rank sums by binary-searching the pre-sorted reference.  This
      reduces the per-group cost from O((n_ref+n_grp) log(n_ref+n_grp))
      to O(n_grp log(n_ref)).
    """
    n_groups = len(rg.groups_order)
    ireference = rg.ireference
    n_ref = int(group_sizes[ireference])

    # ---- build row-index arrays (GPU int32) for every group ----
    codes = rg.group_codes
    ref_row_ids = cp.asarray(np.where(codes == ireference)[0], dtype=cp.int32)

    group_row_ids: dict[int, cp.ndarray] = {}
    for gi in range(n_groups):
        if gi == ireference:
            continue
        group_row_ids[gi] = cp.asarray(np.where(codes == gi)[0], dtype=cp.int32)

    # ---- prepare CSR arrays on GPU if sparse (one-time transfer) ----
    csr_arrays = None
    if sp.issparse(X) or cpsp.issparse(X):
        csr_arrays = _to_gpu_csr_arrays(X)

    # ---- warn for small groups ----
    for gi in group_row_ids:
        n_group = int(group_sizes[gi])
        if n_group <= MIN_GROUP_SIZE_WARNING or n_ref <= MIN_GROUP_SIZE_WARNING:
            warnings.warn(
                f"Group {rg.groups_order[gi]} has size {n_group} "
                f"(reference {n_ref}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=4,
            )

    # ---- pre-allocate outputs ----
    all_scores: dict[int, np.ndarray] = {}
    all_pvals: dict[int, np.ndarray] = {}
    for gi in group_row_ids:
        all_scores[gi] = np.empty(n_total_genes, dtype=np.float64)
        all_pvals[gi] = np.empty(n_total_genes, dtype=np.float64)

    chunk_width = _choose_chunk_size(chunk_size)

    # ---- chunk loop (outer) x group loop (inner) ----
    for start in range(0, n_total_genes, chunk_width):
        stop = min(start + chunk_width, n_total_genes)
        n_cols = stop - start

        # Extract & sort reference columns ONCE per chunk
        ref_block = _extract_dense_block(
            X, ref_row_ids, start, stop, csr_arrays=csr_arrays
        )
        ref_sorted = cp.asfortranarray(cp.sort(ref_block, axis=0))

        # Accumulate reference stats once per chunk (CPU-data path)
        if rg._compute_stats_in_chunks and start not in rg._ref_chunk_computed:
            rg._ref_chunk_computed.add(start)
            ref_mean = ref_block.mean(axis=0)
            rg.means[ireference, start:stop] = cp.asnumpy(ref_mean)
            if n_ref > 1:
                ref_var = ref_block.var(axis=0, ddof=1)
                rg.vars[ireference, start:stop] = cp.asnumpy(ref_var)
            if rg.comp_pts:
                ref_nnz = (ref_block != 0).sum(axis=0)
                rg.pts[ireference, start:stop] = cp.asnumpy(ref_nnz / n_ref)

        for gi, grp_rows in group_row_ids.items():
            n_group = int(group_sizes[gi])
            n_combined = n_group + n_ref

            # Extract & sort group columns (small, fast)
            grp_block = _extract_dense_block(
                X, grp_rows, start, stop, csr_arrays=csr_arrays
            )
            grp_sorted = cp.asfortranarray(cp.sort(grp_block, axis=0))

            # Accumulate group stats (CPU-data path)
            if rg._compute_stats_in_chunks:
                grp_mean = grp_block.mean(axis=0)
                rg.means[gi, start:stop] = cp.asnumpy(grp_mean)
                if n_group > 1:
                    grp_var = grp_block.var(axis=0, ddof=1)
                    rg.vars[gi, start:stop] = cp.asnumpy(grp_var)
                if rg.comp_pts:
                    grp_nnz = (grp_block != 0).sum(axis=0)
                    rg.pts[gi, start:stop] = cp.asnumpy(grp_nnz / n_group)

            # ---- rank sums via binary search (no combined sort) ----
            rank_sums = _rank_sums_searchsorted(ref_sorted, grp_sorted)

            # ---- tie correction (optional) ----
            if tie_correct:
                tie_corr = _tie_correction_merged(ref_sorted, grp_sorted)
            else:
                tie_corr = cp.ones(n_cols, dtype=cp.float64)

            # ---- z-scores & p-values ----
            expected = n_group * (n_combined + 1) / 2.0
            variance = tie_corr * n_group * n_ref * (n_combined + 1) / 12.0
            diff = rank_sums - expected
            if use_continuity:
                diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
            z = diff / cp.sqrt(variance)
            cp.nan_to_num(z, copy=False)
            p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))

            all_scores[gi][start:stop] = z.get()
            all_pvals[gi][start:stop] = p_values.get()

    # ---- return in group order ----
    return [
        (gi, all_scores[gi], all_pvals[gi])
        for gi in range(n_groups)
        if gi != ireference
    ]
