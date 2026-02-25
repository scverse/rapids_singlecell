from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc
from rapids_singlecell._utils._csr_to_csc import _fast_csr_to_csc
from rapids_singlecell._utils._multi_gpu import (
    _create_category_index_mapping,
    parse_device_ids,
)
from rapids_singlecell.preprocessing._utils import _check_gpu_X

from ._utils import _choose_chunk_size

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from numpy.typing import NDArray

    from ._core import _RankGenes

_SMALL_GROUP_THRESHOLD = 25


def _warn_small_vs_rest(
    groups_order: Sequence[str],
    group_sizes: NDArray,
    n_cells: int,
) -> None:
    """Warn when any group or its complement is too small for normal approx."""
    for name, size in zip(groups_order, group_sizes, strict=False):
        rest = n_cells - size
        if size <= _SMALL_GROUP_THRESHOLD or rest <= _SMALL_GROUP_THRESHOLD:
            warnings.warn(
                f"Group {name} has size {size} (rest {rest}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=5,
            )


def _warn_small_with_ref(
    group_name: str,
    n_group: int,
    n_ref: int,
) -> None:
    """Warn when the group or reference is too small for normal approx."""
    if n_group <= _SMALL_GROUP_THRESHOLD or n_ref <= _SMALL_GROUP_THRESHOLD:
        warnings.warn(
            f"Group {group_name} has size {n_group} "
            f"(reference {n_ref}); normal approximation "
            "of the Wilcoxon statistic may be inaccurate.",
            RuntimeWarning,
            stacklevel=5,
        )


def _alloc_sort_workspace(n_rows: int, n_cols: int) -> dict[str, cp.ndarray]:
    """Pre-allocate CuPy buffers for CUB segmented sort."""
    cub_temp_bytes = _wc.get_sort_temp_bytes(n_rows=n_rows, n_cols=n_cols)
    return {
        "sorted_vals": cp.empty((n_rows, n_cols), dtype=cp.float64, order="F"),
        "sorter": cp.empty((n_rows, n_cols), dtype=cp.int32, order="F"),
        "iota": cp.empty((n_rows, n_cols), dtype=cp.int32, order="F"),
        "offsets": cp.empty(n_cols + 1, dtype=cp.int32),
        "cub_temp": cp.empty(cub_temp_bytes, dtype=cp.uint8),
    }


def _average_ranks(
    matrix: cp.ndarray,
    workspace: dict[str, cp.ndarray] | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute average ranks and tie correction for each column."""
    n_rows, n_cols = matrix.shape
    if workspace is None:
        workspace = _alloc_sort_workspace(n_rows, n_cols)
    correction = cp.empty(n_cols, dtype=cp.float64)
    stream = cp.cuda.get_current_stream().ptr
    _wc.compute_ranks(
        matrix,
        correction,
        workspace["sorted_vals"],
        workspace["sorter"],
        workspace["iota"],
        workspace["offsets"],
        workspace["cub_temp"],
        n_rows=n_rows,
        n_cols=n_cols,
        stream=stream,
    )
    return matrix, correction


# ============================================================================
# Helpers
# ============================================================================


def _to_gpu_csc(X) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Convert any supported matrix format to CSC arrays on GPU.

    Returns (data, indices, indptr) with indices/indptr as int32.
    Never downloads GPU data to CPU.
    """
    # GPU data — convert on device, no CPU round-trip
    if isinstance(X, cpsp.spmatrix):
        if not isinstance(X, cpsp.csc_matrix):
            X = cpsp.csc_matrix(X)
        return X.data, X.indices.astype(cp.int32), X.indptr.astype(cp.int32)
    if isinstance(X, cp.ndarray):
        csc = cpsp.csc_matrix(X)
        return csc.data, csc.indices.astype(cp.int32), csc.indptr.astype(cp.int32)

    # CPU data — convert on host, upload once
    if isinstance(X, sp.spmatrix | sp.sparray):
        if X.format == "csr":
            X = _fast_csr_to_csc(X)
        elif X.format != "csc":
            X = X.tocsc()
    elif isinstance(X, np.ndarray):
        X = sp.csc_matrix(X)
    else:
        msg = f"Unsupported matrix type: {type(X)}"
        raise TypeError(msg)
    return (
        cp.asarray(X.data),
        cp.asarray(X.indices).astype(cp.int32),
        cp.asarray(X.indptr).astype(cp.int32),
    )


def _build_group_mapping(
    rg: _RankGenes,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Build CSR-like group mapping reordered for selected groups.

    Returns (cell_indices, cat_offsets, group_sizes_dev).
    """
    labels_codes = cp.asarray(rg.labels.cat.codes.values, dtype=cp.int32)
    n_cats = len(rg.labels.cat.categories)
    cat_offsets, cell_indices = _create_category_index_mapping(labels_codes, n_cats)

    cat_names = list(rg.labels.cat.categories)
    cat_to_idx = {str(name): i for i, name in enumerate(cat_names)}
    group_cat_indices = [cat_to_idx[str(name)] for name in rg.groups_order]

    n_groups = len(rg.groups_order)
    new_offsets = cp.zeros(n_groups + 1, dtype=cp.int32)
    all_cells = []
    for i, cat_idx in enumerate(group_cat_indices):
        start = int(cat_offsets[cat_idx])
        end = int(cat_offsets[cat_idx + 1])
        all_cells.append(cell_indices[start:end])
        new_offsets[i + 1] = new_offsets[i] + (end - start)

    cell_indices_reordered = (
        cp.concatenate(all_cells) if all_cells else cp.array([], dtype=cp.int32)
    )
    group_sizes_dev = cp.asarray(rg.groups_masks_obs.sum(axis=1), dtype=cp.float64)

    return cell_indices_reordered, new_offsets, group_sizes_dev


def _to_host_csc(X) -> sp.csc_matrix:
    """Convert CPU data to scipy CSC."""
    if isinstance(X, sp.spmatrix | sp.sparray):
        if X.format == "csr":
            return _fast_csr_to_csc(X)
        if X.format != "csc":
            return X.tocsc()
        return X
    if isinstance(X, np.ndarray):
        return sp.csc_matrix(X)
    msg = f"Unsupported matrix type for host path: {type(X)}"
    raise TypeError(msg)


def _build_group_mapping_host(
    rg: _RankGenes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR-like group mapping on host (numpy).

    Returns (cell_indices, cat_offsets, group_sizes) as numpy arrays.
    """
    codes = rg.labels.cat.codes.values.astype(np.int32)
    n_cats = len(rg.labels.cat.categories)

    # Build CSR-like mapping: count → cumsum → scatter
    cat_counts = np.zeros(n_cats, dtype=np.int32)
    for c in codes:
        cat_counts[c] += 1
    cat_offsets = np.zeros(n_cats + 1, dtype=np.int32)
    np.cumsum(cat_counts, out=cat_offsets[1:])

    # Stable sort by category
    cell_indices = np.argsort(codes, kind="stable").astype(np.int32)

    # Reorder to match rg.groups_order
    cat_names = list(rg.labels.cat.categories)
    cat_to_idx = {str(name): i for i, name in enumerate(cat_names)}
    group_cat_indices = [cat_to_idx[str(name)] for name in rg.groups_order]

    n_groups = len(rg.groups_order)
    new_offsets = np.zeros(n_groups + 1, dtype=np.int32)
    all_cells = []
    for i, cat_idx in enumerate(group_cat_indices):
        start = int(cat_offsets[cat_idx])
        end = int(cat_offsets[cat_idx + 1])
        all_cells.append(cell_indices[start:end])
        new_offsets[i + 1] = new_offsets[i] + (end - start)

    cell_indices_reordered = (
        np.concatenate(all_cells) if all_cells else np.array([], dtype=np.int32)
    )
    group_sizes = rg.groups_masks_obs.sum(axis=1).astype(np.float64)

    return cell_indices_reordered, new_offsets, group_sizes


def _compute_stats_from_sums(
    rg: _RankGenes,
    sums: np.ndarray,
    sq_sums: np.ndarray,
    nnz: np.ndarray,
    group_sizes: np.ndarray,
) -> None:
    """Compute means, vars, pts, and rest stats from raw sums."""
    n = group_sizes[:, None]

    rg.means = sums / n
    group_ss = sq_sums - n * rg.means**2
    rg.vars = np.maximum(group_ss / np.maximum(n - 1, 1), 0)

    if rg.comp_pts:
        rg.pts = nnz / n

    if rg.ireference is None:
        n_rest = n.sum() - n
        means_rest = (sums.sum(axis=0) - sums) / n_rest
        rest_ss = (sq_sums.sum(axis=0) - sq_sums) - n_rest * means_rest**2
        rg.means_rest = means_rest
        rg.vars_rest = np.maximum(rest_ss / np.maximum(n_rest - 1, 1), 0)

        if rg.comp_pts:
            total_nnz = nnz.sum(axis=0)
            rg.pts_rest = (total_nnz - nnz) / n_rest


# ============================================================================
# Entry point
# ============================================================================


def wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool = False,
    chunk_size: int | None = None,
    multi_gpu: bool | list[int] | str | None = False,
) -> Generator[tuple[int, NDArray, NDArray], None, None]:
    """Compute Wilcoxon rank-sum test statistics."""
    X = rg.X
    n_cells, n_total_genes = X.shape

    # Check if data is already on GPU
    try:
        _check_gpu_X(X, allow_dask=False)
    except TypeError:
        is_gpu = False
    else:
        is_gpu = True

    if is_gpu:
        # GPU data path: convert to CSC on device
        csc_data, csc_indices, csc_indptr = _to_gpu_csc(X)
        rg.X = cpsp.csc_matrix(
            (csc_data, csc_indices, csc_indptr), shape=(n_cells, n_total_genes)
        )

    rg._basic_stats()

    if rg.ireference is not None:
        yield from _wilcoxon_with_reference(
            rg,
            is_gpu=is_gpu,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            chunk_size=chunk_size,
        )
    else:
        yield from _wilcoxon_vs_rest(
            rg,
            is_gpu=is_gpu,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            chunk_size=chunk_size,
            multi_gpu=multi_gpu,
        )


# ============================================================================
# vs-rest
# ============================================================================


def _vs_rest_gpu(
    rg: _RankGenes,
    n_cells: int,
    n_total_genes: int,
    n_groups: int,
    *,
    chunk_width: int,
    tie_correct: bool,
    use_continuity: bool,
    multi_gpu: bool | list[int] | str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU multi-device pipeline for vs-rest. Returns (z_all, p_all) as numpy."""
    csc_gpu = rg.X
    csc_data = csc_gpu.data
    csc_indices = csc_gpu.indices.astype(cp.int32)
    csc_indptr = csc_gpu.indptr.astype(cp.int32)

    cell_indices, cat_offsets, group_sizes_dev = _build_group_mapping(rg)

    device_ids = parse_device_ids(multi_gpu=multi_gpu)
    n_devices = len(device_ids)
    genes_per_device = (n_total_genes + n_devices - 1) // n_devices

    # Phase 1: Transfer data to each device
    per_device: list[dict | None] = []
    for i, device_id in enumerate(device_ids):
        g_start = min(i * genes_per_device, n_total_genes)
        g_stop = min(g_start + genes_per_device, n_total_genes)
        if g_start >= g_stop:
            per_device.append(None)
            continue

        with cp.cuda.Device(device_id):
            if device_id == device_ids[0]:
                d_data = csc_data
                d_indices = csc_indices
                d_indptr = csc_indptr
                d_cells = cell_indices
                d_offsets = cat_offsets
                d_sizes = group_sizes_dev
            else:
                d_data = cp.asarray(csc_data)
                d_indices = cp.asarray(csc_indices)
                d_indptr = cp.asarray(csc_indptr)
                d_cells = cp.asarray(cell_indices)
                d_offsets = cp.asarray(cat_offsets)
                d_sizes = cp.asarray(group_sizes_dev)

            per_device.append(
                {
                    "csc_data": d_data,
                    "csc_indices": d_indices,
                    "csc_indptr": d_indptr,
                    "cell_indices": d_cells,
                    "cat_offsets": d_offsets,
                    "group_sizes": d_sizes,
                    "gene_range": (g_start, g_stop),
                    "device_id": device_id,
                }
            )

    # Phase 2: Launch chunks on each device (async — no .get() between devices)
    device_results: list[tuple[cp.ndarray, cp.ndarray, int] | None] = []
    for d in per_device:
        if d is None:
            device_results.append(None)
            continue

        device_id = d["device_id"]
        g_start, g_stop = d["gene_range"]

        with cp.cuda.Device(device_id):
            stream_ptr = cp.cuda.get_current_stream().ptr
            z_parts: list[cp.ndarray] = []
            p_parts: list[cp.ndarray] = []

            for start in range(g_start, g_stop, chunk_width):
                stop = min(start + chunk_width, g_stop)
                actual_width = stop - start

                z_chunk = cp.empty((n_groups, actual_width), dtype=cp.float64)
                p_chunk = cp.empty((n_groups, actual_width), dtype=cp.float64)

                _wc.wilcoxon_chunk_vs_rest(
                    d["csc_data"],
                    d["csc_indices"],
                    d["csc_indptr"],
                    n_cells,
                    start,
                    stop,
                    d["cell_indices"],
                    d["cat_offsets"],
                    d["group_sizes"],
                    n_groups,
                    tie_correct,
                    use_continuity,
                    z_chunk,
                    p_chunk,
                    stream=stream_ptr,
                )
                z_parts.append(z_chunk)
                p_parts.append(p_chunk)

            if z_parts:
                z_dev = (
                    cp.concatenate(z_parts, axis=1) if len(z_parts) > 1 else z_parts[0]
                )
                p_dev = (
                    cp.concatenate(p_parts, axis=1) if len(p_parts) > 1 else p_parts[0]
                )
                device_results.append((z_dev, p_dev, device_id))
            else:
                device_results.append(None)

    # Phase 3: Sync all devices then gather to host
    z_host_parts: list[np.ndarray] = []
    p_host_parts: list[np.ndarray] = []
    for result in device_results:
        if result is None:
            continue
        z_dev, p_dev, device_id = result
        with cp.cuda.Device(device_id):
            cp.cuda.Device(device_id).synchronize()
            z_host_parts.append(z_dev.get())
            p_host_parts.append(p_dev.get())

    z_all = (
        np.concatenate(z_host_parts, axis=1)
        if len(z_host_parts) > 1
        else z_host_parts[0]
    )
    p_all = (
        np.concatenate(p_host_parts, axis=1)
        if len(p_host_parts) > 1
        else p_host_parts[0]
    )
    return z_all, p_all


def _wilcoxon_vs_rest(
    rg: _RankGenes,
    *,
    is_gpu: bool,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    multi_gpu: bool | list[int] | str | None = False,
) -> Generator[tuple[int, NDArray, NDArray], None, None]:
    """Wilcoxon test: each group vs rest. Dispatches between GPU and host."""
    n_cells, n_total_genes = rg.X.shape
    n_groups = len(rg.groups_order)
    chunk_width = _choose_chunk_size(chunk_size)
    group_sizes = rg.groups_masks_obs.sum(axis=1).astype(np.int64)

    _warn_small_vs_rest(rg.groups_order, group_sizes, n_cells)

    if is_gpu:
        z_all, p_all = _vs_rest_gpu(
            rg,
            n_cells,
            n_total_genes,
            n_groups,
            chunk_width=chunk_width,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            multi_gpu=multi_gpu,
        )
    else:
        csc = _to_host_csc(rg.X)
        cell_indices, cat_offsets, group_sizes_f = _build_group_mapping_host(rg)

        out_size = n_groups * n_total_genes
        z_out = np.empty(out_size, dtype=np.float64)
        p_out = np.empty(out_size, dtype=np.float64)
        sums_out = np.empty(out_size, dtype=np.float64)
        sq_sums_out = np.empty(out_size, dtype=np.float64)
        nnz_out = np.empty(out_size, dtype=np.float64)

        csc_data = np.ascontiguousarray(csc.data)
        csc_indices = np.ascontiguousarray(csc.indices, dtype=np.int32)
        csc_indptr = np.ascontiguousarray(csc.indptr, dtype=np.int64)

        device_ids = np.array(parse_device_ids(multi_gpu=multi_gpu), dtype=np.int32)

        _wc.wilcoxon_vs_rest_host(
            csc_data,
            csc_indices,
            csc_indptr,
            cell_indices,
            cat_offsets,
            group_sizes_f,
            n_cells,
            n_groups,
            n_total_genes,
            tie_correct,
            use_continuity,
            chunk_width,
            device_ids,
            z_out,
            p_out,
            sums_out,
            sq_sums_out,
            nnz_out,
        )

        z_all = z_out.reshape(n_groups, n_total_genes)
        p_all = p_out.reshape(n_groups, n_total_genes)
        sums_2d = sums_out.reshape(n_groups, n_total_genes)
        sq_sums_2d = sq_sums_out.reshape(n_groups, n_total_genes)
        nnz_2d = nnz_out.reshape(n_groups, n_total_genes)

        _compute_stats_from_sums(rg, sums_2d, sq_sums_2d, nnz_2d, group_sizes_f)

    for group_index in range(n_groups):
        yield group_index, z_all[group_index], p_all[group_index]


# ============================================================================
# with-reference
# ============================================================================


def _wilcoxon_with_reference(
    rg: _RankGenes,
    *,
    is_gpu: bool,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
) -> Generator[tuple[int, NDArray, NDArray], None, None]:
    """Wilcoxon test: each group vs a specific reference group."""
    n_cells, n_total_genes = rg.X.shape
    chunk_width = _choose_chunk_size(chunk_size)
    group_sizes = rg.groups_masks_obs.sum(axis=1).astype(np.int64)
    mask_ref = rg.groups_masks_obs[rg.ireference]
    n_ref = int(group_sizes[rg.ireference])

    if is_gpu:
        csc_gpu = rg.X  # already a CuPy CSC on GPU from wilcoxon()
    else:
        csc = _to_host_csc(rg.X)
        csc_data = np.ascontiguousarray(csc.data)
        csc_indices = np.ascontiguousarray(csc.indices, dtype=np.int32)
        csc_indptr = np.ascontiguousarray(csc.indptr, dtype=np.int64)

    for group_index, mask_obs in enumerate(rg.groups_masks_obs):
        if group_index == rg.ireference:
            continue

        n_group = int(group_sizes[group_index])
        n_combined = n_group + n_ref

        _warn_small_with_ref(rg.groups_order[group_index], n_group, n_ref)

        mask_combined = mask_obs | mask_ref
        combined_indices = np.where(mask_combined)[0]
        group_in_combined = np.isin(combined_indices, np.where(mask_obs)[0])

        scores = np.empty(n_total_genes, dtype=np.float64)
        pvals = np.empty(n_total_genes, dtype=np.float64)

        if is_gpu:
            # Subset on GPU — no CPU round-trip
            X_subset = csc_gpu[cp.asarray(mask_combined)]
            sub_data, sub_indices, sub_indptr = _to_gpu_csc(X_subset)

            group_mask_gpu = cp.asarray(group_in_combined, dtype=cp.bool_)
            stream = cp.cuda.get_current_stream().ptr

            for start in range(0, n_total_genes, chunk_width):
                stop = min(start + chunk_width, n_total_genes)
                actual_width = stop - start

                z_chunk = cp.empty(actual_width, dtype=cp.float64)
                p_chunk = cp.empty(actual_width, dtype=cp.float64)

                _wc.wilcoxon_chunk_with_ref(
                    sub_data,
                    sub_indices,
                    sub_indptr,
                    n_combined,
                    start,
                    stop,
                    group_mask_gpu,
                    n_group,
                    n_ref,
                    tie_correct,
                    use_continuity,
                    z_chunk,
                    p_chunk,
                    stream=stream,
                )

                scores[start:stop] = z_chunk.get()
                pvals[start:stop] = p_chunk.get()
        else:
            row_map = np.full(n_cells, -1, dtype=np.int32)
            row_map[combined_indices] = np.arange(n_combined, dtype=np.int32)
            group_mask = np.ascontiguousarray(group_in_combined, dtype=np.bool_)

            g_sums = np.empty(n_total_genes, dtype=np.float64)
            g_sq_sums = np.empty(n_total_genes, dtype=np.float64)
            g_nnz = np.empty(n_total_genes, dtype=np.float64)
            r_sums = np.empty(n_total_genes, dtype=np.float64)
            r_sq_sums = np.empty(n_total_genes, dtype=np.float64)
            r_nnz = np.empty(n_total_genes, dtype=np.float64)

            _wc.wilcoxon_with_ref_host(
                csc_data,
                csc_indices,
                csc_indptr,
                row_map,
                group_mask,
                n_cells,
                n_combined,
                n_group,
                n_ref,
                n_total_genes,
                tie_correct,
                use_continuity,
                chunk_width,
                scores,
                pvals,
                g_sums,
                g_sq_sums,
                g_nnz,
                r_sums,
                r_sq_sums,
                r_nnz,
            )

            # Populate stats from sums
            g_n = max(n_group, 1)
            rg.means[group_index] = g_sums / g_n
            if n_group > 1:
                g_var = g_sq_sums / g_n - rg.means[group_index] ** 2
                rg.vars[group_index] = np.maximum(g_var * g_n / (g_n - 1), 0)
            if rg.comp_pts:
                rg.pts[group_index] = g_nnz / g_n

            # Reference stats (computed once from first non-ref group, but
            # the values are identical each time so just overwrite)
            r_n = max(n_ref, 1)
            rg.means[rg.ireference] = r_sums / r_n
            if n_ref > 1:
                r_var = r_sq_sums / r_n - rg.means[rg.ireference] ** 2
                rg.vars[rg.ireference] = np.maximum(r_var * r_n / (r_n - 1), 0)
            if rg.comp_pts:
                rg.pts[rg.ireference] = r_nnz / r_n

        yield group_index, scores, pvals
