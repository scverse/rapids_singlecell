from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csc_matrix as cp_csc
from cupyx.scipy.sparse import issparse as cp_issparse
from cupyx.scipy.sparse import isspmatrix_csc, isspmatrix_csr

from rapids_singlecell.get import X_to_GPU, _get_obs_rep, _set_obs_rep

from ._guide_assignment_mixture import fit_poisson_gauss_mixture

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

_MIXTURE_MODEL_VAR_COLUMNS = (
    "poisson_rate",
    "gaussian_mean",
    "gaussian_std",
    "mix_probs_0",
    "mix_probs_1",
    "threshold",
    "final_loss",
)


def _format_guides(names: Sequence[str], limit: int = 10) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


def _nonzero_entries(X) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Column, row and value of every non-zero entry, dropping explicitly stored zeros."""
    X_csc = X if isspmatrix_csc(X) else (X.tocsc() if cp_issparse(X) else cp_csc(X))
    cols = cp.repeat(cp.arange(X_csc.shape[1], dtype=cp.int32), cp.diff(X_csc.indptr))
    rows = X_csc.indices.astype(cp.int32, copy=False)
    values = X_csc.data.astype(cp.float32, copy=False)
    keep = values != 0
    if not bool(keep.all()):
        cols, rows, values = cols[keep], rows[keep], values[keep]
    return cols, rows, values


def _distinct_values(
    cols: cp.ndarray, values: cp.ndarray, n_guides: int
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Sorted distinct values and their multiplicities per guide, zero padded.

    Also returns the number of distinct values, the number of non-zero cells and the maximum count per guide.
    """
    if cols.size == 0:
        empty = cp.zeros((n_guides, 0), dtype=cp.float32)
        counts = cp.zeros(n_guides, dtype=cp.int32)
        return empty, empty, counts, counts, cp.zeros(n_guides, dtype=cp.float32)
    order = cp.argsort(cols.astype(cp.float64) * (float(values.max()) + 1.0) + values)
    cols_sorted, values_sorted = cols[order], values[order]
    starts = cp.flatnonzero(
        cp.concatenate(
            [
                cp.ones(1, dtype=cp.bool_),
                (cols_sorted[1:] != cols_sorted[:-1])
                | (values_sorted[1:] != values_sorted[:-1]),
            ]
        )
    )
    multiplicity = cp.diff(cp.append(starts, cols.size))
    distinct_cols = cols_sorted[starts].astype(cp.int64)
    distinct_values = values_sorted[starts]
    n_distinct = cp.bincount(distinct_cols, minlength=n_guides)
    first_of_guide = cp.cumsum(n_distinct) - n_distinct
    rank = cp.arange(starts.size) - cp.repeat(first_of_guide, n_distinct)
    width = int(n_distinct.max())
    values_padded = cp.zeros((n_guides, width), dtype=cp.float32)
    weights_padded = cp.zeros((n_guides, width), dtype=cp.float32)
    values_padded[distinct_cols, rank] = cp.log2(distinct_values)
    weights_padded[distinct_cols, rank] = multiplicity.astype(cp.float32)
    has_values = n_distinct > 0
    max_counts = cp.zeros(n_guides, dtype=cp.float32)
    max_counts[has_values] = distinct_values[
        (first_of_guide + n_distinct - 1)[has_values]
    ]
    return (
        values_padded,
        weights_padded,
        n_distinct.astype(cp.int32),
        cp.bincount(cols, minlength=n_guides),
        max_counts,
    )


def _per_cell_assignments(
    indptr: cp.ndarray,
    indices: cp.ndarray,
    guide_names: np.ndarray,
    *,
    no_grna_assigned_key: str,
    max_assignments_per_cell: int,
    multiple_grna_assigned_key: str,
    multiple_grna_assignment_string: str,
) -> pd.Categorical:
    """Per-cell assignment labels from the CSR structure of the binary assignment."""
    n_cells = indptr.size - 1
    n_guides = len(guide_names)
    n_assigned = cp.diff(indptr)
    separator_in_names = any(
        multiple_grna_assignment_string in name for name in guide_names
    )
    if separator_in_names:
        warnings.warn(
            "Some guide names contain the separator "
            f"{multiple_grna_assignment_string!r}, so assignments of several guides "
            "to one cell cannot be split unambiguously. "
            "Pass a different `multiple_grna_assignment_string`.",
            UserWarning,
            stacklevel=3,
        )
    labels = [
        np.array([no_grna_assigned_key, multiple_grna_assigned_key], dtype=str),
        guide_names,
    ]
    codes = cp.zeros(n_cells, dtype=cp.int64)
    codes[n_assigned > max_assignments_per_cell] = 1
    single = n_assigned == 1
    codes[single] = 2 + indices[indptr[:-1][single]]
    multi = cp.flatnonzero((n_assigned > 1) & (n_assigned <= max_assignments_per_cell))
    if multi.size:
        k = int(n_assigned[multi].max())
        slot = cp.arange(k)
        positions = cp.minimum(indptr[multi][:, None] + slot, indices.size - 1)
        members = cp.where(slot < n_assigned[multi][:, None], indices[positions] + 1, 0)
        order = cp.lexsort(members.T[::-1])
        members = members[order]
        new_row = cp.ones(multi.size, dtype=cp.bool_)
        new_row[1:] = (members[1:] != members[:-1]).any(axis=1)
        inverse = cp.empty(multi.size, dtype=cp.int64)
        inverse[order] = cp.cumsum(new_row) - 1
        codes[multi] = 2 + n_guides + inverse
        unique = cp.asnumpy(members[new_row])
        joined = guide_names[unique[:, 0] - 1]
        for j in range(1, k):
            present = unique[:, j] > 0
            joined = np.char.add(
                joined,
                np.where(
                    present,
                    np.char.add(
                        multiple_grna_assignment_string,
                        guide_names[np.maximum(unique[:, j] - 1, 0)],
                    ),
                    "",
                ),
            )
        labels.append(joined)
    categories = np.concatenate(labels)
    codes = cp.asnumpy(codes)
    if separator_in_names or np.isin(labels[0], guide_names).any():
        remap, categories = pd.factorize(categories)
        codes = remap[codes]
    used = np.zeros(len(categories), dtype=bool)
    used[codes] = True
    return pd.Categorical.from_codes((np.cumsum(used) - 1)[codes], categories[used])


class GuideAssignment:
    """GPU-accelerated guide RNA assignment.

    Provides the threshold-based and mixture-model-based methods of pertpy's ``GuideAssignment`` with the same parameters and results.
    """

    def assign_by_threshold(
        self,
        adata: AnnData,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        output_layer: str = "assigned_guides",
    ) -> None:
        """Assign cells to gRNAs exceeding a count threshold.

        Each cell is assigned to every gRNA with at least ``assignment_threshold`` counts.
        Expects unnormalized count data.

        Parameters
        ----------
        adata
            Annotated data matrix of shape ``n_obs x n_vars``.
        assignment_threshold
            Minimum count for a viable assignment.
        layer
            Layer with raw counts.
            Uses ``adata.X`` if ``None``.
        output_layer
            Key under which the binary assignment matrix is stored in ``adata.layers``.
        """
        X = X_to_GPU(_get_obs_rep(adata, layer=layer))
        if cp_issparse(X):
            result = X.copy()
            result.data = (X.data >= assignment_threshold).astype(result.dtype)
            result.eliminate_zeros()
        else:
            result = (X >= assignment_threshold).astype(cp.int8)
        _set_obs_rep(adata, result, layer=output_layer)

    def assign_to_max_guide(
        self,
        adata: AnnData,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        obs_key: str = "assigned_guide",
        no_grna_assigned_key: str = "Negative",
    ) -> None:
        """Assign each cell to its most expressed gRNA.

        Each cell is assigned to the gRNA with the highest count, provided that count is at least ``assignment_threshold``.
        Ties are resolved in favour of the first of the tied gRNAs.
        Expects unnormalized count data.

        Parameters
        ----------
        adata
            Annotated data matrix of shape ``n_obs x n_vars``.
        assignment_threshold
            Minimum count for a viable assignment.
        layer
            Layer with raw counts.
            Uses ``adata.X`` if ``None``.
        obs_key
            Column in ``adata.obs`` where the assignment is stored.
        no_grna_assigned_key
            Label for cells with no guide above threshold.
        """
        X = X_to_GPU(_get_obs_rep(adata, layer=layer))
        if cp_issparse(X):
            X = X if isspmatrix_csr(X) else X.tocsr()
            max_values = cp.asnumpy(X.max(axis=1).toarray()).ravel()
        else:
            max_values = cp.asnumpy(X.max(axis=1)).ravel()
        max_positions = cp.asnumpy(X.argmax(axis=1)).ravel()
        adata.obs[obs_key] = np.where(
            max_values >= assignment_threshold,
            adata.var_names.to_numpy()[max_positions],
            no_grna_assigned_key,
        )

    def assign_mixture_model(
        self,
        adata: AnnData,
        *,
        model: Literal["poisson_gauss_mixture"] = "poisson_gauss_mixture",
        layer: str | None = None,
        assigned_guides_key: str = "assigned_guide",
        output_layer: str | None = "assigned_guides",
        no_grna_assigned_key: str = "negative",
        max_assignments_per_cell: int = 5,
        multiple_grna_assigned_key: str = "multiple",
        multiple_grna_assignment_string: str = "+",
        only_return_results: bool = False,
        show_progress: bool = False,
        n_iter: int = 1000,
        n_init_seeds: int = 10,
        seed: int = 2024,
    ) -> np.ndarray | None:
        """Assign gRNAs to cells using a Poisson-Gaussian mixture model.

        The model, priors and per-guide thresholding rule are those of ``crispat.ga_poisson_gauss`` and of pertpy's ``assign_mixture_model``.
        For each guide, the log2 of its non-zero UMI counts is modelled as a mixture of a continuous Poisson background and a Gaussian on-target component with priors ``weights ~ Dirichlet([0.9, 0.1])``, ``mu ~ Normal(3, 2)``, ``scale ~ LogNormal(2, 1)`` and ``lam ~ LogNormal(0, 1)``.
        A cell is assigned to a guide if its UMI count is at least the smallest integer ``t`` for which ``P(Normal | log2(t)) > 0.5``.

        The MAP estimate is found by multi-start EM on the distinct count values of every guide, for all guides in parallel on the GPU.
        The lowest-loss solution is kept after discarding label-switched optima and degenerate optima in which the Gaussian weight vanishes or the Gaussian collapses onto a single count value.

        Parameters
        ----------
        adata
            Annotated data matrix with guide RNA counts.
        model
            The mixture model to use.
            Only ``"poisson_gauss_mixture"`` is supported.
        layer
            Layer with raw counts.
            Uses ``adata.X`` if ``None``.
        assigned_guides_key
            Per-cell assignment is saved on ``adata.obs[assigned_guides_key]``.
        output_layer
            The binary cell-by-guide assignment matrix is saved on ``adata.layers[output_layer]``, or not at all if ``None``.
            Unlike the per-cell assignment it is not capped by ``max_assignments_per_cell``.
        no_grna_assigned_key
            Key to use when a cell is negative for all gRNAs.
        max_assignments_per_cell
            Maximum number of gRNAs that can be assigned to a cell.
        multiple_grna_assigned_key
            Key to use when more than ``max_assignments_per_cell`` gRNAs are assigned.
        multiple_grna_assignment_string
            Separator used to join multiple gRNAs assigned to one cell.
        only_return_results
            If ``True``, do not modify ``adata`` and return the assignment array.
        show_progress
            Whether to print a progress line.
        n_iter
            Maximum number of EM iterations per start.
        n_init_seeds
            Number of prior-median initialisations per guide.
        seed
            Random seed used for the prior-median initialisations.

        Returns
        -------
        The per-cell assignments if ``only_return_results`` is ``True``, else ``None``.
        """
        if model != "poisson_gauss_mixture":
            raise ValueError(
                "Model not implemented. Please use 'poisson_gauss_mixture'."
            )
        X = X_to_GPU(_get_obs_rep(adata, layer=layer))
        guide_names = np.asarray(adata.var_names, dtype=str)
        n_cells, n_guides = X.shape

        cols, rows, values = _nonzero_entries(X)
        if values.size and bool(values.min() < 0):
            raise ValueError(
                "Data contains negative values. Please use non-negative data for guide "
                "assignment with the Mixture Model."
            )
        values_padded, weights_padded, n_distinct, nonzero_counts, max_counts = (
            _distinct_values(cols, values, n_guides)
        )

        fittable = cp.asnumpy((nonzero_counts >= 2) & (max_counts >= 2))
        skipped = np.flatnonzero(~fittable)
        if skipped.size:
            warnings.warn(
                f"Skipping {skipped.size} guides with fewer than 2 expressing cells or a "
                f"maximum count below 2: {_format_guides(list(guide_names[skipped]))}.",
                UserWarning,
                stacklevel=2,
            )

        fittable_idx = np.flatnonzero(fittable)
        estimates = {
            name: np.full(n_guides, np.nan, dtype=np.float64)
            for name in _MIXTURE_MODEL_VAR_COLUMNS
        }
        thresholds = estimates["threshold"]
        if fittable_idx.size > 0:
            if show_progress:
                print(
                    f"Fitting Poisson-Gaussian mixture for {fittable_idx.size} guides "
                    "in parallel."
                )
            fittable_gpu = cp.asarray(fittable_idx)
            fit = fit_poisson_gauss_mixture(
                values_padded[fittable_gpu],
                weights_padded[fittable_gpu],
                n_values=n_distinct[fittable_gpu],
                max_counts=max_counts[fittable_gpu],
                n_iter=n_iter,
                n_init_seeds=n_init_seeds,
                seed=seed,
            )
            thresholds[fittable_idx] = fit.threshold
            estimates["poisson_rate"][fittable_idx] = fit.poisson_rate
            estimates["gaussian_mean"][fittable_idx] = fit.gaussian_mean
            estimates["gaussian_std"][fittable_idx] = fit.gaussian_std
            estimates["mix_probs_0"][fittable_idx] = fit.mix_probs[:, 0]
            estimates["mix_probs_1"][fittable_idx] = fit.mix_probs[:, 1]
            estimates["final_loss"][fittable_idx] = fit.final_loss

            unassignable = fittable_idx[np.isnan(thresholds[fittable_idx])]
            if unassignable.size:
                warnings.warn(
                    f"No positive population found for {unassignable.size} guides, which "
                    "are not assigned to any cell: "
                    f"{_format_guides(list(guide_names[unassignable]))}.",
                    UserWarning,
                    stacklevel=2,
                )

        threshold_of_entry = cp.asarray(
            np.nan_to_num(thresholds, nan=np.inf), dtype=cp.float32
        )
        assigned = values >= threshold_of_entry[cols]
        rows_assigned = rows[assigned]
        indptr = cp.zeros(n_guides + 1, dtype=cp.int64)
        cp.add.at(indptr, cols[assigned] + 1, 1)
        indptr = cp.cumsum(indptr).astype(cp.int32)
        binary = cp_csc(
            (cp.ones(rows_assigned.size, dtype=cp.float32), rows_assigned, indptr),
            shape=(n_cells, n_guides),
        )
        binary_csr = binary.tocsr()
        assignments = _per_cell_assignments(
            binary_csr.indptr,
            binary_csr.indices,
            guide_names,
            no_grna_assigned_key=no_grna_assigned_key,
            max_assignments_per_cell=max_assignments_per_cell,
            multiple_grna_assigned_key=multiple_grna_assigned_key,
            multiple_grna_assignment_string=multiple_grna_assignment_string,
        )
        if only_return_results:
            return np.asarray(assignments)

        for column, estimate in estimates.items():
            adata.var[column] = estimate
        adata.obs[assigned_guides_key] = assignments
        if output_layer is not None:
            if isspmatrix_csc(X):
                layer_value = binary
            elif cp_issparse(X):
                layer_value = binary_csr
            else:
                layer_value = binary_csr.toarray().astype(cp.int8)
            _set_obs_rep(adata, layer_value, layer=output_layer)
        return None
