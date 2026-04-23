from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

EPS = 1e-9


class NoTestGroupsError(ValueError):
    """Raised when skip_empty_groups=True and no test groups remain after
    filtering.  The public ``rank_genes_groups`` catches this and returns
    quietly (after emitting a ``RuntimeWarning``) so callers can iterate
    over data subsets without wrapping each call in try/except."""


def _select_groups(
    labels: pd.Series,
    selected: list | None,
    *,
    skip_empty_groups: bool = False,
    reference: str | None = None,
) -> tuple[NDArray, NDArray[np.int32], NDArray[np.int64]]:
    """Build integer group codes from a categorical Series.

    Parameters
    ----------
    labels
        Categorical Series (from ``adata.obs[groupby]``).
    selected
        Group names to keep, or ``None`` for all groups.
        Must already include the reference group if applicable.
    skip_empty_groups
        If ``True``, drop groups with fewer than 2 cells instead of raising,
        emitting a ``RuntimeWarning`` that lists the dropped groups.  Useful
        when iterating over data subsets (e.g. per cell type) where some
        categorical levels have no cells.  The reference group is never
        silently dropped — if it has <2 cells, a ``ValueError`` is raised.
    reference
        Name of the reference group (``None`` or ``"rest"`` if there is
        no fixed reference).  Only used when ``skip_empty_groups`` is
        ``True`` to validate that the reference is not the one being
        dropped.

    Returns
    -------
    groups_order
        Selected group names as a numpy array.
    group_codes
        Per-cell int32 codes: ``0..n_groups-1`` for selected cells,
        ``n_groups`` (sentinel) for unselected cells.
    group_sizes
        Number of cells per selected group (int64).
    """
    all_categories = labels.cat.categories

    if selected is None:
        selected = list(all_categories)
    elif len(selected) > 1:
        # Sort to match original category order (scanpy convention)
        cat_order = {str(c): i for i, c in enumerate(all_categories)}
        selected.sort(key=lambda x: cat_order.get(str(x), len(all_categories)))

    # First pass: compute sizes for the currently-selected groups so we can
    # optionally drop empty/singleton ones before assigning final codes.
    orig_codes_all = labels.cat.codes.to_numpy()

    def _compute_codes_and_sizes(
        sel: list,
    ) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
        n = len(sel)
        str_to_sel = {str(name): idx for idx, name in enumerate(sel)}
        lookup = np.full(len(all_categories) + 1, n, dtype=np.int32)
        for cat_idx, cat_name in enumerate(all_categories):
            sel_idx = str_to_sel.get(str(cat_name))
            if sel_idx is not None:
                lookup[cat_idx] = sel_idx
        codes = lookup[orig_codes_all]
        sizes = np.bincount(codes, minlength=n + 1)[:n].astype(np.int64)
        return codes, sizes

    _, preview_sizes = _compute_codes_and_sizes(selected)

    if skip_empty_groups:
        ref_str = str(reference) if reference not in (None, "rest") else None
        empty_idx = [i for i, s in enumerate(preview_sizes) if s < 2]
        if empty_idx:
            empty_names = [str(selected[i]) for i in empty_idx]
            if ref_str is not None and ref_str in empty_names:
                msg = (
                    f"Reference group {ref_str!r} has <2 cells; cannot run "
                    "with skip_empty_groups=True."
                )
                raise ValueError(msg)
            warnings.warn(
                f"Dropping {len(empty_names)} group(s) with <2 cells: "
                f"{', '.join(empty_names)}",
                RuntimeWarning,
                stacklevel=3,
            )
            selected = [g for i, g in enumerate(selected) if i not in set(empty_idx)]

        # Need at least one test group once the reference is excluded.
        ref_str = str(reference) if reference not in (None, "rest") else None
        n_test = sum(1 for g in selected if str(g) != ref_str)
        if n_test == 0:
            msg = (
                "No test groups with >=2 cells remain after filtering "
                "(only the reference has enough cells)."
            )
            raise NoTestGroupsError(msg)

    n_groups = len(selected)
    groups_order = np.array(selected)

    if n_groups == 0:
        msg = "No groups with >=2 cells remain after filtering."
        raise ValueError(msg)

    group_codes, group_sizes = _compute_codes_and_sizes(selected)

    # Validate singlet groups (only triggers when skip_empty_groups=False).
    invalid_groups = {str(selected[i]) for i in range(n_groups) if group_sizes[i] < 2}
    if invalid_groups:
        msg = (
            f"Could not calculate statistics for groups {', '.join(invalid_groups)} "
            "since they only contain one sample."
        )
        raise ValueError(msg)

    return groups_order, group_codes, group_sizes


def _select_top_n(scores: NDArray, n_top: int) -> NDArray:
    """Select indices of top n scores.

    Uses argpartition + argsort for O(n + k log k) complexity where k = n_top.
    This is faster than full sorting when k << n.
    """
    if n_top >= scores.shape[0]:
        return np.argsort(scores)[::-1]
    partition = np.argpartition(scores, -n_top)[-n_top:]
    return partition[np.argsort(scores[partition])[::-1]]


def _benjamini_hochberg(pvals: NDArray) -> NDArray:
    """Adjust p-values with the Benjamini-Hochberg FDR procedure."""
    pvals_clean = np.array(pvals, copy=True)
    pvals_clean[np.isnan(pvals_clean)] = 1.0

    n_tests = pvals_clean.size
    order = np.argsort(pvals_clean)
    ordered = pvals_clean[order]
    ranks = np.arange(1, n_tests + 1, dtype=ordered.dtype) / n_tests
    adjusted = ordered / ranks
    np.minimum.accumulate(adjusted[::-1], out=adjusted[::-1])
    np.minimum(adjusted, 1.0, out=adjusted)

    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out
