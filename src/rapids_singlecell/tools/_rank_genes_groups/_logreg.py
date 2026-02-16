from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

from rapids_singlecell._compat import DaskArray, _meta_dense

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes


def logreg(rg: _RankGenes, **kwds) -> list[tuple[int, NDArray, None]]:
    """Compute logistic regression scores."""
    if len(rg.groups_order) == 1:
        msg = "Cannot perform logistic regression on a single cluster."
        raise ValueError(msg)

    n_groups = len(rg.groups_order)
    selected = rg.group_codes < n_groups
    X = rg.X[selected, :]
    grouping_logreg = rg.group_codes[selected].astype(X.dtype)

    if isinstance(X, DaskArray):
        import dask.array as da
        from cuml.dask.linear_model import LogisticRegression

        grouping_logreg = da.from_array(
            grouping_logreg,
            chunks=(X.chunks[0]),
            meta=_meta_dense(grouping_logreg.dtype),
        )
    else:
        from cuml.linear_model import LogisticRegression

    clf = LogisticRegression(**kwds)
    clf.fit(X, grouping_logreg)
    scores_all = cp.array(clf.coef_)
    if n_groups == scores_all.shape[1]:
        scores_all = scores_all.T

    results: list[tuple[int, NDArray, None]] = []
    for igroup in range(n_groups):
        if n_groups <= 2:
            scores = scores_all[0].get()
        else:
            scores = scores_all[igroup].get()

        results.append((igroup, scores, None))

        if n_groups <= 2:
            break

    return results
