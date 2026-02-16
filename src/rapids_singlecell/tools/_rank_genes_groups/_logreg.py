from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell._compat import DaskArray, _meta_dense

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from ._core import _RankGenes


def logreg(rg: _RankGenes, **kwds) -> Generator[tuple[int, NDArray, None], None, None]:
    """Compute logistic regression scores."""
    if len(rg.groups_order) == 1:
        msg = "Cannot perform logistic regression on a single cluster."
        raise ValueError(msg)

    X = rg.X[rg.grouping_mask.values, :]

    grouping_logreg = rg.grouping.cat.codes.to_numpy().astype(X.dtype)
    uniques = np.unique(grouping_logreg)
    for idx, cat in enumerate(uniques):
        grouping_logreg[np.where(grouping_logreg == cat)] = idx

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
    if len(rg.groups_order) == scores_all.shape[1]:
        scores_all = scores_all.T

    for igroup, _group in enumerate(rg.groups_order):
        if len(rg.groups_order) <= 2:
            scores = scores_all[0].get()
        else:
            scores = scores_all[igroup].get()

        yield igroup, scores, None

        if len(rg.groups_order) <= 2:
            break
