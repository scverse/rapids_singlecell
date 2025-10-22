from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse

    from rapids_singlecell.preprocessing._neighbors import _Metrics


def _brute_knn(
    X: cp_sparse.spmatrix | cp.ndarray,
    Y: cp_sparse.spmatrix | cp.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    from cuml.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric=metric,
        output_type="cupy",
        metric_params=metric_kwds,
    )
    nn.fit(X)
    distances, neighbors = nn.kneighbors(Y)
    return neighbors, distances
