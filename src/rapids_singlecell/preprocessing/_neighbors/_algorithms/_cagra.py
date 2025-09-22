from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapids_singlecell.preprocessing._neighbors import _Metrics


def _cagra_knn(
    X: cp.ndarray,
    Y: cp.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    from cuvs.neighbors import cagra

    resources = None
    build_kwargs = {}
    search_kwargs = {}

    if metric == "euclidean":
        metric_to_use = "sqeuclidean"
    else:
        metric_to_use = metric
    build_params = cagra.IndexParams(
        metric=metric_to_use, graph_degree=k, build_algo="nn_descent"
    )
    index = cagra.build(build_params, X, **build_kwargs)

    search_params = cagra.SearchParams()
    distances, neighbors = cagra.search(search_params, index, Y, k, **search_kwargs)

    if resources is not None:
        resources.sync()

    neighbors = cp.asarray(neighbors)
    distances = cp.asarray(distances)

    if metric == "euclidean":
        distances = cp.sqrt(distances)

    return neighbors, distances
