from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

from rapids_singlecell.preprocessing._neighbors._helper import _compute_nlist

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapids_singlecell.preprocessing._neighbors import _Metrics


def _ivf_pq_knn(
    X: cp.ndarray,
    Y: cp.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    from cuvs.neighbors import ivf_pq

    resources = None

    # Extract n_lists and nprobes from algorithm_kwds, with defaults
    n_lists = algorithm_kwds.get("n_lists", _compute_nlist(X.shape[0]))
    n_probes = algorithm_kwds.get("n_probes", 20)

    index_params = ivf_pq.IndexParams(n_lists=n_lists, metric=metric)
    index = ivf_pq.build(index_params, X)
    # Create SearchParams with nprobes if provided
    search_params = ivf_pq.SearchParams(n_probes=n_probes)
    distances, neighbors = ivf_pq.search(search_params, index, Y, k)
    if resources is not None:
        resources.sync()

    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)

    return neighbors, distances
