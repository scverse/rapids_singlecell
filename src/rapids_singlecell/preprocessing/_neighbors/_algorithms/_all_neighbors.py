from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell.preprocessing._neighbors._helper import _compute_nlist

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapids_singlecell.preprocessing._neighbors import _Metrics


def _all_neighbors_knn(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    try:
        from cuvs.neighbors import all_neighbors
    except ImportError:
        raise ImportError(
            "The 'all_neighbors' algorithm is only available in cuvs >= 25.10. "
            "Please update your cuvs installation."
        )
    algo = algorithm_kwds.get("algo", "nn_descent")
    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices == 1:
        from cuvs.common import Resources

        res = Resources()
    else:
        from cuvs.common import MultiGpuResources

        res = MultiGpuResources()
    n_clusters = algorithm_kwds.get("n_clusters", n_devices)
    overlap_factor = algorithm_kwds.get("overlap_factor", 1)
    if algo == "ivf_pq" or algo == "ivfpq":
        from cuvs.neighbors import ivf_pq

        algo = "ivf_pq"
        n_lists = algorithm_kwds.get("n_lists", _compute_nlist(X.shape[0]))
        ivf_pq_params = ivf_pq.IndexParams(n_lists=n_lists)
        nn_descent_params = None
    elif algo == "nn_descent":
        from cuvs.neighbors import nn_descent

        intermediate_graph_degree = algorithm_kwds.get(
            "intermediate_graph_degree", None
        )
        nn_descent_params = nn_descent.IndexParams(
            graph_degree=k, intermediate_graph_degree=intermediate_graph_degree
        )
        ivf_pq_params = None
    else:
        raise ValueError(f"Invalid algorithm: {algo}")
    build_params = all_neighbors.AllNeighborsParams(
        algo=algo,
        overlap_factor=overlap_factor,
        n_clusters=n_clusters,
        metric="sqeuclidean",
        ivf_pq_params=ivf_pq_params,
        nn_descent_params=nn_descent_params,
    )
    neighbors = cp.zeros([X.shape[0], k], dtype=np.int64)
    distances = cp.zeros([X.shape[0], k], dtype=np.float32)

    all_neighbors.build(
        dataset=X,
        k=k,
        params=build_params,
        indices=neighbors,
        distances=distances,
        resources=res,
    )
    neighbors = neighbors.astype(np.int32)
    if metric == "euclidean":
        distances = cp.sqrt(distances)
    return neighbors, distances
