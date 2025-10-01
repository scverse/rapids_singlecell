from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
from packaging.version import parse as parse_version

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapids_singlecell.preprocessing._neighbors import _Metrics


def _nn_descent_knn(
    X: cp.ndarray,
    Y: cp.ndarray,
    k: int,
    *,
    metric: _Metrics,
    metric_kwds: Mapping,
    algorithm_kwds: Mapping,
) -> tuple[cp.ndarray, cp.ndarray]:
    from cuvs import __version__ as cuvs_version

    if parse_version(cuvs_version) <= parse_version("24.12"):
        raise ValueError(
            "The 'nn_descent' algorithm is only available in cuvs >= 25.02. "
            "Please update your cuvs installation."
        )
    from cuvs.neighbors import nn_descent

    # Extract intermediate_graph_degree from algorithm_kwds, with default
    intermediate_graph_degree = algorithm_kwds.get("intermediate_graph_degree", None)

    idxparams = nn_descent.IndexParams(
        graph_degree=k,
        intermediate_graph_degree=intermediate_graph_degree,
        metric="sqeuclidean" if metric == "euclidean" else metric,
    )
    idx = nn_descent.build(
        idxparams,
        dataset=X,
    )
    neighbors = cp.array(idx.graph).astype(cp.uint32)
    if metric == "euclidean" or metric == "sqeuclidean":
        from ._kernels._nn_descent import calc_distance_kernel as dist_func
    elif metric == "cosine":
        from ._kernels._nn_descent import calc_distance_kernel_cos as dist_func
    elif metric == "inner_product":
        from ._kernels._nn_descent import calc_distance_kernel_inner as dist_func
    grid_size = (X.shape[0] + 32 - 1) // 32
    distances = cp.zeros((X.shape[0], neighbors.shape[1]), dtype=cp.float32)

    dist_func(
        (grid_size,),
        (32,),
        (X, distances, neighbors, X.shape[0], X.shape[1], neighbors.shape[1]),
    )
    if metric == "euclidean":
        distances = cp.sqrt(distances)
    if metric in ("cosine", "euclidean", "sqeuclidean"):
        # Add self-neighbors and self-distances for distance metrics.
        # This is not needed for inner_product, as it is a similarity metric.
        add_self_neighbors = cp.arange(X.shape[0], dtype=cp.uint32)
        neighbors = cp.concatenate(
            (add_self_neighbors[:, None], neighbors[:, :-1]), axis=1
        )
        add_self_distances = cp.zeros((X.shape[0], 1), dtype=cp.float32)
        distances = cp.concatenate((add_self_distances, distances[:, :-1]), axis=1)
    return neighbors, distances
