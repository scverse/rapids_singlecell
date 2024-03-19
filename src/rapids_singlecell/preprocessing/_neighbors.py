from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Union

import cupy as cp
import numpy as np
from cuml.manifold.simpl_set import fuzzy_simplicial_set
from cupyx.scipy import sparse as cp_sparse
from pylibraft.common import DeviceResources
from scipy import sparse as sc_sparse

from rapids_singlecell.tools._utils import _choose_representation

if TYPE_CHECKING:
    from anndata import AnnData

AnyRandom = Union[None, int, np.random.RandomState]
_Alogithms = Literal["brute", "ivfflat", "ivfpq", "cagra"]
_MetricsDense = Literal[
    "l2",
    "chebyshev",
    "manhattan",
    "taxicab",
    "correlation",
    "inner_product",
    "euclidean",
    "canberra",
    "lp",
    "minkowski",
    "cosine",
    "jensenshannon",
    "linf",
    "cityblock",
    "l1",
    "haversine",
    "sqeuclidean",
]
_MetricsSparse = Literal[
    "canberra",
    "chebyshev",
    "cityblock",
    "cosine",
    "euclidean",
    "hellinger",
    "inner_product",
    "jaccard",
    "l1",
    "l2",
    "linf",
    "lp",
    "manhattan",
    "minkowski",
    "taxicab",
]
_Metrics = Union[_MetricsDense, _MetricsSparse]


def _brute_knn(X, k, metric, metric_kwds):
    from cuml.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric=metric,
        output_type="cupy",
        metric_params=metric_kwds,
    )
    nn.fit(X)
    distances, neighbors = nn.kneighbors(X)
    return neighbors, distances


def _cagra_knn(X, k, metric):
    try:
        from pylibraft.neighbors import cagra
    except ImportError:
        raise ImportError(
            "The 'cagra' module is not available in your current RAFT installation. "
            "Please update RAFT to a version that supports 'cagra'."
        )

    handle = DeviceResources()
    build_params = cagra.IndexParams(metric="sqeuclidean", build_algo="nn_descent")
    index = cagra.build(build_params, X, handle=handle)

    n_samples = X.shape[0]
    all_neighbors = cp.zeros((n_samples, k), dtype=cp.int32)
    all_distances = cp.zeros((n_samples, k), dtype=cp.float32)

    batchsize = 65000
    n_batches = math.ceil(X.shape[0] / batchsize)
    for batch in range(n_batches):
        start_idx = batch * batchsize
        stop_idx = min(batch * batchsize + batchsize, X.shape[0])
        batch_X = X[start_idx:stop_idx, :]
        search_params = cagra.SearchParams()
        distances, neighbors = cagra.search(
            search_params, index, batch_X, k, handle=handle
        )
        all_neighbors[start_idx:stop_idx, :] = cp.asarray(neighbors)
        all_distances[start_idx:stop_idx, :] = cp.asarray(distances)
        handle.sync()
    if metric == "euclidean":
        all_distances = cp.sqrt(all_distances)
    return all_neighbors, all_distances


def _ivf_flat_knn(X, k, metric):
    from pylibraft.neighbors import ivf_flat

    handle = DeviceResources()
    if X.shape[0] < 2048:
        n_lists = X.shape[0] // 2
    else:
        n_lists = 1024
    index_params = ivf_flat.IndexParams(n_lists=n_lists, metric=metric)
    index = ivf_flat.build(index_params, X, handle=handle)

    distances, neighbors = ivf_flat.search(
        ivf_flat.SearchParams(), index, X, k, handle=handle
    )
    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)
    handle.sync()
    return neighbors, distances


def _ivf_pq_knn(X, k, metric):
    from pylibraft.neighbors import ivf_pq

    handle = DeviceResources()
    if X.shape[0] < 2048:
        n_lists = X.shape[0] // 2
    else:
        n_lists = 1024
    index_params = ivf_pq.IndexParams(n_lists=n_lists, metric=metric)
    index = ivf_pq.build(index_params, X, handle=handle)

    distances, neighbors = ivf_pq.search(
        ivf_pq.SearchParams(), index, X, k, handle=handle
    )
    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)
    handle.sync()
    return neighbors, distances


def _check_neighbors_X(X, algorithm):
    """
    Check and convert input X to the expected format based on algorithm.

    Parameters:
    X (array-like or sparse matrix): Input data.
    algorithm (str): The algorithm to be used.

    Returns:
    X_contiguous (cupy.ndarray or sparse.csr_matrix): Contiguous array or CSR matrix.
    """
    if cp_sparse.issparse(X) or sc_sparse.issparse(X):
        if algorithm != "brute":
            raise ValueError(
                f"Sparse input is not supported for {algorithm} algorithm. Use 'brute' instead."
            )
        X_contiguous = X.tocsr()

    else:
        if isinstance(X, np.ndarray):
            X_contiguous = cp.asarray(X, order="C", dtype=np.float32)
        elif isinstance(X, cp.ndarray):
            X_contiguous = cp.ascontiguousarray(X, dtype=np.float32)
        else:
            raise TypeError(
                "Unsupported type for X. Expected ndarray or sparse matrix."
            )

    return X_contiguous


def _check_metrics(algorithm, metric):
    """
    Check if the provided metric is compatible with the chosen algorithm.

    Parameters:
    algorithm (str): The algorithm to be used.
    metric (str): The metric for distance computation.

    Returns:
    bool: True if the metric is compatible, otherwise ValueError is raised.
    """
    if algorithm == "brute":
        # 'brute' support all metrics, no need to check further.
        return True
    elif algorithm == "cagra":
        if metric not in ["euclidean", "sqeuclidean"]:
            raise ValueError(
                "cagra only supports 'euclidean' and 'sqeuclidean' metrics."
            )
    elif algorithm in ["ivfpq", "ivfflat"]:
        if metric not in ["euclidean", "sqeuclidean", "inner_product"]:
            raise ValueError(
                f"{algorithm} only supports 'euclidean', 'sqeuclidean', and 'inner_product' metrics."
            )
    else:
        raise NotImplementedError(f"The {algorithm} algorithm is not implemented yet.")

    return True


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use_rep: str | None = None,
    random_state: AnyRandom = 0,
    algorithm: _Alogithms = "brute",
    metric: _Metrics = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Compute a neighborhood graph of observations with cuml.

    The neighbor search efficiency of this heavily relies on cuml,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
    n_pcs
        Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
    use_rep
        Use the indicated representation. `'X'` or any key for `.obsm` is valid.
        If None, the representation is chosen automatically: For .n_vars < 50, .X
        is used, otherwise `'X_pca'` is used. If `'X_pca'` is not present, it's
        computed with default parameters or `n_pcs` if present.
    random_state
        A numpy random seed.
    algorithm
        The query algorithm to use. Valid options are:
            * 'brute': Brute-force search that computes distances to all data points, guaranteeing exact results.

            * 'ivfflat': Uses inverted file indexing to partition the dataset into coarse quantizer cells and performs the search within the relevant cells.

            * 'ivfpq': Combines inverted file indexing with product quantization to encode sub-vectors of the dataset, facilitating faster distance computation.

            * 'cagra': Employs the Compressed, Accurate Graph-based search to quickly find nearest neighbors by traversing a graph structure.

        Please ensure that the chosen algorithm is compatible with your dataset and the specific requirements of your search problem.
    metric
        A known metric's name or a callable that returns a distance.
    metric_kwds
        Options for the metric.
    key_added
        If not specified, the neighbors data is stored in .uns['neighbors'],
        distances and connectivities are stored in .obsp['distances'] and
        .obsp['connectivities'] respectively.
        If specified, the neighbors data is added to .uns[key_added],
        distances are stored in .obsp[key_added+'_distances'] and
        connectivities in .obsp[key_added+'_connectivities'].
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:

    See `key_added` parameter description for the storage path of
    connectivities and distances.

        **connectivities** : sparse matrix of dtype `float32`.
            Weighted adjacency matrix of the neighborhood graph of data
            points. Weights should be interpreted as connectivities.
        **distances** : sparse matrix of dtype `float32`.
            Instead of decaying weights, this stores distances for each pair of
            neighbors.
    """

    adata = adata.copy() if copy else adata
    if adata.is_view:
        adata._init_as_actual(adata.copy())
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

    X_contiguous = _check_neighbors_X(X, algorithm)
    _check_metrics(algorithm, metric)

    n_obs = adata.shape[0]
    if algorithm == "brute":
        knn_indices, knn_dist = _brute_knn(
            X_contiguous, n_neighbors, metric, metric_kwds
        )
    elif algorithm == "cagra":
        knn_indices, knn_dist = _cagra_knn(X_contiguous, n_neighbors, metric)
    elif algorithm == "ivfflat":
        knn_indices, knn_dist = _ivf_flat_knn(X_contiguous, n_neighbors, metric)
    elif algorithm == "ivfpq":
        knn_indices, knn_dist = _ivf_pq_knn(X_contiguous, n_neighbors, metric)

    n_nonzero = n_obs * n_neighbors
    rowptr = cp.arange(0, n_nonzero + 1, n_neighbors)
    distances = cp_sparse.csr_matrix(
        (cp.ravel(knn_dist), cp.ravel(knn_indices), rowptr), shape=(n_obs, n_obs)
    )

    set_op_mix_ratio = 1.0
    local_connectivity = 1.0
    X_conn = cp.empty((n_obs, 1), dtype=np.float32)
    connectivities = fuzzy_simplicial_set(
        X_conn,
        n_neighbors,
        random_state,
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_dist,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )
    connectivities = connectivities.tocsr().get()
    distances = distances.get()
    if key_added is None:
        key_added = "neighbors"
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = key_added + "_connectivities"
        dists_key = key_added + "_distances"
    adata.uns[key_added] = {}

    neighbors_dict = adata.uns[key_added]

    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = dists_key

    neighbors_dict["params"] = {"n_neighbors": n_neighbors}
    neighbors_dict["params"]["method"] = "rapids"
    neighbors_dict["params"]["random_state"] = random_state
    neighbors_dict["params"]["metric"] = metric
    if metric_kwds:
        neighbors_dict["params"]["metric_kwds"] = metric_kwds
    if use_rep is not None:
        neighbors_dict["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        neighbors_dict["params"]["n_pcs"] = n_pcs

    adata.obsp[dists_key] = distances
    adata.obsp[conns_key] = connectivities

    return adata if copy else None
