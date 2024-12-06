from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import cupy as cp
import numpy as np
import pylibraft
from cuml.manifold.simpl_set import fuzzy_simplicial_set
from cupyx.scipy import sparse as cp_sparse
from packaging.version import parse as parse_version
from pylibraft.common import DeviceResources
from scipy import sparse as sc_sparse

from rapids_singlecell.tools._utils import _choose_representation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata import AnnData

AnyRandom = None | int | np.random.RandomState
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
_Metrics = _MetricsDense | _MetricsSparse


def _cuvs_switch():
    return parse_version(pylibraft.__version__) > parse_version("24.10")


def _brute_knn(
    X: cp_sparse.spmatrix | cp.ndarray,
    Y: cp_sparse.spmatrix | cp.ndarray,
    k: int,
    metric: _Metrics,
    metric_kwds: Mapping,
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


def _cagra_knn(
    X: cp.ndarray, Y: cp.ndarray, k: int, metric: _Metrics, metric_kwds: Mapping
) -> tuple[cp.ndarray, cp.ndarray]:
    if not _cuvs_switch():
        try:
            from pylibraft.neighbors import cagra
        except ImportError:
            raise ImportError(
                "The 'cagra' module is not available in your current RAFT installation. "
                "Please update RAFT to a version that supports 'cagra'."
            )
        resources = DeviceResources()
        build_kwargs = {"handle": resources}
        search_kwargs = {"handle": resources}
    else:
        from cuvs.neighbors import cagra

        resources = None
        build_kwargs = {}
        search_kwargs = {}

    build_params = cagra.IndexParams(metric="sqeuclidean", build_algo="nn_descent")
    index = cagra.build(build_params, X, **build_kwargs)

    n_samples = Y.shape[0]
    all_neighbors = cp.zeros((n_samples, k), dtype=cp.int32)
    all_distances = cp.zeros((n_samples, k), dtype=cp.float32)

    batchsize = 65000
    n_batches = math.ceil(n_samples / batchsize)
    for batch in range(n_batches):
        start_idx = batch * batchsize
        stop_idx = min((batch + 1) * batchsize, n_samples)
        batch_Y = Y[start_idx:stop_idx, :]

        search_params = cagra.SearchParams()
        distances, neighbors = cagra.search(
            search_params, index, batch_Y, k, **search_kwargs
        )
        all_neighbors[start_idx:stop_idx, :] = cp.asarray(neighbors)
        all_distances[start_idx:stop_idx, :] = cp.asarray(distances)

    if resources is not None:
        resources.sync()

    if metric == "euclidean":
        all_distances = cp.sqrt(all_distances)

    return all_neighbors, all_distances


def _ivf_flat_knn(
    X: cp.ndarray, Y: cp.ndarray, k: int, metric: _Metrics, metric_kwds: Mapping
) -> tuple[cp.ndarray, cp.ndarray]:
    if not _cuvs_switch():
        from pylibraft.neighbors import ivf_flat

        resources = DeviceResources()
        build_kwargs = {"handle": resources}  # pylibraft uses 'handle'
        search_kwargs = {"handle": resources}
    else:
        from cuvs.neighbors import ivf_flat

        resources = None
        build_kwargs = {}  # cuvs does not need handle/resources
        search_kwargs = {}

    n_lists = int(math.sqrt(X.shape[0]))
    index_params = ivf_flat.IndexParams(n_lists=n_lists, metric=metric)
    index = ivf_flat.build(index_params, X, **build_kwargs)
    distances, neighbors = ivf_flat.search(
        ivf_flat.SearchParams(), index, Y, k, **search_kwargs
    )

    if resources is not None:
        resources.sync()

    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)

    return neighbors, distances


def _ivf_pq_knn(
    X: cp.ndarray, Y: cp.ndarray, k: int, metric: _Metrics, metric_kwds: Mapping
) -> tuple[cp.ndarray, cp.ndarray]:
    if not _cuvs_switch():
        from pylibraft.neighbors import ivf_pq

        resources = DeviceResources()
        build_kwargs = {"handle": resources}
        search_kwargs = {"handle": resources}
    else:
        from cuvs.neighbors import ivf_pq

        resources = None
        build_kwargs = {}
        search_kwargs = {}

    n_lists = int(math.sqrt(X.shape[0]))
    index_params = ivf_pq.IndexParams(n_lists=n_lists, metric=metric)
    index = ivf_pq.build(index_params, X, **build_kwargs)
    distances, neighbors = ivf_pq.search(
        ivf_pq.SearchParams(), index, Y, k, **search_kwargs
    )
    if resources is not None:
        resources.sync()

    distances = cp.asarray(distances)
    neighbors = cp.asarray(neighbors)

    return neighbors, distances


KNN_ALGORITHMS = {
    "brute": _brute_knn,
    "cagra": _cagra_knn,
    "ivfflat": _ivf_flat_knn,
    "ivfpq": _ivf_pq_knn,
}


def _check_neighbors_X(
    X: cp_sparse.spmatrix | sc_sparse.spmatrix | np.ndarray | cp.ndarray,
    algorithm: _Alogithms,
) -> cp_sparse.spmatrix | cp.ndarray:
    """Check and convert input X to the expected format based on algorithm.

    Parameters
    ----------
    X (array-like or sparse matrix): Input data.
    algorithm (str): The algorithm to be used.

    Returns
    -------
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


def _check_metrics(algorithm: _Alogithms, metric: _Metrics) -> bool:
    """Check if the provided metric is compatible with the chosen algorithm.

    Parameters
    ----------
    algorithm (str): The algorithm to be used.
    metric (str): The metric for distance computation.

    Returns
    -------
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


def _get_connectivities(
    n_neighbors: int,
    *,
    n_obs: int,
    random_state: AnyRandom,
    metric: _Metrics,
    knn_indices: cp.ndarray,
    knn_dist: cp.ndarray,
) -> cp_sparse.coo_matrix:
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
    return connectivities


def _trimming(cnts: cp_sparse.csr_matrix, trim: int) -> cp_sparse.csr_matrix:
    from ._kernels._bbknn import cut_smaller_func, find_top_k_per_row_kernel

    n_rows = cnts.shape[0]
    vals_gpu = cp.zeros(n_rows, dtype=cp.float32)

    threads_per_block = 64
    blocks_per_grid = (n_rows + threads_per_block - 1) // threads_per_block

    shared_mem_per_thread = trim * cp.dtype(cp.float32).itemsize
    shared_mem_size = threads_per_block * shared_mem_per_thread

    find_top_k_per_row_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (cnts.data, cnts.indptr, cnts.shape[0], trim, vals_gpu),
        shared_mem=shared_mem_size,
    )
    cut_smaller_func(
        (cnts.shape[0],),
        (64,),
        (cnts.indptr, cnts.indices, cnts.data, vals_gpu, cnts.shape[0]),
    )
    cnts.eliminate_zeros()
    return cnts


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
    knn_indices, knn_dist = KNN_ALGORITHMS[algorithm](
        X_contiguous,
        X_contiguous,
        k=n_neighbors,
        metric=metric,
        metric_kwds=metric_kwds,
    )

    n_nonzero = n_obs * n_neighbors
    rowptr = cp.arange(0, n_nonzero + 1, n_neighbors)
    distances = cp_sparse.csr_matrix(
        (cp.ravel(knn_dist), cp.ravel(knn_indices), rowptr), shape=(n_obs, n_obs)
    )

    connectivities = _get_connectivities(
        n_neighbors=n_neighbors,
        n_obs=n_obs,
        random_state=random_state,
        metric=metric,
        knn_indices=knn_indices,
        knn_dist=knn_dist,
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

    params = dict(
        n_neighbors=n_neighbors,
        method="rapids",
        random_state=random_state,
        metric=metric,
        **({"metric_kwds": metric_kwds} if metric_kwds else {}),
        **({"use_rep": use_rep} if use_rep is not None else {}),
        **({"n_pcs": n_pcs} if n_pcs is not None else {}),
    )
    neighbors_dict = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": params,
    }
    adata.uns[key_added] = neighbors_dict

    adata.obsp[dists_key] = distances
    adata.obsp[conns_key] = connectivities

    return adata if copy else None


def bbknn(
    adata: AnnData,
    neighbors_within_batch: int = 3,
    n_pcs: int | None = None,
    *,
    batch_key: str | None = None,
    use_rep: str | None = None,
    random_state: AnyRandom = 0,
    algorithm: _Alogithms = "brute",
    metric: _Metrics = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    trim: int | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Batch balanced KNN, altering the KNN procedure to identify each cell's top neighbours in
    each batch separately instead of the entire cell pool with no accounting for batch.
    The nearest neighbours for each batch are then merged to create a final list of
    neighbours for the cell.

    Parameters
    ----------
    adata
        Annotated data matrix.
    neighbors_within_batch
        How many top neighbours to report for each batch; total number of neighbours
        in the initial k-nearest-neighbours computation will be this number times
        the number of batches. This then serves as the basis for the construction
        of a symmetrical matrix of connectivities.
    n_pcs
        Use this many PCs. If `n_pcs==0` and `use_rep is None`, use `.X`.
    use_rep
        Use the indicated representation. `'X'` or any key for `.obsm` is valid.
        If `None`, the representation is chosen automatically: For `.n_vars < 50`, `.X`
        is used, otherwise `'X_pca'` is used. If `'X_pca'` is not present, it's
        computed with default parameters or `n_pcs` if present.
    random_state
        A numpy random seed.
    algorithm
        The query algorithm to use. Valid options are:

        `'brute'`
            Brute-force search that computes distances to all data points, guaranteeing exact results.
        `'ivfflat'`
            Uses inverted file indexing to partition the dataset into coarse quantizer cells and performs the search within the relevant cells.
        `'ivfpq'`
            Combines inverted file indexing with product quantization to encode sub-vectors of the dataset, facilitating faster distance computation.
        `'cagra'`
            Employs the Compressed, Accurate Graph-based search to quickly find nearest neighbors by traversing a graph structure.

        Please ensure that the chosen algorithm is compatible with your dataset and the specific requirements of your search problem.
    metric
        A known metric's name or a callable that returns a distance.
    metric_kwds
        Options for the metric.
    trim
        Trim the neighbours of each cell to these many top connectivities.
        May help with population independence and improve the tidiness of clustering.
        The lower the value the more independent the individual populations,
        at the cost of more conserved batch effect.
        If `None`, sets the parameter value automatically to
        10 times `neighbors_within_batch` times the number of batches.
        Set to 0 to skip.
    key_added
        If not specified, the neighbors data is stored in `.uns['neighbors']`,
        distances and connectivities are stored in `.obsp['distances']` and
        `.obsp['connectivities']` respectively.
        If specified, the neighbors data is added to `.uns[key_added]`,
        distances are stored in `.obsp[f'{key_added}_distances']` and
        connectivities in `.obsp[f'{key_added}_connectivities']`.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:

    connectivities : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    distances : sparse matrix of dtype `float32`.
        Instead of decaying weights, this stores distances for each pair of
        neighbors.

    See `key_added` parameter description for the storage path of
    connectivities and distances.
    """

    if batch_key is None:
        raise ValueError("Please provide a batch key to perform batch-balanced KNN.")

    if batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not present in `adata.obs`.")

    adata = adata.copy() if copy else adata
    if adata.is_view:
        adata._init_as_actual(adata.copy())

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    X_contiguous = _check_neighbors_X(X, algorithm)
    _check_metrics(algorithm, metric)

    n_obs = adata.shape[0]
    batch_array = adata.obs[batch_key].values
    unique_batches = np.unique(batch_array)
    total_neighbors = neighbors_within_batch * len(unique_batches)
    knn_dist = cp.zeros((X.shape[0], total_neighbors), dtype=np.float32)
    knn_indices = cp.zeros_like(knn_dist).astype(int)

    index_array = cp.arange(n_obs)

    for idx, batch in enumerate(unique_batches):
        mask_to = batch_array == batch
        X_to = X_contiguous[mask_to]
        ind_to = index_array[mask_to]

        sub_ind, sub_dist = KNN_ALGORITHMS[algorithm](
            X_to,
            X_contiguous,
            k=neighbors_within_batch,
            metric=metric,
            metric_kwds=metric_kwds,
        )

        col_range = cp.arange(
            idx * neighbors_within_batch, (idx + 1) * neighbors_within_batch
        )
        knn_indices[:, col_range] = ind_to[sub_ind]
        knn_dist[:, col_range] = sub_dist

    n_nonzero = n_obs * total_neighbors
    rowptr = cp.arange(0, n_nonzero + 1, total_neighbors)
    distances = cp_sparse.csr_matrix(
        (cp.ravel(knn_dist), cp.ravel(knn_indices), rowptr), shape=(n_obs, n_obs)
    )
    connectivities = _get_connectivities(
        total_neighbors,
        n_obs=n_obs,
        random_state=random_state,
        metric=metric,
        knn_indices=knn_indices,
        knn_dist=knn_dist,
    )

    connectivities = connectivities.tocsr()
    if trim is None:
        trim = 10 * neighbors_within_batch
    if trim > 0:
        connectivities = _trimming(connectivities, trim)

    connectivities = connectivities.get()
    distances = distances.get()
    if key_added is None:
        key_added = "neighbors"
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = key_added + "_connectivities"
        dists_key = key_added + "_distances"

    params = dict(
        n_neighbors=total_neighbors,
        method="rapids",
        random_state=random_state,
        metric=metric,
        trim=trim,
        **({"metric_kwds": metric_kwds} if metric_kwds else {}),
        **({"use_rep": use_rep} if use_rep is not None else {}),
        **({"n_pcs": n_pcs} if n_pcs is not None else {}),
    )
    neighbors_dict = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": params,
    }
    adata.uns[key_added] = neighbors_dict
    adata.obsp[dists_key] = distances
    adata.obsp[conns_key] = connectivities

    return adata if copy else None
