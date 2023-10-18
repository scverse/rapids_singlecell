from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Union

import cupy as cp
import numpy as np
from anndata import AnnData
from cuml.manifold.simpl_set import fuzzy_simplicial_set
from cuml.neighbors import NearestNeighbors
from cupyx.scipy import sparse

from rapids_singlecell.tools._utils import _choose_representation

AnyRandom = Union[None, int, np.random.RandomState]
_Alogithms = Literal["auto", "brute", "ivfflat", "ivfpq"]
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


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    random_state: AnyRandom = 0,
    metric: _Metrics = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    key_added: Optional[str] = None,
    algorithm: _Alogithms = "auto",
    copy: bool = False,
) -> Optional[AnnData]:
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

    nn = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm=algorithm, metric=metric, output_type="cupy"
    )
    if isinstance(X, cp.ndarray):
        X_contiguous = cp.ascontiguousarray(X, dtype=np.float32)
    elif isinstance(X, sparse.spmatrix):
        X_contiguous = X
    else:
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    nn.fit(X_contiguous)
    distances = nn.kneighbors_graph(X_contiguous, mode="distance")
    n_obs = adata.shape[0]
    knn_dist = distances.data.reshape(n_obs, n_neighbors)
    knn_indices = distances.indices.reshape(n_obs, n_neighbors)
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
