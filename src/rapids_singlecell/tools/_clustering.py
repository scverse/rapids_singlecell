from typing import Optional

import cudf
import numpy as np
import pandas as pd
from anndata import AnnData
from natsort import natsorted


def leiden(
    adata: AnnData,
    resolution: float = 1.0,
    n_iterations: int = 100,
    use_weights: bool = True,
    neighbors_key: Optional[int] = None,
    key_added: str = "leiden",
) -> None:
    """
    Performs Leiden Clustering using cuGraph

    Parameters
    ----------
        adata :
            annData object with 'neighbors' field.

        resolution
            A parameter value controlling the coarseness of the clustering.
            Higher values lead to more clusters.

        n_iterations
            This controls the maximum number of levels/iterations of the Louvain algorithm.
            When specified the algorithm will terminate after no more than the specified number of iterations.
            No error occurs when the algorithm terminates early in this manner.

        use_weights
            If `True`, edge weights from the graph are used in the computation
            (placing more emphasis on stronger edges).

        neighbors_key
            If not specified, `leiden` looks at `.obsp['connectivities']` for neighbors connectivities
            If specified, `leiden` looks at `.obsp['neighbors_key_ connectivities']` for neighbors connectivities

        key_added
            `adata.obs` key under which to add the cluster labels.

    """
    # Adjacency graph
    from cugraph import Graph
    from cugraph import leiden as culeiden

    if neighbors_key:
        adjacency = adata.obsp[neighbors_key + "_connectivities"]
    else:
        adjacency = adata.obsp["connectivities"]
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    if use_weights:
        weights = cudf.Series(adjacency.data)
    else:
        weights = None

    g = Graph()

    g.from_cudf_adjlist(offsets, indices, weights)

    # Cluster
    leiden_parts, _ = culeiden(g, resolution=resolution, max_iter=n_iterations)

    # Format output
    groups = (
        leiden_parts.to_pandas().sort_values("vertex")[["partition"]].to_numpy().ravel()
    )

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    # store information on the clustering parameters
    adata.uns["leiden"] = {}
    adata.uns["leiden"]["params"] = {
        "resolution": resolution,
        "n_iterations": n_iterations,
    }


def louvain(
    adata: AnnData,
    resolution: float = 1.0,
    n_iterations: int = 100,
    use_weights: bool = True,
    neighbors_key: Optional[str] = None,
    key_added: str = "louvain",
) -> None:
    """
    Performs Louvain Clustering using cuGraph

    Parameters
    ----------
        adata :
            annData object with 'neighbors' field.

        resolution
            A parameter value controlling the coarseness of the clustering.
            Higher values lead to more clusters.

        n_iterations
            This controls the maximum number of levels/iterations of the Louvain algorithm.
            When specified the algorithm will terminate after no more than the specified number of iterations.
            No error occurs when the algorithm terminates early in this manner.

        use_weights
            If `True`, edge weights from the graph are used in the computation
            (placing more emphasis on stronger edges).

        neighbors_key
            If not specified, `louvain` looks at `.obsp['connectivities']` for neighbors connectivities
            If specified, `louvain` looks at `.obsp['neighbors_key_ connectivities']` for neighbors connectivities

        key_added
            `adata.obs` key under which to add the cluster labels.

    """
    # Adjacency graph
    from cugraph import Graph
    from cugraph import louvain as culouvain

    if neighbors_key:
        adjacency = adata.obsp[neighbors_key + "_connectivities"]
    else:
        adjacency = adata.obsp["connectivities"]

    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    if use_weights:
        weights = cudf.Series(adjacency.data)
    else:
        weights = None

    g = Graph()

    g.from_cudf_adjlist(offsets, indices, weights)

    # Cluster
    louvain_parts, _ = culouvain(g, resolution=resolution, max_iter=n_iterations)

    # Format output
    groups = (
        louvain_parts.to_pandas()
        .sort_values("vertex")[["partition"]]
        .to_numpy()
        .ravel()
    )

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    adata.uns["louvain"] = {}
    adata.uns["louvain"]["params"] = {"resolution": resolution}


def kmeans(
    adata: AnnData,
    n_clusters: int = 8,
    key_added: str = "kmeans",
    random_state: float = 42,
) -> None:
    """
    KMeans is a basic but powerful clustering method which is optimized via Expectation Maximization.

    Parameters
    ----------
        adata: adata object with `.obsm['X_pca']`

        n_clusters: int (default:8)
            Number of clusters to compute

        random_state: float (default: 42)
            if you want results to be the same when you restart Python, select a
            state.

    """
    from cuml.cluster import KMeans

    kmeans_out = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
        adata.obsm["X_pca"]
    )
    groups = kmeans_out.labels_.astype(str)

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
