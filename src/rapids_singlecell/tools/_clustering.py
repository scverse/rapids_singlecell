from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import cudf
import numpy as np
import pandas as pd
from natsort import natsorted
from packaging import version
from scanpy.tools._utils import _choose_graph
from scanpy.tools._utils_clustering import rename_groups, restrict_adjacency

if TYPE_CHECKING:
    from anndata import AnnData
    from scipy import sparse


def _create_graph(adjacency, use_weights=True):
    from cugraph import Graph

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    df = cudf.DataFrame({"source": sources, "destination": targets, "weights": weights})
    g = Graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_weights:
            g.from_cudf_edgelist(
                df, source="source", destination="destination", weight="weights"
            )
        else:
            g.from_cudf_edgelist(df, source="source", destination="destination")
    return g


def leiden(
    adata: AnnData,
    resolution: float = 1.0,
    *,
    random_state: int | None = 0,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    key_added: str = "leiden",
    adjacency: sparse.spmatrix | None = None,
    n_iterations: int = 100,
    use_weights: bool = True,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """
    Performs Leiden clustering using cuGraph, which implements the method
    described in:

    Traag, V.A., Waltman, L., & van Eck, N.J. (2019). From Louvain to
    Leiden: guaranteeing well-connected communities. Sci. Rep., 9(1), 5233.
    DOI: 10.1038/s41598-019-41695-z

    Parameters
    ----------
        adata :
            annData object

        resolution
            A parameter value controlling the coarseness of the clustering.
            (called gamma in the modularity formula). Higher values lead to
            more clusters.

        random_state
            Change the initialization of the optimization. Defaults to 0.

        restrict_to
            Restrict the clustering to the categories within the key for
            sample annotation, tuple needs to contain
            `(obs_key, list_of_categories)`.

        key_added
            `adata.obs` key under which to add the cluster labels.

        adjacency
            Sparse adjacency matrix of the graph, defaults to neighbors
            connectivities.

        n_iterations
            This controls the maximum number of levels/iterations of the
            Leiden algorithm. When specified, the algorithm will terminate
            after no more than the specified number of iterations. No error
            occurs when the algorithm terminates early in this manner.

        use_weights
            If `True`, edge weights from the graph are used in the
            computation (placing more emphasis on stronger edges).

        neighbors_key
            If not specified, `leiden` looks at `.obsp['connectivities']`
            for neighbors connectivities. If specified, `leiden` looks at
            `.obsp[.uns[neighbors_key]['connectivities_key']]` for neighbors
            connectivities.

        obsp
            Use .obsp[obsp] as adjacency. You can't specify both
            `obsp` and `neighbors_key` at the same time.

        copy
            Whether to copy `adata` or modify it in place.
    """
    # Adjacency graph
    from cugraph import leiden as culeiden

    adata = adata.copy() if copy else adata

    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata=adata,
            restrict_key=restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )

    g = _create_graph(adjacency, use_weights)
    # Cluster
    leiden_parts, _ = culeiden(
        g,
        resolution=resolution,
        random_state=random_state,
        max_iter=n_iterations,
    )

    # Format output
    groups = (
        leiden_parts.to_pandas().sort_values("vertex")[["partition"]].to_numpy().ravel()
    )
    if restrict_to is not None:
        if key_added == "leiden":
            key_added += "_R"
        groups = rename_groups(
            adata,
            key_added=key_added,
            restrict_key=restrict_key,
            restrict_categories=restrict_categories,
            restrict_indices=restrict_indices,
            groups=groups,
        )
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    # store information on the clustering parameters
    adata.uns["leiden"] = {}
    adata.uns["leiden"]["params"] = {
        "resolution": resolution,
        "random_state": random_state,
        "n_iterations": n_iterations,
    }
    return adata if copy else None


def louvain(
    adata: AnnData,
    resolution: float = 1.0,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    key_added: str = "louvain",
    adjacency: sparse.spmatrix | None = None,
    n_iterations: int = 100,
    threshold: float = 1e-7,
    use_weights: bool = True,
    neighbors_key: int | None = None,
    obsp: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """
    Performs Louvain clustering using cuGraph, which implements the method
    described in:

    Blondel, V.D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of community hierarchies in large networks, J. Stat.
    Mech., P10008. DOI: 10.1088/1742-5468/2008/10/P10008

    Parameters
    ----------
        adata :
            annData object

        resolution
            A parameter value controlling the coarseness of the clustering
            (called gamma in the modularity formula). Higher values lead to
            more clusters.

        restrict_to
            Restrict the clustering to the categories within the key for
            sample annotation, tuple needs to contain
            `(obs_key, list_of_categories)`.

        key_added
            `adata.obs` key under which to add the cluster labels.

        adjacency
            Sparse adjacency matrix of the graph, defaults to neighbors
            connectivities.

        n_iterations
            This controls the maximum number of levels/iterations of the
            Louvain algorithm. When specified the algorithm will terminate
            after no more than the specified number of iterations. No error
            occurs when the algorithm terminates early in this manner.
            Capped at 500 to prevent excessive runtime.

        threshold
            Modularity gain threshold for each level/iteration. If the gain
            of modularity between two levels of the algorithm is less than
            the given threshold then the algorithm stops and returns the
            resulting communities. Defaults to 1e-7.

        use_weights
            If `True`, edge weights from the graph are used in the
            computation (placing more emphasis on stronger edges).

        neighbors_key
            If not specified, `louvain` looks at `.obsp['connectivities']`
            for neighbors connectivities. If specified, `louvain` looks at
            `.obsp[.uns[neighbors_key]['connectivities_key']]` for neighbors
            connectivities.

        obsp
            Use `.obsp[obsp]` as adjacency. You can't specify both `obsp`
            and `neighbors_key` at the same time.

        copy
            Whether to copy `adata` or modify it in place.

    """
    # Adjacency graph
    from cugraph import __version__ as cuv
    from cugraph import louvain as culouvain

    adata = adata.copy() if copy else adata
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )

    g = _create_graph(adjacency, use_weights)

    # Cluster
    if version.parse(cuv) >= version.parse("23.08.00"):
        louvain_parts, _ = culouvain(
            g,
            resolution=resolution,
            max_level=n_iterations,
            threshold=threshold,
        )
    else:
        louvain_parts, _ = culouvain(
            g,
            resolution=resolution,
            max_iter=n_iterations,
        )

    # Format output
    groups = (
        louvain_parts.to_pandas()
        .sort_values("vertex")[["partition"]]
        .to_numpy()
        .ravel()
    )
    if restrict_to is not None:
        if key_added == "louvain":
            key_added += "_R"
        groups = rename_groups(
            adata,
            key_added=key_added,
            restrict_key=restrict_key,
            restrict_categories=restrict_categories,
            restrict_indices=restrict_indices,
            groups=groups,
        )

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    adata.uns["louvain"] = {}
    adata.uns["louvain"]["params"] = {
        "resolution": resolution,
        "n_iterations": n_iterations,
        "threshold": threshold,
    }
    return adata if copy else None


def kmeans(
    adata: AnnData,
    n_clusters: int = 8,
    *,
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
