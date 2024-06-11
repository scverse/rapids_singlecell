from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
from cuml import UMAP
from cuml.manifold.umap_utils import find_ab_params
from cupyx.scipy import sparse
from scanpy._utils import NeighborsView
from sklearn.utils import check_random_state

from ._utils import _choose_representation

if TYPE_CHECKING:
    from anndata import AnnData

_InitPos = Literal["auto", "spectral", "random"]


def umap(
    adata: AnnData,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: _InitPos = "auto",
    random_state=0,
    a: float | None = None,
    b: float | None = None,
    copy: bool = False,
    neighbors_key: str | None = None,
) -> AnnData | None:
    """\
    Embed the neighborhood graph using UMAP's cuml implementation.

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. Besides tending to
    be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout rapids-singlecell using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded
        points will be spread out.
    spread
        The effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are.
    n_components
        The number of dimensions of the embedding.
    maxiter
        The number of iterations (epochs) of the optimization. Called `n_epochs`
        in the original UMAP.
    alpha
        The initial learning rate for the embedding optimization.
    negative_sample_rate
        The number of negative edge/1-simplex samples to use per positive
        edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos
        How to initialize the low dimensional embedding. Called `init` in the
        original UMAP. Options are:

            * 'auto': chooses 'spectral' for `'n_samples' < 1000000`, 'random' otherwise.
            * 'spectral': use a spectral embedding of the graph.
            * 'random': assign initial embedding positions at random.

        .. note::
            If your embedding looks odd it's recommended setting `init_pos` to 'random'.

    random_state
        `int`, `random_state` is the seed used by the random number generator
    a
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    b
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    copy
        Return a copy instead of writing to adata.
    neighbors_key
        If not specified, umap looks .uns['neighbors'] for neighbors settings
        and .obsp['connectivities'] for connectivities
        (default storage places for pp.neighbors).
        If specified, umap looks .uns[neighbors_key] for neighbors settings and
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.

        **X_umap** : `adata.obsm` field
            UMAP coordinates of data.
    """
    adata = adata.copy() if copy else adata

    if neighbors_key is None:
        neighbors_key = "neighbors"

    if neighbors_key not in adata.uns:
        raise ValueError(
            f'Did not find .uns["{neighbors_key}"]. Run `sc.pp.neighbors` first.'
        )

    neighbors = NeighborsView(adata, neighbors_key)

    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    else:
        a = a
        b = b
    adata.uns["umap"] = {"params": {"a": a, "b": b}}

    if random_state != 0:
        adata.uns["umap"]["params"]["random_state"] = random_state
    random_state = check_random_state(random_state)

    neigh_params = neighbors["params"]
    X = _choose_representation(
        adata,
        neigh_params.get("use_rep", None),
        neigh_params.get("n_pcs", None),
    )

    n_neighbors = neighbors["params"]["n_neighbors"]
    n_epochs = (
        500 if maxiter is None else maxiter
    )  # 0 is not a valid value for rapids, unlike original umap
    metric = neigh_params.get("metric", "euclidean")

    if isinstance(X, cp.ndarray):
        X_contiguous = cp.ascontiguousarray(X, dtype=np.float32)
    elif isinstance(X, sparse.spmatrix):
        X_contiguous = X
    else:
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)

    n_obs = adata.shape[0]
    if neigh_params.get("method") == "rapids":
        knn_dist = neighbors["distances"].data.reshape(n_obs, n_neighbors)
        knn_indices = neighbors["distances"].indices.reshape(n_obs, n_neighbors)
        pre_knn = (knn_indices, knn_dist)
    else:
        pre_knn = None

    if init_pos == "auto":
        init_pos = "spectral" if n_obs < 1000000 else "random"

    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        n_epochs=n_epochs,
        learning_rate=alpha,
        init=init_pos,
        min_dist=min_dist,
        spread=spread,
        negative_sample_rate=negative_sample_rate,
        a=a,
        b=b,
        random_state=random_state,
        output_type="numpy",
        precomputed_knn=pre_knn,
    )

    X_umap = umap.fit_transform(X_contiguous)
    adata.obsm["X_umap"] = X_umap
    return adata if copy else None
