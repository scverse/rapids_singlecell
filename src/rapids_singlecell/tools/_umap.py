from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from cuml.manifold.umap import UMAP
from cuml.manifold.umap_utils import find_ab_params
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
    random_state: int = 0,
    a: float | None = None,
    b: float | None = None,
    key_added: str | None = None,
    neighbors_key: str | None = None,
    copy: bool = False,
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
    key_added
        If not specified, the embedding is stored as
        :attr:`~anndata.AnnData.obsm`\\ `['X_umap']` and the the parameters in
        :attr:`~anndata.AnnData.uns`\\ `['umap']`.
        If specified, the embedding is stored as
        :attr:`~anndata.AnnData.obsm`\\ ``[key_added]`` and the the parameters in
        :attr:`~anndata.AnnData.uns`\\ ``[key_added]``.
    neighbors_key
        If not specified, umap looks .uns['neighbors'] for neighbors settings
        and .obsp['connectivities'] for connectivities
        (default storage places for pp.neighbors).
        If specified, umap looks .uns[neighbors_key] for neighbors settings and
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.

        `adata.obsm['X_umap' | key_added]` : :class:`~numpy.ndarray` (dtype `float`)
            UMAP coordinates of data.
        `adata.uns['umap' | key_added]['params']` : :class:`dict`
            UMAP parameters `a`, `b`, and `random_state` (if specified).
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

    # store params for adata.uns
    stored_params = {
        "a": a,
        "b": b,
        **({"random_state": random_state} if random_state != 0 else {}),
    }

    random_state = check_random_state(random_state)

    neigh_params = neighbors["params"]
    X = _choose_representation(
        adata,
        neigh_params.get("use_rep", None),
        neigh_params.get("n_pcs", None),
    )

    n_epochs = (
        500 if maxiter is None else maxiter
    )  # 0 is not a valid value for rapids, unlike original umap

    n_obs = adata.shape[0]
    n_neighbors = neigh_params["n_neighbors"]
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
        metric=neigh_params.get("metric", "euclidean"),
        metric_kwds=neigh_params.get("metric_kwds", None),
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

    key_obsm, key_uns = ("X_umap", "umap") if key_added is None else [key_added] * 2
    adata.obsm[key_obsm] = umap.fit_transform(X)

    adata.uns[key_uns] = {"params": stored_params}
    return adata if copy else None
