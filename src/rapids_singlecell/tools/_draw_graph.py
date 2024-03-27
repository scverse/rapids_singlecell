from __future__ import annotations

from typing import TYPE_CHECKING

import cudf
import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def draw_graph(
    adata: AnnData, *, init_pos: str | bool | None = None, max_iter: int = 500
) -> None:
    """
    Force-directed graph drawing with cugraph's implementation of Force Atlas 2.
    This is a reimplementation of scanpys function for GPU compute.

    Parameters
    ----------
        adata
            annData object with 'neighbors' field.

        init_pos
            `'paga'`/`True`, `None`/`False`, or any valid 2d-`.obsm` key.
            Use precomputed coordinates for initialization.
            If `False`/`None` (the default), initialize randomly.
        max_iter
            This controls the maximum number of levels/iterations of the
            Force Atlas algorithm. When specified the algorithm will terminate
            after no more than the specified number of iterations.
            No error occurs when the algorithm terminates in this manner.
            Good short-term quality can be achieved with 50-100 iterations.
            Above 1000 iterations is discouraged.

    Returns
    -------
        updates `adata` with the following fields.

            X_draw_graph_layout_fa : `adata.obsm`
                Coordinates of graph layout.
    """
    from cugraph import Graph
    from cugraph.layout import force_atlas2

    # Adjacency graph
    adjacency = adata.obsp["connectivities"]
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    g = Graph()
    if hasattr(g, "add_adj_list"):
        g.add_adj_list(offsets, indices, None)
    else:
        g.from_cudf_adjlist(offsets, indices, None)
    # Get Initial Positions
    if init_pos in adata.obsm.keys():
        init_coords = adata.obsm[init_pos]
    elif init_pos == "paga" or init_pos:
        if "paga" in adata.uns and "pos" in adata.uns["paga"]:
            groups = adata.obs[adata.uns["paga"]["groups"]]
            pos = adata.uns["paga"]["pos"]
            connectivities_coarse = adata.uns["paga"]["connectivities"]
            init_coords = np.ones((adjacency.shape[0], 2))
            for i, group_pos in enumerate(pos):
                subset = (groups == groups.cat.categories[i]).values
                neighbors = connectivities_coarse[i].nonzero()
                if len(neighbors[1]) > 0:
                    connectivities = connectivities_coarse[i][neighbors]
                    nearest_neighbor = neighbors[1][np.argmax(connectivities)]
                    noise = np.random.random((len(subset[subset]), 2))
                    dist = pos[i] - pos[nearest_neighbor]
                    noise = noise * dist
                    init_coords[subset] = group_pos - 0.5 * dist + noise
                else:
                    init_coords[subset] = group_pos
        else:
            raise ValueError(
                "Plot PAGA first, so that adata.uns['paga']" "with key 'pos'."
            )

    else:
        init_coords = None

    if init_coords is not None:
        x, y = np.hsplit(init_coords, init_coords.shape[1])
        inital_df = cudf.DataFrame({"x": x.ravel(), "y": y.ravel()})
        inital_df["vertex"] = inital_df.index
    else:
        inital_df = None
    # Run cugraphs Force Atlas 2
    positions = force_atlas2(
        input_graph=g,
        pos_list=inital_df,
        max_iter=max_iter,
        outbound_attraction_distribution=False,
        lin_log_mode=False,
        edge_weight_influence=1.0,
        # Performance
        jitter_tolerance=1.0,  # Tolerance
        barnes_hut_optimize=True,
        barnes_hut_theta=1.2,
        # Tuning
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
    )
    positions = cp.vstack((positions["x"].to_cupy(), positions["y"].to_cupy())).T
    layout = "fa"
    adata.uns["draw_graph"] = {}
    adata.uns["draw_graph"]["params"] = {"layout": layout, "random_state": 0}
    key_added = f"X_draw_graph_{layout}"
    adata.obsm[key_added] = positions.get()  # Format output
