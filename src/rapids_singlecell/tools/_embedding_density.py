from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def embedding_density(
    adata: AnnData,
    basis: str = "umap",
    *,
    groupby: str | None = None,
    key_added: str | None = None,
    batchsize: int = 10000,
    components: str | Sequence[str] = None,
) -> None:
    """\
    Calculate the density of cells in an embedding (per condition).
    Gaussian kernel density estimation is used to calculate the density of
    cells in an embedded space. This can be performed per category over a
    categorical cell annotation. The cell density can be plotted using the
    `pl.embedding_density` function.
    Note that density values are scaled to be between 0 and 1. Thus, the
    density value at each cell is only comparable to densities in
    the same category.
    This function was written by Sophie Tritschler and implemented into
    Scanpy by Malte Luecken.
    This function uses cuML's KernelDensity. It returns log Likelihood as does
    sklearn's implementation. scipy.stats implementation, used
    in scanpy, returns PDF.

    Parameters
    ----------
        adata
            The annotated data matrix.
        basis
            The embedding over which the density will be calculated. This embedded
            representation should be found in `adata.obsm['X_[basis]']``.
        groupby
            Key for categorical observation/cell annotation for which densities
            are calculated per category.
        key_added
            Name of the `.obs` covariate that will be added with the density
            estimates.
        batchsize
            Number of cells that should be processed together.
        components
            The embedding dimensions over which the density should be calculated.
            This is limited to two components.

    Returns
    -------
    Updates `adata.obs` with an additional field specified by the `key_added` \
    parameter. This parameter defaults to `[basis]_density_[groupby]`, where \
    `[basis]` is one of `umap`, `diffmap`, `pca`, `tsne`, or `draw_graph_fa` \
    and `[groupby]` denotes the parameter input.

    Updates `adata.uns` with an additional field `[key_added]_params`.
    """
    # to ensure that newly created covariates are categorical
    # to test for category numbers
    adata._sanitize()
    # Test user inputs
    basis = basis.lower()

    if basis == "fa":
        basis = "draw_graph_fa"

    if f"X_{basis}" not in adata.obsm_keys():
        raise ValueError(
            "Cannot find the embedded representation "
            f"`adata.obsm['X_{basis}']`. Compute the embedding first."
        )

    if components is None:
        components = "1,2"
    if isinstance(components, str):
        components = components.split(",")
    components = np.array(components).astype(int) - 1

    if len(components) != 2:
        raise ValueError("Please specify exactly 2 components, or `None`.")

    if basis == "diffmap":
        components += 1

    if groupby is not None:
        if groupby not in adata.obs:
            raise ValueError(f"Could not find {groupby!r} `.obs` column.")

        if adata.obs[groupby].dtype.name != "category":
            raise ValueError(f"{groupby!r} column does not contain categorical data")

    # Define new covariate name
    if key_added is not None:
        density_covariate = key_added
    elif groupby is not None:
        density_covariate = f"{basis}_density_{groupby}"
    else:
        density_covariate = f"{basis}_density"

    # Calculate the densities over each category in the groupby column
    if groupby is not None:
        categories = adata.obs[groupby].cat.categories

        density_values = np.zeros(adata.n_obs)

        for cat in categories:
            cat_mask = adata.obs[groupby] == cat
            embed_x = adata.obsm[f"X_{basis}"][cat_mask, components[0]]
            embed_y = adata.obsm[f"X_{basis}"][cat_mask, components[1]]

            dens_embed = _calc_density(cp.array(embed_x), cp.array(embed_y), batchsize)
            density_values[cat_mask] = dens_embed

        adata.obs[density_covariate] = density_values
    else:  # if groupby is None
        # Calculate the density over the whole embedding without subsetting
        embed_x = adata.obsm[f"X_{basis}"][:, components[0]]
        embed_y = adata.obsm[f"X_{basis}"][:, components[1]]

        adata.obs[density_covariate] = _calc_density(
            cp.array(embed_x), cp.array(embed_y), batchsize
        )

    # Reduce diffmap components for labeling
    # Note: plot_scatter takes care of correcting diffmap components
    #       for plotting automatically
    if basis != "diffmap":
        components += 1

    adata.uns[f"{density_covariate}_params"] = {
        "covariate": groupby,
        "components": components.tolist(),
    }


def _calc_density(x: cp.ndarray, y: cp.ndarray, batchsize: int):
    """\
    Calculates the density of points in 2 dimensions.
    """
    from cuml.neighbors import KernelDensity

    # Calculate the point density
    xy = cp.vstack([x, y]).T
    bandwidth = cp.power(xy.shape[0], (-1.0 / (xy.shape[1] + 4)))
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(xy)
    z = cp.zeros(xy.shape[0])
    n_batches = math.ceil(xy.shape[0] / batchsize)
    for batch in range(n_batches):
        start_idx = batch * batchsize
        stop_idx = min(batch * batchsize + batchsize, xy.shape[0])
        z[start_idx:stop_idx] = kde.score_samples(xy[start_idx:stop_idx, :])
    min_z = cp.min(z)
    max_z = cp.max(z)

    # Scale between 0 and 1
    scaled_z = (z - min_z) / (max_z - min_z)

    return scaled_z.get()
