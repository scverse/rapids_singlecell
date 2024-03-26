from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy.sparse import linalg
from scipy.sparse import issparse

if TYPE_CHECKING:
    from anndata import AnnData


def diffmap(
    adata: AnnData,
    n_comps: int = 15,
    *,
    neighbors_key: str = None,
    sort: str = "decrease",
    density_normalize: bool = True,
) -> None:
    """
    Diffusion maps has been proposed for visualizing single-cell data.

    This is a reimplementation of scanpys function.

    The width ("sigma") of the connectivity kernel is implicitly determined by
    the number of neighbors used to compute the single-cell graph in
    :func:`scanpy.pp.neighbors` or :func:`~rapids_singlecell.pp.neighbors`.

    Parameters
    ----------
        adata
            Annotated data matrix.
        n_comps
            The number of dimensions of the representation.
        neighbors_key
            If not specified, diffmap looks at `.obsp['connectivities']` for neighbors connectivities
            If specified, diffmap looks at `.obsp['neighbors_key_ connectivities']` for neighbors connectivities
        sort
            Leave as is for the same behavior as sc.tl.diffmap
        density_normalize
            Leave as is for the same behavior as sc.tl.diffmap

    Returns
    -------
        updates `adata` with the following fields.

            `X_diffmap` : :class:`numpy.ndarray` (`adata.obsm`)
                Diffusion map representation of data, which is the right eigen basis of
                the transition matrix with eigenvectors as columns.
            `diffmap_evals` : :class:`numpy.ndarray` (`adata.uns`)
                Array of size (number of eigen vectors).
                Eigenvalues of transition matrix.
    """
    if neighbors_key:
        connectivities = adata.obsp[neighbors_key + "_connectivities"]
    else:
        connectivities = adata.obsp["connectivities"]
    if issparse(connectivities):
        W = sparse.csr_matrix(connectivities, dtype=cp.float32)
    else:
        W = cp.asarray(connectivities)
    if density_normalize:
        # q[i] is an estimate for the sampling density at point i
        # it's also the degree of the underlying graph
        q = cp.asarray(W.sum(axis=0))
        if not sparse.issparse(W):
            Q = cp.diag(1.0 / q)
        else:
            Q = sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
        K = Q @ W @ Q
    else:
        K = W
        # z[i] is the square root of the row sum of K
    z = cp.sqrt(cp.asarray(K.sum(axis=0)))
    if not sparse.issparse(K):
        Z = cp.diag(1.0 / z)
    else:
        Z = sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
    matrix = Z @ K @ Z
    if n_comps == 0:
        evals, evecs = linalg.eigsh(matrix)
    else:
        n_comps = min(matrix.shape[0] - 1, n_comps)
        # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
        ncv = None
        which = "LM" if sort == "decrease" else "SM"
        # it pays off to increase the stability with a bit more precision
        matrix = matrix.astype(cp.float64)
        evals, evecs = linalg.eigsh(matrix, k=n_comps, which=which, ncv=ncv)
        evals, evecs = evals.astype(cp.float32), evecs.astype(cp.float32)
    if sort == "decrease":
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
    adata.uns["diffmap_evals"] = evals.get()
    adata.obsm["X_diffmap"] = evecs.get()
