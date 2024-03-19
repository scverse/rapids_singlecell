from __future__ import annotations

from typing import TYPE_CHECKING

from cuml.manifold import TSNE

from ._utils import _choose_representation

if TYPE_CHECKING:
    from anndata import AnnData


def tsne(
    adata: AnnData,
    n_pcs: int = None,
    *,
    use_rep: str = None,
    perplexity: int = 30,
    early_exaggeration: int = 12,
    learning_rate: int = 200,
    method: str = "barnes_hut",
    metric: str = "euclidean",
    copy: bool = False,
) -> AnnData | None:
    """
    Performs t-distributed stochastic neighborhood embedding (tSNE) using cuml library.

    Parameters
    ----------
        adata
            Annotated data matrix.
        n_pcs
            Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep
            Use the indicated representation. `'X'` or any key for `.obsm` is valid.
            If None, the representation is chosen automatically: For .n_vars < 50, .X
            is used, otherwise `'X_pca'` is used. If `'X_pca'` is not present, it's
            computed with default parameters or `n_pcs` if present.
        perplexity
            The perplexity is related to the number of nearest neighbors that is used
            in other manifold learning algorithms. Larger datasets usually require a larger
            perplexity. Consider selecting a value between 5 and 50. The choice is not extremely
            critical since t-SNE is quite insensitive to this parameter.
        early_exaggeration
            Controls how tight natural clusters in the original space are in the embedded space
            and how much space will be between them. For larger values, the space between natural
            clusters will be larger in the embedded space. Again, the choice of this parameter is
            not very critical. If the cost function increases during initial optimization, the early
            exaggeration factor or the learning rate might be too high.
        learning_rate
            Note that the R-package “Rtsne” and cuML uses a default of 200. The learning rate can be
            a critical parameter. It should be between 100 and 1000. If the cost function increases
            during initial optimization, the early exaggeration factor or the learning rate might
            be too high. If the cost function gets stuck in a bad local minimum increasing the
            learning rate helps sometimes.
        method
            'barnes_hut' and 'fft' are fast approximations. 'exact' is more accurate but slower.
        metric
            Distance metric to use. Supported distances are ['l1, 'cityblock', 'manhattan', 'euclidean',
            'l2', 'sqeuclidean', 'minkowski', 'chebyshev', 'cosine', 'correlation']
        copy
            Return a copy instead of writing to adata.

    Returns
    -------
        Depending on `copy`, returns or updates `adata` with the following fields.

            **X_tsne** : `np.ndarray` (`adata.obsm`, dtype `float`)
                tSNE coordinates of data.
    """

    adata = adata.copy() if copy else adata

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

    adata.obsm["X_tsne"] = TSNE(
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        method=method,
        metric=metric,
    ).fit_transform(X)
    adata.uns["tsne"] = {
        "params": {
            k: v
            for k, v in {
                "perplexity": perplexity,
                "early_exaggeration": early_exaggeration,
                "learning_rate": learning_rate,
                "method": method,
                "metric": metric,
                "use_rep": use_rep,
            }.items()
            if v is not None
        }
    }

    return adata if copy else None
