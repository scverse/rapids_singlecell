from anndata import AnnData
from cuml.manifold import TSNE

def tsne(adata: AnnData, 
         n_pcs:int = None,
         use_rep:str= None,
         perplexity:int = 30, 
         early_exaggeration:int = 12,
         learning_rate:int =200,
         method:str = "barnes_hut",
         metric:str = "euclidean",
         ):
    """
    Performs t-distributed stochastic neighborhood embedding (tSNE) using cuML libraray. Variable description adapted from scanpy and default are the same
    
    Parameters
    ---------
    adata : AnnData
        Annotated data matrix.
    n_pcs: int
        use this many PCs
    use_rep:str
        use this obsm keys (defaults to `X_pca`)
    perplexity: float (default: 30)
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE is quite insensitive to this parameter.
    early_exaggeration : float (default:12)
        Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high.
    learning_rate : float (default: 200)
        Note that the R-package “Rtsne” and cuML uses a default of 200. The learning rate can be a critical parameter. It should be between 100 and 1000. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high. If the cost function gets stuck in a bad local minimum increasing the learning rate helps sometimes.
    method: str (default: 'barnes_hut')
        'barnes_hut' and 'fft' are fast approximations. 'exact' is more accurate but slower.
    metric: str (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'minkowski', 'chebyshev', 'cosine', 'correlation']
    """
    if use_rep == None:
        data = adata.obsm["X_pca"]
    else:
        data = adata.obsm[use_rep]
    if n_pcs is not None:
        data = data[:,:n_pcs]
    adata.obsm['X_tsne'] = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration,learning_rate=learning_rate,method=method,metric = metric).fit_transform(data)
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
