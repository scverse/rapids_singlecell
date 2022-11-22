from cuml.decomposition import PCA
from anndata import AnnData
from typing import Optional

from scipy.sparse import issparse
import warnings


def pca(adata: AnnData, 
        layer = None, 
        n_comps = 50,
        use_highly_variable: Optional[bool] = None):
    """
    Performs PCA using the cuML decomposition function
    
    Parameters
    ----------
    adata : annData object
    
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.

    n_comps: int (default: 50)
        Number of principal components to compute. Defaults to 50
    
    Returns
    
    else adds fields to `adata`:

    `.obsm['X_pca']`
         PCA representation of data.  
    `.uns['pca']['variance_ratio']`
         Ratio of explained variance.
    `.uns['pca']['variance']`
         Explained variance, equivalent to the eigenvalues of the
         covariance matrix.
    """

    if use_highly_variable is True and 'highly_variable' not in adata.var.keys():
        raise ValueError(
            'Did not find adata.var[\'highly_variable\']. '
            'Either your data already only consists of highly-variable genes '
            'or consider running `highly_variable_genes` first.'
        )

    X = adata.layers[layer] if layer is not None else adata.X

    if use_highly_variable is None:
        use_highly_variable = True if 'highly_variable' in adata.var.keys() else False

    if use_highly_variable:
        X = X[:, adata.var['highly_variable']]

    if issparse(X):
        warnings.warn(
            "Your Countmatrix seems to be sparse, this can lead to a massive performance penalty.",
            UserWarning,
        )
    pca_func = PCA(n_components=n_comps, output_type="numpy")
    adata.obsm["X_pca"] = pca_func.fit_transform(X)
    adata.uns['pca'] ={'variance':pca_func.explained_variance_, 'variance_ratio':pca_func.explained_variance_ratio_}