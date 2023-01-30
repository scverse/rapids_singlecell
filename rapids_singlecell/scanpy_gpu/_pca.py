from cuml.decomposition import PCA, TruncatedSVD
from anndata import AnnData
from typing import Optional

from scipy.sparse import issparse
import warnings
import math
import numpy as np

def pca(adata: AnnData, 
        layer = None, 
        n_comps = 50,
        zero_center = True,
        use_highly_variable: Optional[bool] = None,
        chunked = False,
        chunk_size = None):
    """
    Performs PCA using the cuML decomposition function for the `AnnData` object.
    
    Parameters
    ----------
    adata : annData object
    
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.

    n_comps: int (default: 50)
        Number of principal components to compute. Defaults to 50
    
    zero_center
        If `True`, compute standard PCA from covariance matrix.
        If `False`, omit zero-centering variables
    
    use_highly_variable
        Whether to use highly variable genes only, stored in
        `.var['highly_variable']`.
        By default uses them if they have been determined beforehand.
        
    chunked
        If `True`, perform an incremental PCA on segments of `chunk_size`.
        The incremental PCA automatically zero centers and ignores settings of
        `random_seed` and `svd_solver`. If `False`, perform a full PCA.
        
    chunk_size
        Number of observations to include in each chunk.
        Required if `chunked=True` was passed.
    
    Returns
    
    else adds fields to `adata`:

    `.obsm['X_pca']`
         PCA representation of data.  
    `.varm['PCs']`
         The principal components containing the loadings.
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

    if issparse(X) and zero_center:
        warnings.warn(
            "Your Countmatrix seems to be sparse, this can lead to a massive performance penalty.",
            UserWarning,
        )
        
    if chunked:
        from cuml.decomposition import IncrementalPCA

        X_pca = np.zeros((X.shape[0], n_comps), X.dtype)

        pca_func = IncrementalPCA(n_components=n_comps, output_type="numpy",batch_size=chunk_size)
        pca_func.fit(X)

        n_batches = math.ceil(X.shape[0] / chunk_size)
        for batch in range(n_batches):
            start_idx = batch * chunk_size
            stop_idx = min(batch * chunk_size + chunk_size, X.shape[0])
            chunk = X[start_idx:stop_idx,:]
            chunk = chunk.toarray() if issparse(chunk) else chunk
            X_pca[start_idx:stop_idx] = pca_func.transform(chunk)
    elif zero_center:
        pca_func = PCA(n_components=n_comps, output_type="numpy")
        X_pca= pca_func.fit_transform(X)
    
    elif not zero_center:
        pca_func = TruncatedSVD(n_components=n_comps, output_type="numpy")
        X_pca = pca_func.fit_transform(X)
    adata.obsm["X_pca"] = X_pca
    adata.uns['pca'] ={'variance':pca_func.explained_variance_, 'variance_ratio':pca_func.explained_variance_ratio_}
    if use_highly_variable:
        adata.varm['PCs'] = np.zeros(shape=(adata.n_vars, n_comps))
        adata.varm['PCs'][adata.var['highly_variable']] = pca_func.components_.T
    else:
        adata.varm['PCs'] = pca_func.components_.T
