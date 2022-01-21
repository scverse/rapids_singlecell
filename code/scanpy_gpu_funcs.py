#
# created by Severin Dicks (IBSM, Freiburg)
#
#

import cupy as cp
import cudf
import cugraph
import anndata
import os

import numpy as np
import pandas as pd
import scipy
import math
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt


from cuml.manifold import TSNE
from cuml.cluster import KMeans
from cuml.decomposition import PCA
from cuml.linear_model import LinearRegression



    
def select_groups(labels, groups_order_subset='all'):
    groups_order = labels.cat.categories
    groups_masks = np.zeros(
        (len(labels.cat.categories), len(labels.cat.codes)), dtype=bool
    )
    for iname, name in enumerate(labels.cat.categories):
        # if the name is not found, fallback to index retrieval
        if labels.cat.categories[iname] in labels.cat.codes:
            mask = labels.cat.categories[iname] == labels.cat.codes
        else:
            mask = iname == labels.cat.codes
        groups_masks[iname] = mask.values
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != 'all':
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(
                np.where(name == labels.cat.categories)[0]
            )
        if len(groups_ids) == 0:
            # fallback to index retrieval
            groups_ids = np.where(
                np.in1d(
                    np.arange(len(labels.cat.categories)).astype(str),
                    np.array(groups_order_subset),
                )
            )[0]
            
        groups_ids = [groups_id.item() for groups_id in groups_ids]
        groups_masks = groups_masks[groups_ids]
        groups_order_subset = labels.cat.categories[groups_ids].to_numpy()
    else:
        groups_order_subset = groups_order.to_numpy()
    return groups_order_subset, groups_masks


def rank_genes_groups_logreg(
    adata,
    groupby,  
    groups="all",
    reference='rest',
    n_genes = None,
    use_raw = False,
    **kwds,
):

    """
    Rank genes for characterizing groups.

    Parameters
    ----------

    adata : adata object

    labels : cudf.Series of size (n_cells,)
        Observations groupings to consider

    var_names : cudf.Series of size (n_genes,)
        Names of genes in X

    groups : Iterable[str] (default: 'all')
        Subset of groups, e.g. ['g1', 'g2', 'g3'], to which comparison
        shall be restricted, or 'all' (default), for all groups.

    reference : str (default: 'rest')
        If 'rest', compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.

    n_genes : int (default: 100)
        The number of genes that appear in the returned tables.
    """

    #### Wherever we see "adata.obs[groupby], we should just replace w/ the groups"
        
    # for clarity, rename variable
    if groups == 'all' or groups == None:
        groups_order = 'all'
    elif isinstance(groups, (str, int)):
        raise ValueError('Specify a sequence of groups')
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != 'rest' and reference not in set(groups_order):
            groups_order += [reference]
    labels = pd.Series(adata.obs[groupby]).reset_index(drop="True")
    if (
        reference != 'rest'
        and reference not in set(labels.cat.categories)
    ):
        cats = labels.cat.categories.tolist()
        raise ValueError(
            f'reference = {reference} needs to be one of groupby = {cats}.'
        )

    groups_order, groups_masks = select_groups(labels, groups_order)
    
    original_reference = reference
    
    if use_raw == False:
        X = adata.X
        var_names = adata.var_names
    else:
        X = adata.raw.X
        var_names = adata.raw.var_names
    
    # for clarity, rename variable
    n_genes_user = n_genes
    # make sure indices are not OoB in case there are less genes than n_genes
    if n_genes == None or n_genes_user > X.shape[1]:
        n_genes_user = X.shape[1]
    # in the following, n_genes is simply another name for the total number of genes


    n_groups = groups_masks.shape[0]
    ns = np.zeros(n_groups, dtype=int)
    for imask, mask in enumerate(groups_masks):
        ns[imask] = np.where(mask)[0].size
    if reference != 'rest':
        ireference = np.where(groups_order == reference)[0][0]
    reference_indices = cp.arange(X.shape[1], dtype=int)

    rankings_gene_scores = []
    rankings_gene_names = []

    # Perform LogReg
        
    # if reference is not set, then the groups listed will be compared to the rest
    # if reference is set, then the groups listed will be compared only to the other groups listed
    refname = reference
    from cuml.linear_model import LogisticRegression
    reference = groups_order[0]
    if len(groups) == 1:
        raise Exception('Cannot perform logistic regression on a single cluster.')
        
    grouping_mask = labels.isin(pd.Series(groups_order))
    grouping = labels.loc[grouping_mask]
    
    X = X[grouping_mask.values, :]
    # Indexing with a series causes issues, possibly segfault
        
    clf = LogisticRegression(**kwds)
    clf.fit(X, grouping.cat.codes.to_numpy().astype('float32'))
    scores_all = cp.array(clf.coef_).T
    
    for igroup, group in enumerate(groups_order):
        if len(groups_order) <= 2:  # binary logistic regression
            scores = scores_all[0]
        else:
            scores = scores_all[igroup]
        
        partition = cp.argpartition(scores, -n_genes_user)[-n_genes_user:]
        partial_indices = cp.argsort(scores[partition])[::-1]
        global_indices = reference_indices[partition][partial_indices]
        rankings_gene_scores.append(scores[global_indices].get())  
        rankings_gene_names.append(var_names[global_indices.get()])
        if len(groups_order) <= 2:
            break

    groups_order_save = [str(g) for g in groups_order]
    if (len(groups) == 2):
        groups_order_save = [g for g in groups_order if g != reference]
            
    scores = np.rec.fromarrays(
        [n for n in rankings_gene_scores],
        dtype=[(rn, 'float32') for rn in groups_order_save],
    )
    
    names = np.rec.fromarrays(
        [n for n in rankings_gene_names],
        dtype=[(rn, 'U50') for rn in groups_order_save],
    )
    adata.uns["rank_genes_groups"] = {}
    adata.uns["rank_genes_groups"]["params"] = dict(groupby=groupby,method="logreg", reference=refname, use_raw=use_raw)
    adata.uns["rank_genes_groups"]['scores'] = scores
    adata.uns["rank_genes_groups"]['names'] = names



def leiden(adata, resolution=1.0):
    """
    Performs Leiden Clustering using cuGraph
    Parameters
    ----------
    adata : annData object with 'neighbors' field.
       
    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    """
    # Adjacency graph
    adjacency = adata.obsp["connectivities"]
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    g = cugraph.Graph()
    if hasattr(g, 'add_adj_list'):
        g.add_adj_list(offsets, indices, None)
    else:
        g.from_cudf_adjlist(offsets, indices, None)
    
    # Cluster
    leiden_parts, _ = cugraph.leiden(g,resolution = resolution)
    
    # Format output
    clusters = leiden_parts.to_pandas().sort_values('vertex')[['partition']].to_numpy().ravel()
    clusters = pd.Categorical(clusters.astype(str))
    
    adata.obs['leiden'] = clusters
    
def louvain(adata, resolution=1.0):
    """
    Performs Louvain Clustering using cuGraph
    Parameters
    ----------
    adata : annData object with 'neighbors' field.
       
    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    """
    # Adjacency graph
    adjacency = adata.obsp["connectivities"]
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    g = cugraph.Graph()
    if hasattr(g, 'add_adj_list'):
        g.add_adj_list(offsets, indices, None)
    else:
        g.from_cudf_adjlist(offsets, indices, None)
    
    # Cluster
    louvain_parts, _ = cugraph.louvain(g,resolution = resolution)
    
    # Format output
    clusters = louvain_parts.to_pandas().sort_values('vertex')[['partition']].to_numpy().ravel()
    clusters = pd.Categorical(clusters.astype(str))
    
    adata.obs['louvain'] = clusters 
    
def kmeans(adata, n_clusters =8, random_state= 42):
    """
    KMeans is a basic but powerful clustering method which is optimized via
Expectation Maximization. 

    Parameters
    ----------
    adata: adata object with `.obsm['X_pca']`
    
    n_clusters: int (default:8)
        Number of clusters to compute
        
    random_state: float (default: 42)
        if you want results to be the same when you restart Python, select a
    state.
    
    """

    
    kmeans_out = KMeans(n_clusters=n_clusters, random_state=random_state).fit(adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans_out.labels_.astype(str)

def pca(adata, n_comps = 50):
    """
    Performs PCA using the cuML decomposition function
    
    Parameters
    ----------
    adata : annData object
    
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
    pca_func = PCA(n_components=n_comps, output_type="numpy")
    adata.obsm["X_pca"] = pca_func.fit_transform(adata.X)
    adata.uns['pca'] ={'variance':pca_func.explained_variance_, 'variance_ratio':pca_func.explained_variance_ratio_}
    
    
def tsne(adata, n_pcs,perplexity = 30, early_exaggeration = 12,learning_rate =1000):
    """
    Performs t-distributed stochastic neighborhood embedding (tSNE) using cuML libraray. Variable description adapted from scanpy and default are the same
    
    Parameters
    ---------
    adata: adata object with `.obsm['X_pca']`
    
    n_pcs: int
        use this many PCs
    
    perplexity: float (default: 30)
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE is quite insensitive to this parameter.
    
    early_exaggeration : float (default:12)
        Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high.
    
    learning_rate : float (default:1000)
        Note that the R-package “Rtsne” and cuML uses a default of 200. The learning rate can be a critical parameter. It should be between 100 and 1000. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high. If the cost function gets stuck in a bad local minimum increasing the learning rate helps sometimes.


    """
    
    adata.obsm['X_tsne'] = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration,learning_rate=learning_rate).fit_transform(adata.obsm["X_pca"][:,:n_pcs])

def diffmap(adata, n_comps=15, neighbors_key = None, sort = 'decrease',density_normalize = True):
    """
    Diffusion maps has been proposed for visualizing single-cell data.
    
    This is a reimplementation of scanpys function.
    
    The width ("sigma") of the connectivity kernel is implicitly determined by
    the number of neighbors used to compute the single-cell graph in
    :func:`~scanpy.pp.neighbors`. 
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_comps : int, optional (default: 15)
        The number of dimensions of the representation.
    neighbors_key : typing.Union[str, NoneType], optional (default: None)
        If not specified, diffmap looks at .obsp['connectivities'] for neighbors connectivities
        If specified, diffmap looks at .obsp['neighbors_key_ connectivities'] for neighbors connectivities
    sort: string (default:'decrease')
        Leave as is for the same behavior as sc.tl.diffmap
    density_normalize: boolean(default: True)
        Leave as is for the same behavior as sc.tl.diffmap
    
    Returns
    ----------
    updates `adata` with the following fields.

    `X_diffmap` : :class:`numpy.ndarray` (`adata.obsm`)
        Diffusion map representation of data, which is the right eigen basis of
        the transition matrix with eigenvectors as columns.
    `diffmap_evals` : :class:`numpy.ndarray` (`adata.uns`)
        Array of size (number of eigen vectors).
        Eigenvalues of transition matrix.
    """
    from scipy.sparse import issparse
    import cupyx.scipy.sparse.linalg
    import cupyx.scipy.sparse
    import cupyx as cpx
    if neighbors_key:
        connectivities = adata.obsp[neighbors_key+"_connectivities"]
    else:
        connectivities = adata.obsp["connectivities"]
    if issparse(connectivities):
        W = cp.sparse.csr_matrix(connectivities, dtype=cp.float32)
    else:
        W = cp.asarray(connectivities)
    if density_normalize:
            # q[i] is an estimate for the sampling density at point i
            # it's also the degree of the underlying graph
            q = cp.asarray(W.sum(axis=0))
            if not cpx.scipy.sparse.issparse(W):
                Q = cp.diag(1.0 / q)
            else:
                Q = cpx.scipy.sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
            K = Q @ W @ Q
    else:
        K = W
            # z[i] is the square root of the row sum of K
    z = cp.sqrt(cp.asarray(K.sum(axis=0)))
    if not cpx.scipy.sparse.issparse(K):
        Z = cp.diag(1.0 / z)
    else:
        Z = cpx.scipy.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
    matrix = Z @ K @ Z
    if n_comps == 0:
                evals, evecs = cpx.scipy.sparse.linalg.eigsh(matrix)
    else:
        n_comps = min(matrix.shape[0] - 1, n_comps)
        # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
        ncv = None
        which = 'LM' if sort == 'decrease' else 'SM'
        # it pays off to increase the stability with a bit more precision
        matrix = matrix.astype(cp.float64)
        evals, evecs = cpx.scipy.sparse.linalg.eigsh(
            matrix, k=n_comps, which=which, ncv=ncv
        )
        evals, evecs = evals.astype(cp.float32), evecs.astype(cp.float32)
    if sort == 'decrease':
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
    adata.uns["diffmap_evals"] = evals.get()
    adata.obsm["X_diffmap"] = evecs.get()
    
def plt_scatter(cudata, x, y, save = None, show =True, dpi =300):
    """
    Violin plot.
    Wraps :func:`seaborn.scaterplot` for :class:`~cunnData.cunnData`. This plotting function so far is really basic and doesnt include all the features form sc.pl.scatter.
    
    Parameters
    ---------
    cudata:
        cunnData object
    
    x:
        Keys for accessing variables of fields of `.obs`.
    
    y:
        Keys for accessing variables of fields of `.obs`.

    
    save: str default(None (no plot will be saved))
        file name to save plot as in ./figures
        
    show: boolean (default: True)
        if you want to display the plot
    
    dpi: int (default: 300)
        The resolution in dots per inch for save
    
    Returns
    ------
    nothing
    
    """
    fig,ax = plt.subplots()
    sns.scatterplot(data=cudata.obs, x=x, y=y,s=2, color='grey', edgecolor='grey')
    if save:
        os.makedirs("./figures/",exist_ok=True)
        fig_path = "./figures/"+save
        plt.savefig(fig_path, dpi=dpi ,bbox_inches = 'tight')
    if show is False:
        plt.close()

        
def plt_violin(cudata, key, groupby=None, size =1, save = None, show =True, dpi =300):
    """
    Violin plot.
    Wraps :func:`seaborn.violinplot` for :class:`~cunnData.cunnData`. This plotting function so far is really basic and doesnt include all the features form sc.pl.violin.
    
    Parameters
    ---------
    cudata:
        cunnData object
    
    key:
        Keys for accessing variables of fields of `.obs`.
    
    groupby:
        The key of the observation grouping to consider.(e.g batches)
    
    size:
        pt_size for stripplot if 0 no strip plot will be shown.
    
    save: str default(None (no plot will be saved))
        file name to save plot as in ./figures
        
    show: boolean (default: True)
        if you want to display the plot
    
    dpi: int (default: 300)
        The resolution in dots per inch for save
    
    Returns
    ------
    nothing
    
    """
    fig,ax = plt.subplots()
    ax = sns.violinplot(data=cudata.obs, y=key,scale='width',x= groupby, inner = None)
    if size:
        ax = sns.stripplot(data=cudata.obs, y=key,x= groupby, color='k', size= size, dodge = True, jitter = True)
    if save:
        os.makedirs("./figures/",exist_ok=True)
        fig_path = "./figures/"+save
        plt.savefig(fig_path, dpi=dpi ,bbox_inches = 'tight')
    if show is False:
        plt.close()
