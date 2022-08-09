#
# created by Severin Dicks (IBSM, Freiburg)
#
#

import cupy as cp
import cudf
import cugraph
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted

from scanpy._utils import sanitize_anndata
from anndata import AnnData
from typing import Union, Optional, Sequence, Literal

from cuml.manifold import TSNE
from cuml.cluster import KMeans
from cuml.decomposition import PCA



def _select_groups(labels, groups_order_subset='all'):
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
        if len(groups_ids) >2:    
            groups_ids = np.sort(groups_ids)
        groups_masks = groups_masks[groups_ids]
        groups_order_subset = labels.cat.categories[groups_ids].to_numpy()
    else:
        groups_order_subset = groups_order.to_numpy()
    return groups_order_subset, groups_masks


def rank_genes_groups_logreg(
    adata: AnnData,
    groupby,  
    groups="all",
    reference='rest',
    n_genes = None,
    use_raw = None,
    layer= None,
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

    groups_order, groups_masks = _select_groups(labels, groups_order)
    
    if layer and use_raw== True:
        raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
    elif layer:
        X = adata.layers[layer]
        var_names = adata.var_names
    elif use_raw == None and adata.raw:
        print("defaulting to using `.raw`")
        X = adata.raw.X
        var_names = adata.raw.var_names
    elif use_raw == True:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names
    
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
        reference = np.where(groups_order == reference)[0][0]
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
        
    grouping_logreg = grouping.cat.codes.to_numpy().astype('float32')
    uniques = np.unique(grouping_logreg)
    for idx, cat in enumerate(uniques):
        grouping_logreg[np.where(grouping_logreg == cat)] = idx

    
    clf = LogisticRegression(**kwds)
    clf.fit(X, grouping_logreg)
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
        groups_order_save = [groups_order_save[0]]

    
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

def leiden(adata: AnnData, 
           resolution=1.0,
           key_added: str = 'leiden'):
    """
    Performs Leiden Clustering using cuGraph
    Parameters
    ----------
    adata : annData object with 'neighbors' field.
       
    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    
    key_added
        `adata.obs` key under which to add the cluster labels.
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
    groups = leiden_parts.to_pandas().sort_values('vertex')[['partition']].to_numpy().ravel()
   
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=natsorted(map(str, np.unique(groups))),
    )

    
def louvain(adata: AnnData, 
            resolution=1.0,
            key_added: str = 'louvain'):
    """
    Performs Louvain Clustering using cuGraph
    Parameters
    ----------
    adata : annData object with 'neighbors' field.
       
    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    
    key_added
        `adata.obs` key under which to add the cluster labels.
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
    groups = louvain_parts.to_pandas().sort_values('vertex')[['partition']].to_numpy().ravel()

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=natsorted(map(str, np.unique(groups))),
    )
    
def kmeans(adata: AnnData, 
           n_clusters =8,
           random_state= 42):
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

def pca(adata: AnnData, 
        layer = None, 
        n_comps = 50):
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
    X = adata.layers[layer] if layer is not None else adata.X
    pca_func = PCA(n_components=n_comps, output_type="numpy")
    adata.obsm["X_pca"] = pca_func.fit_transform(X)
    adata.uns['pca'] ={'variance':pca_func.explained_variance_, 'variance_ratio':pca_func.explained_variance_ratio_}
    
    
def tsne(adata: AnnData, 
         n_pcs:int = None,
         use_rep:str= None,
         perplexity:int = 30, 
         early_exaggeration:int = 12,
         learning_rate:int =1000):
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
    learning_rate : float (default:1000)
        Note that the R-package “Rtsne” and cuML uses a default of 200. The learning rate can be a critical parameter. It should be between 100 and 1000. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high. If the cost function gets stuck in a bad local minimum increasing the learning rate helps sometimes.
    """
    if use_rep == None:
        data = adata.obsm["X_pca"]
    else:
        data = adata.obsm[use_rep]
    if n_pcs is not None:
        data = data[:,:n_pcs]
    adata.obsm['X_tsne'] = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration,learning_rate=learning_rate).fit_transform(data)
    
def mde(
    adata: AnnData,
    device: Optional[Literal["cpu", "cuda"]] = None,
    n_neighbors: int = 15,
    n_pcs = None,
    use_rep = None,
    **kwargs,
) -> None:
    """
    Util to run :func:`pymde.preserve_neighbors` for visualization of single cell embeddings.
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    device : str
        Whether to run on cpu or gpu ("cuda"). If None, tries to run on gpu if available.
    n_neighbors: int
        use this many neighbors  
    n_pcs: int
        use this many PCs    
    use_rep:str
        use this obsm keys (defaults to `X_pca`)
    kwargs
        Keyword args to :func:`pymde.preserve_neighbors`
    Returns
    -------
    The pymde embedding, defaults to two dimensions.
    Notes
    -----
    This function adapted from scvi-tools.
    The appropriateness of use of visualization of high-dimensional spaces in single-
    cell omics remains an open research questions. See:
    Chari, Tara, Joeyta Banerjee, and Lior Pachter. "The specious art of single-cell genomics." bioRxiv (2021).
    If you use this function in your research please cite:
    Agrawal, Akshay, Alnur Ali, and Stephen Boyd. "Minimum-distortion embedding." arXiv preprint arXiv:2103.02559 (2021).
    """
    import torch
    try:
        import pymde
    except ImportError:
        raise ImportError("Please install pymde package via `pip install pymde`")
        
    if use_rep == None:
        data = adata.obsm["X_pca"]
    else:
        data = adata.obsm[use_rep]
        
    if isinstance(data, pd.DataFrame):
        data = data.values
    if n_pcs is not None:
        data = data[:,:n_pcs]
    
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    _kwargs = dict(
        embedding_dim=2,
        constraint=pymde.Standardized(),
        repulsive_fraction=0.7,
        verbose=False,
        device=device,
        n_neighbors=n_neighbors,
    )
    _kwargs.update(kwargs)

    emb = pymde.preserve_neighbors(data, **_kwargs).embed(verbose=_kwargs["verbose"])

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
        torch.cuda.empty_cache()

    adata.obsm["X_mde"] = emb    

def diffmap(adata: AnnData, 
            n_comps=15, 
            neighbors_key = None, 
            sort = 'decrease',
            density_normalize = True):
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

def draw_graph(adata: AnnData,
               init_pos = None,
               max_iter = 500):
    """
    Force-directed graph drawing with cugraph's implementation of Force Atlas 2.
    This is a reimplementation of scanpys function for GPU compute.

    Parameters
    ----------
    adata : AnnData
        annData object with 'neighbors' field.
       
    init_pos
        `'paga'`/`True`, `None`/`False`, or any valid 2d-`.obsm` key.
        Use precomputed coordinates for initialization.
        If `False`/`None` (the default), initialize randomly.
    max_iter : integer
        This controls the maximum number of levels/iterations of the
        Force Atlas algorithm. When specified the algorithm will terminate
        after no more than the specified number of iterations.
        No error occurs when the algorithm terminates in this manner.
        Good short-term quality can be achieved with 50-100 iterations.
        Above 1000 iterations is discouraged.
    
    Returns
    ----------
    updates `adata` with the following fields.

    X_draw_graph_layout_fa : `adata.obsm`
         Coordinates of graph layout.
    """
    
    from cugraph.layout import force_atlas2
    # Adjacency graph
    adjacency = adata.obsp["connectivities"]
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    g = cugraph.Graph()
    if hasattr(g, 'add_adj_list'):
        g.add_adj_list(offsets, indices, None)
    else:
        g.from_cudf_adjlist(offsets, indices, None)
    #Get Intial Positions
    if init_pos in adata.obsm.keys():
        init_coords = adata.obsm[init_pos]
    elif init_pos == 'paga' or init_pos:
        if 'paga' in adata.uns and 'pos' in adata.uns['paga']:
            groups = adata.obs[adata.uns['paga']['groups']]
            pos = adata.uns['paga']['pos']
            connectivities_coarse = adata.uns['paga']['connectivities']
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
            raise ValueError('Plot PAGA first, so that adata.uns[\'paga\']' 'with key \'pos\'.')
        
    else:
        init_coords = None
    
    if init_coords is not None: 
        x,y = np.hsplit(init_coords, init_coords.shape[1])
        inital_df = cudf.DataFrame({"x":x.ravel(),"y":y.ravel()})
        inital_df["vertex"] = inital_df.index
    else:
        inital_df= None
    #Run cugraphs Force Atlas 2 
    positions = force_atlas2(input_graph = g, pos_list=inital_df, max_iter= max_iter,outbound_attraction_distribution=False,  
            lin_log_mode=False,  
            edge_weight_influence=1.0,
            # Performance
            jitter_tolerance=1.0,  # Tolerance
            barnes_hut_optimize=True,
            barnes_hut_theta=1.2,
            # Tuning
            scaling_ratio=2.0,
            strong_gravity_mode=False,
            gravity=1.0,)
    positions = cp.vstack((positions["x"].to_cupy(),positions["y"].to_cupy())).T
    layout = "fa"
    adata.uns['draw_graph'] = {}
    adata.uns['draw_graph']['params'] = dict(layout=layout, random_state=0)
    key_added = f'X_draw_graph_{layout}'
    adata.obsm[key_added] = positions.get()    # Format output
    
def embedding_density(
    adata: AnnData,
    basis: str = 'umap',
    groupby = None,
    key_added = None,
    components = None,
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
    components
        The embedding dimensions over which the density should be calculated.
        This is limited to two components.
    Returns
    -------
    Updates `adata.obs` with an additional field specified by the `key_added`
    parameter. This parameter defaults to `[basis]_density_[groupby]`, where
    `[basis]` is one of `umap`, `diffmap`, `pca`, `tsne`, or `draw_graph_fa`
    and `[groupby]` denotes the parameter input.
    Updates `adata.uns` with an additional field `[key_added]_params`.
    """
    # to ensure that newly created covariates are categorical
    # to test for category numbers
    sanitize_anndata(adata)
    # Test user inputs
    basis = basis.lower()

    if basis == 'fa':
        basis = 'draw_graph_fa'

    if f'X_{basis}' not in adata.obsm_keys():
        raise ValueError(
            "Cannot find the embedded representation "
            f"`adata.obsm['X_{basis}']`. Compute the embedding first."
        )

    if components is None:
        components = '1,2'
    if isinstance(components, str):
        components = components.split(',')
    components = np.array(components).astype(int) - 1

    if len(components) != 2:
        raise ValueError('Please specify exactly 2 components, or `None`.')

    if basis == 'diffmap':
        components += 1

    if groupby is not None:
        if groupby not in adata.obs:
            raise ValueError(f'Could not find {groupby!r} `.obs` column.')

        if adata.obs[groupby].dtype.name != 'category':
            raise ValueError(f'{groupby!r} column does not contain categorical data')

    # Define new covariate name
    if key_added is not None:
        density_covariate = key_added
    elif groupby is not None:
        density_covariate = f'{basis}_density_{groupby}'
    else:
        density_covariate = f'{basis}_density'

    # Calculate the densities over each category in the groupby column
    if groupby is not None:
        categories = adata.obs[groupby].cat.categories

        density_values = np.zeros(adata.n_obs)

        for cat in categories:
            cat_mask = adata.obs[groupby] == cat
            embed_x = adata.obsm[f'X_{basis}'][cat_mask, components[0]]
            embed_y = adata.obsm[f'X_{basis}'][cat_mask, components[1]]

            dens_embed = _calc_density(cp.array(embed_x), cp.array(embed_y))
            density_values[cat_mask] = dens_embed

        adata.obs[density_covariate] = density_values
    else:  # if groupby is None
        # Calculate the density over the whole embedding without subsetting
        embed_x = adata.obsm[f'X_{basis}'][:, components[0]]
        embed_y = adata.obsm[f'X_{basis}'][:, components[1]]

        adata.obs[density_covariate] = _calc_density(cp.array(embed_x), cp.array(embed_y))

    # Reduce diffmap components for labeling
    # Note: plot_scatter takes care of correcting diffmap components
    #       for plotting automatically
    if basis != 'diffmap':
        components += 1

    adata.uns[f'{density_covariate}_params'] = dict(
        covariate=groupby, components=components.tolist()
    )

def _calc_density(x: cp.ndarray, y: cp.ndarray):
    """\
    Calculates the density of points in 2 dimensions.
    """
    from cuml.neighbors import KernelDensity
    
    # Calculate the point density
    xy = cp.vstack([x, y]).T
    bandwidth = cp.power(xy.shape[0],(-1./(xy.shape[1]+4)))
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(xy)
    z = kde.score_samples(xy)
    min_z = cp.min(z)
    max_z = cp.max(z)

    # Scale between 0 and 1
    scaled_z = (z - min_z) / (max_z - min_z)
    
    return scaled_z.get()

def plt_scatter(cudata, x, y, color = None, save = None, show =True, dpi =300):
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
    if color == None:
        sns.scatterplot(data=cudata.obs, x=x, y=y,s=2, color="grey", edgecolor="grey")
    else:
        sns.scatterplot(data=cudata.obs, x=x, y=y,s=2, hue=color)

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


def harmony_integrate(
    adata: AnnData,
    key: str,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    **kwargs,
):
    """\
    Use harmonypy to integrate different experiments.
    Harmony is an algorithm for integrating single-cell
    data from multiple experiments. This function uses the python
    gpu-computing based port of Harmony, to integrate single-cell data
    stored in an AnnData object. As Harmony works by adjusting the
    principal components, this function should be run after performing
    PCA but before computing the neighbor graph, as illustrated in the
    example below.
    Parameters
    ----------
    adata
        The annotated data matrix.
    key
        The name of the column in ``adata.obs`` that differentiates
        among experiments/batches.
    basis
        The name of the field in ``adata.obsm`` where the PCA table is
        stored. Defaults to ``'X_pca'``, which is the default for
        ``sc.tl.pca()``.
    adjusted_basis
        The name of the field in ``adata.obsm`` where the adjusted PCA
        table will be stored after running this function. Defaults to
        ``X_pca_harmony``.
    kwargs
        Any additional arguments will be passed to
        ``harmonypy.run_harmony()``.
    Returns
    -------
    Updates adata with the field ``adata.obsm[obsm_out_field]``,
    containing principal components adjusted by Harmony such that
    different experiments are integrated.
    """
    from . import harmonpy_gpu
    harmony_out = harmonpy_gpu.run_harmony(adata.obsm[basis], adata.obs, key, **kwargs)

    adata.obsm[adjusted_basis] = harmony_out.Z_corr.T.get()