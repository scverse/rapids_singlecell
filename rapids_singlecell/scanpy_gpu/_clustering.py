import cupy as cp
import cudf
import cugraph
import numpy as np
import pandas as pd
from cuml.cluster import KMeans

from anndata import AnnData

from natsort import natsorted

import warnings

def leiden(adata: AnnData, 
           resolution=1.0,
           use_weights: bool =True,
           neighbors_key = None,
           key_added: str = 'leiden'):
    """
    Performs Leiden Clustering using cuGraph
    Parameters
    ----------
    adata : annData object with 'neighbors' field.
       
    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        
    use_weights : bool (default: True) 
        If `True`, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    
    neighbors_key : (default: None)
        If not specified, `leiden` looks at .obsp['connectivities'] for neighbors connectivities
        If specified, `leiden` looks at .obsp['neighbors_key_ connectivities'] for neighbors connectivities
    
    key_added
        `adata.obs` key under which to add the cluster labels.
    """
    # Adjacency graph
    
    if neighbors_key:
        adjacency = adata.obsp[neighbors_key+"_connectivities"]
    else:
        adjacency = adata.obsp["connectivities"]
        
    if use_weights:
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        df =cudf.DataFrame({"0":sources,"1":targets,"2":weights})
    else:
        sources, targets = adjacency.nonzero()
        weights = np.ones(len(targets))
        df =cudf.DataFrame({"0":sources,"1":targets,"2":weights})
    
    g = cugraph.from_cudf_edgelist(df, source='0', destination='1',edge_attr='2', renumber=False)
    
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
            use_weights: bool =True,
            neighbors_key = None,
            key_added: str = 'louvain'):
    """
    Performs Louvain Clustering using cuGraph
    Parameters
    ----------
    adata : annData object with 'neighbors' field.
       
    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
    
    use_weights : bool (default: True) 
        If `True`, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
        
    neighbors_key : (default: None)
        If not specified, `louvain` looks at .obsp['connectivities'] for neighbors connectivities
        If specified, `louvain` looks at .obsp['neighbors_key_ connectivities'] for neighbors connectivities
    
    key_added
        `adata.obs` key under which to add the cluster labels.
    """
    # Adjacency graph
    
    if neighbors_key:
        adjacency = adata.obsp[neighbors_key+"_connectivities"]
    else:
        adjacency = adata.obsp["connectivities"]
    
    if use_weights:
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        df =cudf.DataFrame({"0":sources,"1":targets,"2":weights})
    else:
        sources, targets = adjacency.nonzero()
        weights = np.ones(len(targets))
        df =cudf.DataFrame({"0":sources,"1":targets,"2":weights})
    
    g = cugraph.from_cudf_edgelist(df, source='0', destination='1',edge_attr='2', renumber=False)
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
           key_added = "kmeans",
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
    groups = kmeans_out.labels_.astype(str)

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=natsorted(map(str, np.unique(groups))),
    )
