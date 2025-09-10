from __future__ import annotations

import os
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

def compute_edistance_sklearn(X, Y, dtype=np.float64):
    """Compute edistance using sklearn's pairwise_distances with specified precision"""
    X = np.array(X, dtype=dtype)
    Y = np.array(Y, dtype=dtype)
    
    # Compute pairwise distances using sklearn
    sigma_X = sklearn_pairwise_distances(X, X, metric='euclidean').mean()
    sigma_Y = sklearn_pairwise_distances(Y, Y, metric='euclidean').mean()
    delta = sklearn_pairwise_distances(X, Y, metric='euclidean').mean()
    
    return 2 * delta - sigma_X - sigma_Y

def compute_edistance_pairwise_sklearn(adata, groupby, obsm_key="X_pca", dtype=np.float64):
    """Compute pairwise edistance matrix using sklearn with specified precision"""
    # Get data and convert to CPU numpy
    embedding = np.array(adata.obsm[obsm_key], dtype=dtype)
    
    groups = adata.obs[groupby].cat.categories
    k = len(groups)
    
    print(f"Computing edistance for {k} groups with dtype {dtype}...")
    
    # Build edistance matrix
    edistance_matrix = np.zeros((k, k), dtype=dtype)
    
    for i, group_a in enumerate(groups):
        mask_a = adata.obs[groupby] == group_a
        X = embedding[mask_a]
        
        for j, group_b in enumerate(groups):
            if i == j:
                edistance_matrix[i, j] = 0.0
            elif i < j:  # Only compute upper triangle
                mask_b = adata.obs[groupby] == group_b
                Y = embedding[mask_b]
                
                edist = compute_edistance_sklearn(X, Y, dtype=dtype)
                edistance_matrix[i, j] = edist
                edistance_matrix[j, i] = edist  # Symmetric
                
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{k} groups")
    
    return pd.DataFrame(edistance_matrix, index=groups, columns=groups)

if __name__ == "__main__":
    obs_key = "perturbation"
    
    # Load the data
    save_dir = os.path.join(os.path.expanduser("~"), "data")
    adata_path = os.path.join(save_dir, "adamson_2016_upr_epistasis_pca.h5ad")
    
    print(f"Loading data from {adata_path}...")
    adata = ad.read_h5ad(adata_path)
    
    print(f"Data shape: {adata.shape}")
    print(f"Groups: {len(adata.obs[obs_key].cat.categories)}")
    
    # Generate the float64 reference CSV using sklearn
    print("\nGenerating float64 reference using sklearn...")
    df_reference = compute_edistance_pairwise_sklearn(adata, obs_key, obsm_key="X_pca", dtype=np.float64)
    
    # Save the float64 reference
    output_path = os.path.join(save_dir, "df_cpu_float64.csv")
    df_reference.to_csv(output_path)
    print(f"Saved float64 reference to: {output_path}")
    
    # Show a sample of values
    print("\nSample values:")
    groups = df_reference.index[:3]
    for i, group_a in enumerate(groups):
        for j, group_b in enumerate(groups):
            if i < j:
                val = df_reference.loc[group_a, group_b]
                print(f"  {group_a} vs {group_b}: {val:.10f}")
    
    print("Float64 reference generation complete!")
