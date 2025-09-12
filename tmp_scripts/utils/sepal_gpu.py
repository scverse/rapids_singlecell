from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
import cupy as cp
from spatialdata import SpatialData
from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import (
    _assert_connectivity_key,
    _assert_non_empty_sequence,
    _assert_spatial_basis,
    _extract_expression,
    _save_data,
)
from .kernels.sepal_kernels import (
    get_nhood_idx_with_distance,
    sepal_simulation,  # Import the CUDA kernel
)

from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import isspmatrix_csr as cp_isspmatrix_csr
from cupyx.scipy.sparse import isspmatrix_csc as cp_isspmatrix_csc


def sepal_gpu(
    adata: AnnData | SpatialData,
    max_neighs: Literal[4, 6],
    genes: str | Sequence[str] | None = None,
    n_iter: int = 30000,
    dt: float = 0.001,
    thresh: float = 1e-8,
    connectivity_key: str = Key.obsp.spatial_conn(),
    spatial_key: str = Key.obsm.spatial,
    layer: str | None = None,
    use_raw: bool = False,
    copy: bool = False,
    debug: bool = False,
) -> pd.DataFrame | None:
    """
    Identify spatially variable genes with *Sepal* using custom CUDA kernel.

    *Sepal* is a method that simulates a diffusion process to quantify spatial structure in tissue.
    See :cite:`andersson2021` for reference.

    This implementation uses a custom CUDA kernel for maximum performance and CPU compatibility.

    Parameters
    ----------
    %(adata)s
    max_neighs
        Maximum number of neighbors of a node in the graph. Valid options are:

            - `4` - for a square-grid (ST, Dbit-seq).
            - `6` - for a hexagonal-grid (Visium).
    genes
        List of gene names, as stored in :attr:`anndata.AnnData.var_names`, used to compute sepal score.

        If `None`, it's computed :attr:`anndata.AnnData.var` ``['highly_variable']``, if present.
        Otherwise, it's computed for all genes.
    n_iter
        Maximum number of iterations for the diffusion simulation.
        If ``n_iter`` iterations are reached, the simulation will terminate
        even though convergence has not been achieved.
    dt
        Time step in diffusion simulation.
    thresh
        Entropy threshold for convergence of diffusion simulation.
    %(conn_key)s
    %(spatial_key)s
    layer
        Layer in :attr:`anndata.AnnData.layers` to use. If `None`, use :attr:`anndata.AnnData.X`.
    use_raw
        Whether to access :attr:`anndata.AnnData.raw`.
    debug
        Whether to run in debug mode.
    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the sepal scores.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['sepal_score']`` - the sepal scores.

    Notes
    -----
    If some genes in :attr:`anndata.AnnData.uns` ``['sepal_score']`` are `NaN`,
    consider re-running the function with increased ``n_iter``.
    """
    if isinstance(adata, SpatialData):
        adata = adata.table
    
    _assert_connectivity_key(adata, connectivity_key)
    _assert_spatial_basis(adata, key=spatial_key)
    if max_neighs not in (4, 6):
        raise ValueError(f"Expected `max_neighs` to be either `4` or `6`, found `{max_neighs}`.")

    # Setup exactly like CPU but on GPU
    spatial = cp.asarray(adata.obsm[spatial_key], dtype=cp.float64)
    
    if genes is None:
        genes = adata.var_names.values
        if "highly_variable" in adata.var.columns:
            genes = genes[adata.var["highly_variable"].values]
    genes = _assert_non_empty_sequence(genes, name="genes")

    # Graph setup
    g = adata.obsp[connectivity_key]
    if not cp_isspmatrix_csr(g):
        g = cp_csr_matrix(g)
    g.eliminate_zeros()

    degrees = cp.diff(g.indptr)
    max_n = degrees.max()
    if max_n != max_neighs:
        raise ValueError(f"Expected `max_neighs={max_neighs}`, found node with `{max_n}` neighbors.")

    # Get saturated/unsaturated nodes
    sat, sat_idx, unsat, unsat_to_nearest_sat = _compute_idxs_gpu(
        g=g,
        degrees=degrees,
        spatial=spatial,
        sat_thresh=max_neighs,
    )
    
    # Get expression data
    vals, genes = _extract_expression(adata, genes=genes, use_raw=use_raw, layer=layer)
    start = logg.info(f"Calculating sepal score for `{len(genes)}` genes using CUDA kernel")
    print("genes", genes)
    # Convert to GPU arrays with consistent dtype
    if cp_isspmatrix_csr(vals) or cp_isspmatrix_csc(vals):
        vals = vals.toarray()

    vals = cp.asarray(vals, dtype=cp.float64)
    
    # Run CUDA kernel-based simulation
    scores = _cuda_kernel_diffusion_gpu(
        vals=vals,
        sat=sat,
        sat_idx=sat_idx,
        unsat=unsat,
        unsat_to_nearest_sat=unsat_to_nearest_sat,
        max_neighs=max_neighs,
        n_iter=n_iter,
        dt=dt,
        thresh=thresh,
        debug=debug,
    )
    
    # Convert back to CPU
    score = cp.asnumpy(scores)
    
    # Create results dataframe
    key_added = "sepal_score"
    sepal_score = pd.DataFrame(score, index=genes, columns=[key_added])
    
    if sepal_score[key_added].isna().any():
        logg.warning("Found `NaN` in sepal scores, consider increasing `n_iter` to a higher value")
    sepal_score = sepal_score.sort_values(by=key_added, ascending=False)

    if copy:
        logg.info("Finish", time=start)
        return sepal_score

    _save_data(adata, attr="uns", key=key_added, data=sepal_score, time=start)
    return sepal_score


def _cuda_kernel_diffusion_gpu(
    vals: cp.ndarray,                    # (n_cells, n_genes) - all gene expressions
    sat: cp.ndarray,                     # (n_sat,) - saturated node indices
    sat_idx: cp.ndarray,                 # (n_sat, max_neighs) - neighborhood indices for sat nodes
    unsat: cp.ndarray,                   # (n_unsat,) - unsaturated node indices  
    unsat_to_nearest_sat: cp.ndarray,    # (n_unsat,) - nearest sat for each unsat
    max_neighs: int,
    n_iter: int,
    dt: float,
    thresh: float,
    debug: bool = False,
) -> cp.ndarray:
    """
    Run diffusion simulation using custom CUDA kernel for each gene.
    This processes genes one by one using the optimized CUDA kernel.
    """
    n_cells, n_genes = vals.shape
    n_sat = len(sat)
    n_unsat = len(unsat)
    
    # Flatten sat_idx for kernel (kernel expects flattened array)
    # sat_idx_flat = sat_idx.flatten(order="C")
    
    # Allocate working arrays for the kernel (reused for each gene)
    concentration = cp.zeros(n_cells, dtype=cp.float64)
    derivatives = cp.zeros(n_cells, dtype=cp.float64)
    result_gpu = cp.zeros(1, dtype=cp.float32)
    
    # Store results for all genes
    all_scores = []
    
    # Process each gene using the CUDA kernel
    for gene_idx in range(n_genes):
        if debug:
            print(f"Processing gene {gene_idx}/{n_genes}")
            
        # Copy gene data to working array
        
        # Launch CUDA kernel for this gene
        # Block size should be a multiple of warp size (32) and handle convergence sync
        block_size = 256
        grid_size = 1  # Single block since kernel uses shared memory for convergence
        
        sepal_simulation(
            (grid_size,), (block_size,),
            (
                vals,              # const double* gene_data [n_cells]
                gene_idx,             # int gene_idx
                sat,                    # const int* sat [n_sat]  
                sat_idx,           # const int* sat_idx [n_sat , sat_thresh]
                unsat,                  # const int* unsat [n_unsat]
                unsat_to_nearest_sat,   # const int* unsat_idx [n_unsat]
                concentration,          # double* concentration [n_cells] (working array)
                derivatives,            # double* derivatives [n_cells] (working array)
                result_gpu,             # float* result [1] (output)
                n_cells,                # int n_cells
                n_genes,                # int n_genes
                n_sat,                  # int n_sat  
                n_unsat,                # int n_unsat
                max_neighs,             # int sat_thresh (same as max_neighs)
                max_neighs,             # int max_neighs
                n_iter,                 # int n_iter
                dt,                     # double dt
                thresh,                 # double thresh
                debug                   # bool debug
            )
        )
        
        # Get result and handle special values
        kernel_result = float(result_gpu[0])
        if kernel_result == -999999.0:  # Kernel's NaN indicator
            final_score = np.nan
        else:
            final_score = dt * kernel_result  # Apply dt multiplication here
            
        all_scores.append(final_score)
        
        if debug:
            print(f"  Gene {gene_idx} score: {final_score}")
    
    return cp.asarray(all_scores, dtype=cp.float64)


def _compute_idxs_gpu(
    g: cp_csr_matrix,
    degrees: cp.ndarray,
    spatial: cp.ndarray, 
    sat_thresh: int, 
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Compute saturated/unsaturated indices on GPU with unified distance computation."""
    
    # Get saturated and unsaturated nodes
    unsat_mask = degrees < sat_thresh
    sat_mask = degrees == sat_thresh
    
    unsat = cp.asarray(cp.where(unsat_mask)[0], dtype=cp.int32)
    sat = cp.asarray(cp.where(sat_mask)[0], dtype=cp.int32)
    
    # Extract saturated neighborhoods with vectorized CuPy
    nearest_sat = cp.full(len(unsat), -1, dtype=cp.int32)
    sat_idx = g.indices[g.indptr[sat][:, None] + cp.arange(sat_thresh)]
    
    # Single kernel handles both graph neighbors and distance fallback
    if len(unsat) > 0:
        threads_per_block = 256
        blocks = (len(unsat) + threads_per_block - 1) // threads_per_block
        
        get_nhood_idx_with_distance(
            (blocks,), (threads_per_block,),
            (
                unsat,                              # unsaturated nodes (read only assumed to be int32)
                spatial,                             # spatial coordinates [n_nodes, 2] (read only assumed to be float64)
                sat,                                # saturated node list (read only assumed to be int32)
                g.indptr,                           # CSR indptr (read only assumed to be float32)
                g.indices,                          # CSR indices (read only assumed to be float32)
                sat_mask,                           # boolean mask for saturated nodes (read only assumed to be bool)
                nearest_sat,                        # output
                len(unsat),                         # number of unsaturated nodes
                len(sat)                            # number of saturated nodes
            )
        )
    
    return sat, sat_idx, unsat, nearest_sat
