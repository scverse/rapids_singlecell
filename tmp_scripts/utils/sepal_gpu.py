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
    GPU-accelerated sepal implementation with unlimited scalability.
    Handles datasets from thousands to millions of cells.
    """
    if isinstance(adata, SpatialData):
        adata = adata.table
    
    _assert_connectivity_key(adata, connectivity_key)
    _assert_spatial_basis(adata, key=spatial_key)
    if max_neighs not in (4, 6):
        raise ValueError(f"Expected `max_neighs` to be either `4` or `6`, found `{max_neighs}`.")

    # Setup (unchanged)
    spatial = cp.asarray(adata.obsm[spatial_key], dtype=cp.float64)
    
    if genes is None:
        genes = adata.var_names.values
        if "highly_variable" in adata.var.columns:
            genes = genes[adata.var["highly_variable"].values]
    genes = _assert_non_empty_sequence(genes, name="genes")

    # Graph and index computation (unchanged)
    g = adata.obsp[connectivity_key]
    if not cp_isspmatrix_csr(g):
        g = cp_csr_matrix(g)
    g.eliminate_zeros()

    degrees = cp.diff(g.indptr)
    max_n = degrees.max()
    if max_n != max_neighs:
        raise ValueError(f"Expected `max_neighs={max_neighs}`, found node with `{max_n}` neighbors.")

    sat, sat_idx, unsat, unsat_to_nearest_sat = _compute_idxs_gpu(
        g=g,
        degrees=degrees,
        spatial=spatial,
        sat_thresh=max_neighs,
    )
    
    # Expression data
    vals, genes = _extract_expression(adata, genes=genes, use_raw=use_raw, layer=layer)
    start = logg.info(f"Calculating sepal score for `{len(genes)}` genes using scalable GPU kernel")
    
    if debug:
        print(f"Dataset: {len(genes)} genes, {len(sat)} saturated, {len(unsat)} unsaturated cells")
    
    # Convert to optimal format
    if cp_isspmatrix_csr(vals) or cp_isspmatrix_csc(vals):
        vals = vals.toarray()
    
    vals = cp.ascontiguousarray(cp.asarray(vals, dtype=cp.float64))
    
    # Run scalable simulation - handles ANY dataset size!
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
    
    # Results processing (unchanged)
    score = cp.asnumpy(scores)
    
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
    
    n_cells, n_genes = vals.shape
    n_sat = len(sat)
    n_unsat = len(unsat)
    
    # Reorder: [sat_nodes, unsat_nodes] for coalesced access
    reorder_indices = cp.concatenate([sat, unsat])
    vals_reordered = vals[reorder_indices, :]  # (n_cells, n_genes) reordered
    
    # Create a flat mapping for unsat nodes to their nearest saturated
    unsat_to_nearest_sat_remapped = cp.searchsorted(sat, unsat_to_nearest_sat[unsat])
    
    
    reordered_size = n_cells  # Total cells after reordering
    
    # **MULTI-GENE PARALLEL PROCESSING**
    # Grid size = n_genes (one block per gene)
    threads_per_block = 256
    blocks_per_grid = n_genes  # Process ALL genes in parallel!
    
    print(f"🚀 LAUNCHING LARGE GRID: {blocks_per_grid} blocks × {threads_per_block} threads = {blocks_per_grid * threads_per_block} total threads")
    
    # Allocate arrays for ALL genes at once
    concentration_all = cp.ascontiguousarray(vals_reordered.T, dtype=cp.float64)  # (n_genes, n_cells)
    derivatives_all = cp.zeros((n_genes, reordered_size), dtype=cp.float64)
    results_all = cp.full(n_genes, -999999.0, dtype=cp.float32)  # Results for ALL genes
    
    # Calculate shared memory (fixed size per block, independent of n_cells)
    tile_size = 1024  # Fixed tile size for scalability
    min_blocks = 256  # Hardware-specific minimum
    blocks_per_grid = max(n_genes, min_blocks)
    shared_mem_size = tile_size * 2 * 8  # 2 double arrays per tile
    
    print(f"💾 Memory layout: {n_genes} genes × {reordered_size} cells = {concentration_all.nbytes / 1e6:.1f} MB")
    print(f"🔧 Shared memory per block: {shared_mem_size / 1024:.1f} KB (independent of dataset size)")
    
    # **SINGLE KERNEL LAUNCH FOR ALL GENES**
    sepal_simulation(
        (blocks_per_grid,),                    # Grid: one block per gene
        (threads_per_block,),                  # Block: 256 threads
        (
            concentration_all,                 # (n_genes, n_cells) - all genes
            derivatives_all,                   # (n_genes, n_cells) - all derivatives
            sat_idx,                     
            unsat_to_nearest_sat_remapped,    
            results_all,                       # (n_genes,) - results for all genes
            reordered_size,                    # n_cells (can be 1M+)
            n_genes,                          # Number of genes to process
            n_sat,
            n_unsat, 
            max_neighs,
            max_neighs,  # sat_thresh
            n_iter,
            cp.float32(dt),                    # **FIX: Convert to float32**
            cp.float32(thresh),                # **FIX: Convert to float32**
            debug
        ),
        shared_mem=shared_mem_size
    )
    
    # Convert results
    final_scores = cp.where(results_all == -999999.0, cp.nan, dt * results_all)
    
    return final_scores  # Shape: (n_genes,)


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
