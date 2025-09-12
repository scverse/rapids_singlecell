from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
import cupy as cp
# from scipy.sparse import csr_matrix, issparse, isspmatrix_csr, spmatrix
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
    sepal_simulation_debug,
    sepal_simulation,

)



from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import isspmatrix_csr as cp_isspmatrix_csr
from cupyx.scipy.sparse import isspmatrix_csc as cp_isspmatrix_csc



def _score_helper_gpu_debug(
    gene_idx: int,
    vals: cp.ndarray,
    max_neighs: int,
    n_iter: int,
    sat: cp.ndarray,
    sat_idx: cp.ndarray, 
    unsat: cp.ndarray,
    unsat_idx: cp.ndarray,
    dt: float = 0.001,
    thresh: float = 1e-8,
    debug: bool = False,
) -> cp.ndarray:
    """GPU-accelerated score computation for a single gene with debug."""
    
    n_cells = vals.shape[0]
    n_sat = len(sat)
    n_unsat = len(unsat)
    sat_thresh = max_neighs
    
    if debug:
        print(f"Processing gene {gene_idx} of 1")
        print(f"GPU Input validation:")
        print(f"  n_cells: {n_cells}, n_sat: {n_sat}, n_unsat: {n_unsat}")
        print(f"  vals[0:3]: {vals[0:3].get()}")
        print(f"  sat[0:3]: {sat[0:3].get()}")
    
    # Check GPU memory limits
    device = cp.cuda.Device()
    max_shared_mem = device.attributes['MaxSharedMemoryPerBlock']
    if debug:
        print(f"Max shared memory per block: {max_shared_mem} bytes")
    
    # Calculate shared memory requirements
    conc_floats = n_cells
    nhood_floats = n_sat  
    dcdt_floats = n_cells
    total_floats = conc_floats + nhood_floats + dcdt_floats
    shared_mem_bytes = total_floats * 4  # 4 bytes per float
    if debug:
        print(f"Required shared memory: {total_floats} floats = {shared_mem_bytes} bytes")
    
    if shared_mem_bytes > max_shared_mem:
        if debug:
            print(f"ERROR: Required shared memory ({shared_mem_bytes}) exceeds limit ({max_shared_mem})")
            print("Consider using global memory instead of shared memory")
        result = cp.array([-999999.0], dtype=cp.float32)
        return result
    
    # Allocate result on GPU
    result = cp.zeros(1, dtype=cp.float32)
    
    # Launch debug kernel with proper block size
    threads_per_block = min(256, n_cells)
    if debug:
        try:
            sepal_simulation_debug(
                (1,),  # 1 block for single gene
                (threads_per_block,),
                (
                    vals.astype(cp.float32),           # gene_data
                    sat.astype(cp.int32),              # sat
                    sat_idx.astype(cp.int32),          # sat_idx  
                    unsat.astype(cp.int32),            # unsat
                    unsat_idx.astype(cp.int32),        # unsat_idx
                    result,                            # result
                    n_cells,                           # n_cells
                    n_sat,                             # n_sat
                    n_unsat,                           # n_unsat
                    sat_thresh,                        # sat_thresh
                    max_neighs,                        # max_neighs
                    n_iter,                            # n_iter
                    cp.float32(dt),                    # dt
                    cp.float32(thresh)                 # thresh
                ),
                shared_mem=shared_mem_bytes
            )
            cp.cuda.Device().synchronize()  # Wait for completion
            print(f"GPU result: {result.get()}")
        except Exception as e:
            print(f"GPU kernel error: {e}")
            result[0] = -999999.0
    else:
        sepal_simulation(
            (1,),  # 1 block for single gene
            (threads_per_block,),
            (
                vals.astype(cp.float32),           # gene_data
                sat.astype(cp.int32),              # sat
                sat_idx.astype(cp.int32),          # sat_idx  
                unsat.astype(cp.int32),            # unsat
                unsat_idx.astype(cp.int32),        # unsat_idx
                result,                            # result
                n_cells,                           # n_cells
                n_sat,                             # n_sat
                n_unsat,                           # n_unsat
                sat_thresh,                        # sat_thresh
                max_neighs,                        # max_neighs
                n_iter,                            # n_iter
                cp.float32(dt),                    # dt
                cp.float32(thresh)                 # thresh
            ),
            shared_mem=shared_mem_bytes
        )
    
    return result

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
    Identify spatially variable genes with *Sepal*.

    *Sepal* is a method that simulates a diffusion process to quantify spatial structure in tissue.
    See :cite:`andersson2021` for reference.

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

    spatial = adata.obsm[spatial_key].astype(cp.float32)

    if genes is None:
        genes = adata.var_names.values
        if "highly_variable" in adata.var.columns:
            genes = genes[adata.var["highly_variable"].values]
    genes = _assert_non_empty_sequence(genes, name="genes")

    g = adata.obsp[connectivity_key]
    if not cp_isspmatrix_csr(g):
        g = cp_csr_matrix(g)
    g.eliminate_zeros()

    degrees = cp.diff(g.indptr)
    max_n = degrees.max()
    if max_n != max_neighs:
        raise ValueError(f"Expected `max_neighs={max_neighs}`, found node with `{max_n}` neighbors.")

    # get saturated/unsaturated nodes
    sat, sat_idx, unsat, unsat_idx = _compute_idxs_gpu(
        g=g,
        degrees=degrees,
        spatial=spatial,
        sat_thresh=max_neighs,
    )
    # sat_idx is [n_sat, sat_thresh]
    # unsat_idx is [n_unsat]
    # get counts
    vals, genes = _extract_expression(adata, genes=genes, use_raw=use_raw, layer=layer)
    start = logg.info(f"Calculating sepal score for `{len(genes)}` genes using GPU")

    # vals is already on GPU

        # Process genes in batches
    all_scores = []
    if cp_isspmatrix_csr(vals) or cp_isspmatrix_csc(vals):
        vals = vals.toarray()
    for gene_idx in range(len(genes)):
        print(f"Processing gene {gene_idx} of {len(genes)}")
        gene_score = _score_helper_gpu_debug(
            gene_idx=gene_idx,
            vals=vals[:, gene_idx],
            max_neighs=max_neighs,
            n_iter=n_iter,
            sat=sat,
            sat_idx=sat_idx, 
            unsat=unsat,
            unsat_idx=unsat_idx,
            dt=dt,
            thresh=thresh,
            debug=debug,
        )
        all_scores.append(gene_score.get())
    

    score = np.concatenate(all_scores)

    key_added = "sepal_score"
    sepal_score = pd.DataFrame(score, index=genes, columns=[key_added])

    if sepal_score[key_added].isna().any():
        logg.warning("Found `NaN` in sepal scores, consider increasing `n_iter` to a higher value")
    sepal_score = sepal_score.sort_values(by=key_added, ascending=False)

    if copy:
        start = logg.info("Finish", time=start)
        logg.info("Finish", time=start)
        return sepal_score

    _save_data(adata, attr="uns", key=key_added, data=sepal_score, time=start)
    return sepal_score


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
                spatial,                             # spatial coordinates [n_nodes, 2] (read only assumed to be float32)
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
