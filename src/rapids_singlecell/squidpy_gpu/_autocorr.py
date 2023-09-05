from typing import (
    Literal,  # < 3.8
    Optional,
    Sequence,
    Union,
)

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy import sparse as sparse_gpu
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from ._gearysc import _gearys_C_cupy
from ._moransi import _morans_I_cupy
from ._utils import _p_value_calc


def spatial_autocorr(
    adata: AnnData,
    connectivity_key: str = "spatial_connectivities",
    genes: Union[str, Sequence[str], None] = None,
    mode: Literal["moran", "geary"] = "moran",
    transformation: bool = True,
    n_perms: Union[int, None] = None,
    two_tailed: bool = False,
    corr_method: Union[str, None] = "fdr_bh",
    layer: Union[str, None] = None,
    use_raw: bool = False,
    use_sparse: bool = False,
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Calculate spatial autocorrelation for genes in an AnnData object.

    This function computes spatial autocorrelation scores (Moran's I or Geary's C) for each gene in an AnnData object.
    The function also calculates p-values and corrected p-values for multiple testing.

    Note:
        This implementation uses single-precision (float32) for calculations, which may result in decreased accuracy for weak
        correlations when compared to double-precision (float64) calculations. For strongly correlated data, the difference in p-values
        should be minimal. However, for weakly correlated data with I or C values close to their expected values, the lack of precision
        may lead to larger discrepancies in p-values.


    Parameters
    ----------
        adata
            Annotated data matrix.
        connectivity_key
            Key of the connectivity matrix in `adata.obsp`, by default "spatial_connectivities".
        genes
            Genes for which to compute the autocorrelation scores. If None, all genes or highly variable genes will be used.
        mode
            Spatial autocorrelation method to use, either "moran" or "geary", by default "moran".
        transformation
            If True, row-normalize the connectivity matrix, by default True.
        n_perms
            Number of permutations for calculating p-values, by default None.
        two_tailed
            If True, calculate two-tailed p-values, by default False.
        corr_method
            Multiple testing correction method to use, by default "fdr_bh".
        layer
            Layer in the AnnData object to use, by default None.
        use_raw
            If True, use the raw data in the AnnData object, by default False.
        use_sparse
            If True, use a sparse representation for the input matrix `vals` when it is a sparse matrix, by default False.
        copy
            If True, return the results as a DataFrame instead of storing them in `adata.uns`, by default False.

    Returns
    -------
            DataFrame containing the autocorrelation scores, p-values, and corrected p-values for each gene. \
            If `copy` is False, the results are stored in `adata.uns` and None is returned.
    """
    if genes is None:
        if "highly_variable" in adata.var:
            genes = adata[:, adata.var["highly_variable"]].var_names.values
        else:
            genes = adata.var_names.values
    if isinstance(genes, str):
        genes = [genes]
    if use_raw:
        if adata.raw is None:
            raise AttributeError(
                "No `.raw` attribute found. Try specifying `use_raw=False`."
            )
        genes = list(set(genes) & set(adata.raw.var_names))
        vals = adata.raw[:, genes].X
    else:
        if layer:
            vals = adata[:, genes].layers["layer"]
        else:
            vals = adata[:, genes].X
    # create Adj-Matrix
    adj_matrix = adata.obsp[connectivity_key]
    adj_matrix_cupy = sparse_gpu.csr_matrix(adj_matrix, dtype=cp.float32)

    if transformation:  # row-normalize
        row_sums = adj_matrix_cupy.sum(axis=1).reshape(-1, 1)
        non_zero_rows = row_sums != 0
        row_sums[non_zero_rows] = 1.0 / row_sums[non_zero_rows]
        adj_matrix_cupy = adj_matrix_cupy.multiply(sparse_gpu.csr_matrix(row_sums))

    params = {"two_tailed": two_tailed}

    # check sparse:
    if use_sparse:
        vals = sparse.csr_matrix(vals)
        data = sparse_gpu.csr_matrix(vals, dtype=cp.float32)
    else:
        if sparse.issparse(vals):
            vals = vals.toarray()
        data = cp.array(vals, dtype=cp.float32)
    # Run Spartial Autocorr
    if mode == "moran":
        score, score_perms = _morans_I_cupy(
            data, adj_matrix_cupy, n_permutations=n_perms
        )
        params["stat"] = "I"
        params["expected"] = -1.0 / (adata.shape[0] - 1)  # expected score
        params["ascending"] = False
        params["mode"] = "moranI"
    elif mode == "geary":
        score, score_perms = _gearys_C_cupy(
            data, adj_matrix_cupy, n_permutations=n_perms
        )
        params["stat"] = "C"
        params["expected"] = 1.0
        params["ascending"] = True
        params["mode"] = "gearyC"
    else:
        raise NotImplementedError(f"Mode `{mode}` is not yet implemented.")
    g = sparse.csr_matrix(adj_matrix_cupy.get())
    score = score.get()
    if n_perms is not None:
        score_perms = score_perms.get()
    with np.errstate(divide="ignore"):
        pval_results = _p_value_calc(score, score_perms, g, params)

    df = pd.DataFrame({params["stat"]: score, **pval_results}, index=genes)

    if corr_method is not None:
        for pv in filter(lambda x: "pval" in x, df.columns):
            _, pvals_adj, _, _ = multipletests(
                df[pv].values, alpha=0.05, method=corr_method
            )
            df[f"{pv}_{corr_method}"] = pvals_adj

    df.sort_values(by=params["stat"], ascending=params["ascending"], inplace=True)
    if copy:
        return df
    adata.uns[params["mode"]] = df
