from __future__ import annotations

import math

import cupy as cp
import numpy as np

from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta

# Kernel definition
reduce_sum_2D_kernel = cp.RawKernel(
    r"""
    extern "C" __global__
    void aucell_reduce(const int* input, int* output, int R, int C, int max) {
        int row = blockIdx.x *blockDim.x+ threadIdx.x ;
        if (row>=R){
            return;
        }
        int sum = 0;
        int prev = input[row*C+0];
        int switcher = 1;
        if (prev < max){
            switcher = 0;
            for(int i  = 1; i<C; i++){
                int data = input[row*C+i];
                if(data<max) {
                    sum+= i *(data-prev);
                    prev = data;
                }
                else{
                    sum += i *(max-prev);
                    switcher = 1;
                    break;
                }
            }
        }
        if (switcher == 0){
            sum += C *(max-prev);
        }
        output[row] = sum;
    }
    """,
    "aucell_reduce",
)


def _auc(
    row, cnct, *, starts=None, offsets=None, n_up=None, n_fsets=None, max_aucs=None
):
    # Rank row
    row = cp.argsort(cp.argsort(-row), axis=1) + 1
    row = row.astype(cp.int32)
    # Empty acts
    es = cp.zeros((row.shape[0], n_fsets), dtype=cp.float32)
    es_inter = cp.zeros(row.shape[0], dtype=cp.int32)

    threads_per_block = 32
    blocks_per_grid = math.ceil(row.shape[0] / threads_per_block)

    for j in range(n_fsets):
        # Extract feature set
        srt = starts[j]
        off = offsets[j] + srt
        fset = cnct[srt:off]
        # Compute AUC
        x = row[:, fset]
        x.sort(axis=1)

        reduce_sum_2D_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (x, es_inter, x.shape[0], x.shape[1], n_up),
        )
        # Update acts matrix
        es[:, j] = (es_inter / max_aucs[j]).astype(cp.float32)
        es_inter[:] = 0
    return es


def _validate_n_up(
    nvar: int,
    n_up: int | float | None = None,
) -> int:
    assert isinstance(n_up, int | float) or n_up is None, (
        "n_up must be numerical or None"
    )
    if n_up is None:
        n_up = np.ceil(0.05 * nvar)
        n_up = int(np.clip(n_up, a_min=2, a_max=nvar))
    else:
        n_up = int(np.ceil(n_up))
    assert nvar >= n_up > 1, (
        f"For nvar={nvar}, n_up={n_up} must be between 1 and {nvar}"
    )
    return n_up


@docs.dedent
def _func_aucell(
    mat: cp.ndarray,
    *,
    cnct: cp.ndarray,
    starts: cp.ndarray,
    offsets: cp.ndarray,
    n_up: int | float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, None]:
    r"""
    Area Under the Curve for set enrichment within single cells (AUCell).

    Given a ranked list of features per observation, AUCell calculates the AUC by measuring how early the features in
    the set appear in this ranking. Specifically, the enrichment score :math:`ES` is:

    .. math::

       {ES}_{i, F} = \int_0^1 {RecoveryCurve}_{i, F}(r_i) \, dr

    Where:

    - :math:`i` is the observation
    - :math:`F` is the feature set
    - :math:`{RecoveryCurve}_{i, F}(r_i)` is the proportion of features from :math:`F` recovered in the top :math:`r_i`-fraction of the ranked list for observation :math:`i`

    %(notest)s

    %(params)s
    n_up
        Number of features to include in the AUC calculation.
        If ``None``, the top 5% of features based on their magnitude are selected.

    %(returns)s

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        adata, net = dc.ds.toy()
        rsc.dcg.aucell(adata, net, tmin=3)
    """
    nobs, nvar = mat.shape
    nsrc = starts.size
    n_up = _validate_n_up(nvar, n_up)
    m = f"aucell - calculating {nsrc} AUCs for {nvar} targets across {nobs} observations, categorizing features at rank={n_up}"
    _log(m, level="info", verbose=verbose)
    k = cp.minimum(offsets, n_up - 1)
    max_aucs = (k * (k - 1) / 2 + (n_up - k) * k).astype(cp.int32)
    es = _auc(
        row=mat,
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        n_up=n_up,
        n_fsets=nsrc,
        max_aucs=max_aucs,
    )
    return es.get(), None


_aucell = MethodMeta(
    name="aucell",
    desc="AUCell",
    func=_func_aucell,
    stype="categorical",
    adj=False,
    weight=False,
    test=False,
    limits=(0, 1),
    reference="https://doi.org/10.1038/nmeth.4463",
)
aucell = Method(_method=_aucell)
