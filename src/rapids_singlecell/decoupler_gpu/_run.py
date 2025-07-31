from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.stats as sts
from anndata import AnnData
from decoupler._log import _log
from decoupler.pp.data import extract
from decoupler.pp.net import adjmat, idxmat, prune
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from decoupler._datatype import DataType


def _return(
    name: str,
    data: DataType,
    es: pd.DataFrame,
    pv: pd.DataFrame,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | AnnData | None:
    if isinstance(data, AnnData):
        if data.obs_names.size != es.index.size:
            m = "Provided AnnData contains empty observations, returning repaired object"
            _log(m, level="warn", verbose=verbose)
            data = data[es.index, :].copy()
            data.obsm[f"score_{name}"] = es
            if pv is not None:
                data.obsm[f"padj_{name}"] = pv
            return data
        else:
            data.obsm[f"score_{name}"] = es
            if pv is not None:
                data.obsm[f"padj_{name}"] = pv
            return None
    else:
        return es, pv


def _run(
    name: str,
    func: Callable,
    *,
    adj: bool,
    test: bool,
    data: DataType,
    net: pd.DataFrame,
    tmin: int | float = 5,
    layer: str | None = None,
    raw: bool = False,
    empty: bool = True,
    bsize: int | float = 250_000,
    verbose: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame] | AnnData | None:
    _log(f"{name} - Running {name}", level="info", verbose=verbose)
    # Process data
    mat, obs, var = extract(
        data, layer=layer, raw=raw, empty=empty, verbose=verbose, bsize=bsize
    )
    issparse = sps.issparse(mat)
    isbacked = isinstance(mat, tuple)
    # Process net
    net = prune(features=var, net=net, tmin=tmin, verbose=verbose)
    # Handle stat type
    if adj:
        sources, targets, adjm = adjmat(features=var, net=net, verbose=verbose)
        # Handle batches
        if issparse or isbacked:
            nbatch = int(np.ceil(obs.size / bsize))
            es, pv = [], []
            for i in tqdm(range(nbatch), disable=not verbose):
                if i == 0 and verbose:
                    batch_verbose = True
                else:
                    batch_verbose = False
                srt, end = i * bsize, i * bsize + bsize
                if sps.issparse(mat):
                    bmat = mat[srt:end].toarray()
                else:
                    bmat, msk_col = mat
                    bmat = bmat[srt:end, :]
                    if sps.issparse(bmat):
                        bmat = bmat.toarray()
                    bmat = bmat[:, msk_col]
                bes, bpv = func(bmat, adjm, verbose=batch_verbose, **kwargs)
                es.append(bes)
                pv.append(bpv)
            es = np.vstack(es)
            es = pd.DataFrame(es, index=obs, columns=sources)
        else:
            es, pv = func(mat, adjm, verbose=verbose, **kwargs)
            es = pd.DataFrame(es, index=obs, columns=sources)
    else:
        sources, cnct, starts, offsets = idxmat(features=var, net=net, verbose=verbose)
        if isbacked:
            nbatch = int(np.ceil(obs.size / bsize))
            es, pv = [], []
            for i in tqdm(range(nbatch), disable=not verbose):
                if i == 0 and verbose:
                    batch_verbose = True
                else:
                    batch_verbose = False
                srt, end = i * bsize, i * bsize + bsize
                bmat, msk_col = mat
                bmat = bmat[srt:end, msk_col]
                bes, bpv = func(
                    bmat, cnct, starts, offsets, verbose=batch_verbose, **kwargs
                )
                es.append(bes)
                pv.append(bpv)
            es = np.vstack(es)
        else:
            es, pv = func(mat, cnct, starts, offsets, verbose=verbose, **kwargs)
        es = pd.DataFrame(es, index=obs, columns=sources)
    # Handle pvals and FDR correction
    if test:
        pv = np.vstack(pv)
        pv = pd.DataFrame(pv, index=obs, columns=sources)
        if name != "mlm":
            _log(f"{name} - adjusting p-values by FDR", level="info", verbose=verbose)
            pv.loc[:, :] = sts.false_discovery_control(pv.values, axis=1, method="bh")
    else:
        pv = None
    _log(f"{name} - done", level="info", verbose=verbose)
    return _return(name, data, es, pv, verbose=verbose)
