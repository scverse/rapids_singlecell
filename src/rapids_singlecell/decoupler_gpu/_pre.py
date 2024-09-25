# adapted from decoupler/_pre.py
from __future__ import annotations

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import issparse as cp_issparse
from cupyx.scipy.special import betainc
from scanpy.get import _get_obs_rep
from scipy.sparse import csr_matrix, issparse

from rapids_singlecell.preprocessing._utils import _check_use_raw

getnnz_0 = cp.ElementwiseKernel(
    "int32 idx",
    "raw int32 sum",
    """
    atomicAdd(&sum[idx], 1);
    """,
    "get_nnz_0",
)


def __stdtr(df, t):
    x = df / (t**2 + df)
    tail = betainc(df / 2, 0.5, x) / 2
    return cp.where(t < 0, tail, 1 - tail)


def check_mat(m, r, c, verbose=False):
    # Accept any sparse format but transform to csr
    if issparse(m) and not isinstance(m, csr_matrix):
        m = csr_matrix(m)
    elif cp_issparse(m) and not isinstance(m, cp_csr_matrix):
        m = cp_csr_matrix(m)
    # Check for empty features
    if isinstance(m, csr_matrix):
        msk_features = m.getnnz(axis=0) == 0
    elif cp_issparse(m):
        msk_features = getnnz_0(m.indices, cp.zeros(m.shape[1], dtype=np.int32)) == 0
    elif isinstance(m, cp.ndarray):
        msk_features = cp.count_nonzero(m, axis=0) == 0
    else:
        msk_features = np.count_nonzero(m, axis=0) == 0

    n_empty_features = msk_features.sum()
    if n_empty_features > 0:
        if verbose:
            print(
                f"{n_empty_features} features of mat are empty, they will be removed."
            )
        if isinstance(msk_features, cp.ndarray):
            msk_features = msk_features.get()
        c = c[~msk_features]
        m = m[:, ~msk_features]

    # Check for repeated features
    if np.any(c[1:] == c[:-1]):
        raise ValueError(
            """mat contains repeated feature names, please make them unique."""
        )

    # Check for empty samples
    if type(m) is csr_matrix:
        msk_samples = m.getnnz(axis=1) == 0
    elif cp_issparse(m):
        msk_samples = cp.diff(m.indptr) == 0
    elif isinstance(m, cp.ndarray):
        msk_samples = cp.count_nonzero(m, axis=1) == 0
    else:
        msk_samples = np.count_nonzero(m, axis=1) == 0
    n_empty_samples = msk_samples.sum()
    if n_empty_samples > 0:
        if verbose:
            print(f"{n_empty_samples} samples of mat are empty, they will be removed.")
        if isinstance(msk_samples, cp.ndarray):
            msk_samples = msk_samples.get()
        r = r[~msk_samples]
        m = m[~msk_samples]

    # Check for non finite values
    check = False
    if isinstance(m, cp.ndarray):
        if np.any(~np.isfinite(m)):
            check = True
    else:
        if np.any(~np.isfinite(m.data)):
            check = True
    if check:
        raise ValueError(
            """mat contains non finite values (nan or inf), please set them to 0 or remove them."""
        )

    return m, r, c


def extract(
    mat, *, use_raw=None, layer=None, verbose=False, dtype=np.float32, pre_load=False
):
    """
    Processes different input types so that they can be used downstream.

    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [matrix, samples, features], dataframe (samples x features) or an AnnData instance.
    use_raw : bool
        Use `raw` attribute of `adata` if present.
    dtype : type
        Type of float used.

    Returns
    -------
    m : csr_matrix
        Sparse matrix containing molecular readouts or statistics.
    r : ndarray
        Array of sample names.
    c : ndarray
        Array of feature names.
    """

    if type(mat) is list:
        m, r, c = mat
        m = np.array(m, dtype=dtype)
        r = np.array(r, dtype="U")
        c = np.array(c, dtype="U")
    elif isinstance(mat, pd.DataFrame) or isinstance(mat, cudf.DataFrame):
        m = mat.values.astype(dtype)
        r = mat.index.to_numpy(dtype="U")
        c = mat.columns.to_numpy(dtype="U")
    elif type(mat) is AnnData:
        use_raw = _check_use_raw(mat, use_raw, layer)
        m = _get_obs_rep(mat, layer=layer, use_raw=use_raw)
        c = (
            mat.raw.var.index.values.astype("U")
            if use_raw
            else mat.var.index.values.astype("U")
        )
        r = mat.obs.index.values.astype("U")

    else:
        raise ValueError("""mat must be a list of [matrix, samples, features], dataframe (samples x features) or an AnnData
        instance.""")

    if pre_load is None:
        pre_load = False
    if pre_load:
        if issparse(m):
            m = cp_csr_matrix(m)

    if isinstance(m, np.ndarray):
        m = cp.array(m)

    if isinstance(r, cp.ndarray):
        r = cp.asnumpy(r)
    if isinstance(c, cp.ndarray):
        c = cp.asnumpy(c)

    # Check mat for empty or not finite values
    m, r, c = check_mat(m, r, c, verbose=verbose)

    return m, r, c


def match(c, r, net):
    """
    Matches `mat` with a regulatory adjacency matrix.

    Parameters
    ----------
    c : ndarray
        Column names of `mat`.
    r : ndarray
        Row  names of `net`.
    net : ndarray
        Regulatory adjacency matrix.

    Returns
    -------
    regX : ndarray
        Matching regulatory adjacency matrix.
    """

    # Init empty regX
    regX = np.zeros((c.shape[0], net.shape[1]), dtype=np.float32)

    # Create an index array for rows of c corresponding to r
    c_dict = {gene: i for i, gene in enumerate(c)}
    idxs = [c_dict[gene] for gene in r if gene in c_dict]

    # Populate regX using advanced indexing
    regX[idxs, :] = net[: len(idxs), :]

    return regX


def filt_min_n(c, net, min_n=5):
    """
    Removes sources of a `net` with less than min_n targets.

    First it filters target features in `net` that are not in `mat` and then removes sources with less than `min_n` targets.

    Parameters
    ----------
    c : ndarray
        Column names of `mat`.
    net : DataFrame
        Network in long format.
    min_n : int
        Minimum of targets per source. If less, sources are removed.

    Returns
    -------
    net : DataFrame
        Filtered net in long format.
    """

    # Find shared targets between mat and net
    msk = np.isin(net["target"].values.astype("U"), c)
    net = net.iloc[msk]

    # Count unique sources
    sources, counts = np.unique(net["source"].values.astype("U"), return_counts=True)

    # Find sources with more than min_n targets
    msk = np.isin(net["source"].values.astype("U"), sources[counts >= min_n])

    # Filter
    net = net[msk]

    if net.shape[0] == 0:
        raise ValueError(f"""No sources with more than min_n={min_n} targets. Make sure mat and net have shared target features or
        reduce the number assigned to min_n""")

    return net


def rename_net(net, source="source", target="target", weight="weight"):
    """
    Renames input network to match decoupler's format (source, target, weight).

    Parameters
    ----------
    net : DataFrame
        Network in long format.
    source : str
        Column name where to extract source features.
    target : str
        Column name where to extract target features.
    weight : str, None
        Column name where to extract features' weights. If no weights are available, set to None.

    Returns
    -------
    net : DataFrame
        Renamed network.
    """

    # Check if names are in columns
    msg = 'Column name "{0}" not found in net. Please specify a valid column.'
    assert source in net.columns, msg.format(source)
    assert target in net.columns, msg.format(target)
    if weight is not None:
        assert weight in net.columns, (
            msg.format(weight)
            + """Alternatively, set to None if no weights are available."""
        )
    else:
        net = net.copy()
        net["weight"] = 1.0
        weight = "weight"

    # Rename
    net = net.rename(columns={source: "source", target: "target", weight: "weight"})

    # Sort
    net = net.reindex(columns=["source", "target", "weight"])

    # Check if duplicated
    is_d = net.duplicated(["source", "target"]).sum()
    if is_d > 0:
        raise ValueError("net contains repeated edges, please remove them.")

    return net


def get_net_mat(net):
    """
    Transforms a given network to a regulatory adjacency matrix (targets x sources).

    Parameters
    ----------
    net : DataFrame
        Network in long format.

    Returns
    -------
    sources : ndarray
        Array of source names.
    targets : ndarray
        Array of target names.
    X : ndarray
        Array of interactions between sources and targets (target x source).
    """

    # Pivot df to a wider format
    X = net.pivot(columns="source", index="target", values="weight")
    X[np.isnan(X)] = 0

    # Store node names and weights
    sources = X.columns.values
    targets = X.index.values
    X = X.values

    return sources.astype("U"), targets.astype("U"), X.astype(np.float32)
