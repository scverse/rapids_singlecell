# adapted from decoupler/_pre.py
from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse


def check_mat(m, r, c, verbose=False):
    # Accept any sparse format but transform to csr
    if issparse(m) and not isinstance(m, csr_matrix):
        m = csr_matrix(m)

    # Check for empty features
    if type(m) is csr_matrix:
        msk_features = m.getnnz(axis=0) == 0
    else:
        msk_features = np.count_nonzero(m, axis=0) == 0
    n_empty_features = np.sum(msk_features)
    if n_empty_features > 0:
        if verbose:
            print(
                f"{n_empty_features} features of mat are empty, they will be removed."
            )
        c = c[~msk_features]
        m = m[:, ~msk_features]

    # Sort features
    # msk = np.argsort(c)
    # m, r, c = m[:, msk], r.astype('U'), c[msk].astype('U')
    # Check for repeated features
    if np.any(c[1:] == c[:-1]):
        raise ValueError(
            """mat contains repeated feature names, please make them unique."""
        )

    # Check for empty samples
    if type(m) is csr_matrix:
        msk_samples = m.getnnz(axis=1) == 0
    else:
        msk_samples = np.count_nonzero(m, axis=1) == 0
    n_empty_samples = np.sum(msk_samples)
    if n_empty_samples > 0:
        if verbose:
            print(f"{n_empty_samples} samples of mat are empty, they will be removed.")
        r = r[~msk_samples]
        m = m[~msk_samples]

    # Check for non finite values
    if np.any(~np.isfinite(m.data)):
        raise ValueError(
            """mat contains non finite values (nan or inf), please set them to 0 or remove them."""
        )

    return m, r, c


def extract(mat, use_raw=True, verbose=False, dtype=np.float32):
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
    elif type(mat) is pd.DataFrame:
        m = mat.values.astype(dtype)
        r = mat.index.values.astype("U")
        c = mat.columns.values.astype("U")
    elif type(mat) is AnnData:
        if use_raw:
            if mat.raw is None:
                raise ValueError("Received `use_raw=True`, but `mat.raw` is empty.")
            m = mat.raw.X.astype(dtype)
            c = mat.raw.var.index.values.astype("U")
        else:
            m = mat.X.astype(dtype)
            c = mat.var.index.values.astype("U")
        r = mat.obs.index.values.astype("U")

    else:
        raise ValueError("""mat must be a list of [matrix, samples, features], dataframe (samples x features) or an AnnData
        instance.""")

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
