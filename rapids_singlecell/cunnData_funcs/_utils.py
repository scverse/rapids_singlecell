import cupy as cp

def _get_mean_var(X):
    mean = (X.sum(axis=0) / X.shape[0]).flatten()
    mean_sq = cp.sparse.csr_matrix((X.data ** 2,X.indices, X.indptr)).sum(axis=0).flatten() / X.shape[0]
    var = (mean_sq - mean ** 2) * (X.shape[0] / (X.shape[0] - 1))
    return mean, var

def _check_nonnegative_integers(X):
    """Checks values of X to ensure it is count data"""
    data = X.data
    # Check no negatives
    if cp.signbit(data).any():
        return False
    elif cp.any(~cp.equal(cp.mod(data, 1), 0)):
        return False
    else:
        return True
    