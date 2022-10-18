import cupy as cp


def _get_mean_var(X):
    mean = (X.sum(axis =0)/X.shape[0]).ravel()
    X.data **= 2
    inter = (X.sum(axis =0)/X.shape[0]).ravel()
    var = inter - mean ** 2
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