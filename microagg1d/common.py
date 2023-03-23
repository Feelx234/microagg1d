import numpy as np
from numba import njit, float64, int64


USE_CACHE=True

@njit([(float64[:],)], cache=USE_CACHE)
def calc_cumsum(v):
    cumsum = np.empty(len(v)+1, dtype=np.float64)
    cumsum[0]=0
    cumsum[1:] = np.cumsum(v)
    return cumsum

@njit([(float64[:],)], cache=USE_CACHE)
def calc_cumsum2(v):
    cumsum2 = np.empty(len(v)+1, dtype=np.float64)
    cumsum2[0]=0
    cumsum2[1:] = np.cumsum(np.square(v))
    return cumsum2



@njit([(float64[:], float64[:], int64, int64)], cache=USE_CACHE)
def calc_objective_upper_exclusive(cumsum, cumsum2, i, j):
    """Compute the cluster cost of clustering points including i excluding j"""
    if j <= i:
        return 0.0
    mu = (cumsum[j]-cumsum[i])/(j-i)
    result = cumsum2[j] - cumsum2[i]
    result += (j - i) * (mu * mu)
    result -= (2 * mu) * (cumsum[j] - cumsum[i])
    return max(result, 0)


@njit([(float64[:], float64[:], int64, int64)], cache=USE_CACHE)
def calc_objective_upper_inclusive(cumsum, cumsum2, i, j):
    """Compute the cluster cost of clustering points including both i and j"""
    if j <= i:
        return 0.0
    mu = (cumsum[j + 1]-cumsum[i])/(j + 1-i)
    result = cumsum2[j + 1] - cumsum2[i]
    result += (j - i + 1) * (mu * mu)
    result -= (2 * mu) * (cumsum[j + 1] - cumsum[i])
    return max(result, 0)
