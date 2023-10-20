import numpy as np
from numba import float64, int64, njit

USE_CACHE = True


@njit([(float64[:],)], cache=USE_CACHE)
def calc_cumsum(v):
    cumsum = np.empty(len(v) + 1, dtype=np.float64)
    cumsum[0] = 0
    cumsum[1:] = np.cumsum(v)
    return cumsum


@njit([(float64, float64, int64)], cache=USE_CACHE)
def _calc_objective(val1, val2, n):
    """Compute the cluster cost of clustering points including i excluding j"""
    result = val2
    result -= val1 / n * val1
    return max(result, 0)  # max to avoid some numerical issues


@njit(cache=USE_CACHE)
def calc_num_clusters_plus_one(result):
    """Compute the number of clusters encoded in results
    Can be used on e.g. the result of _conventional_algorithm, Wilber
    """
    num_clusters = 0
    curr_pos = len(result) - 1
    while result[curr_pos] > 0:
        curr_pos = result[curr_pos] - 1
        num_clusters += 1
    return num_clusters + 1


@njit(cache=USE_CACHE)
def convert_implicit_to_explicit_clustering(result):
    """Converts an implicit cluster assignment into an explicit assignment
    Example: the implicit assignment [0, 0, 1, 1, 2, 2]
    eactually indicates the clustering [0, 1, 2, 2, 2, 2] that is the first
    two points are their own cluster while the remaining four points are one cluster
    """
    num_clusters = calc_num_clusters_plus_one(result) - 1
    out = np.empty_like(result)
    curr_pos = len(result) - 1
    while result[curr_pos] > 0:
        # assign output the cluster values
        # we need curr_pos +1 as the square bracket operator is upper exclusive
        out[result[curr_pos] : curr_pos + 1] = num_clusters
        # adjust loop variables
        curr_pos = result[curr_pos] - 1
        num_clusters -= 1
    out[0 : curr_pos + 1] = num_clusters
    return out


def trivial_cases(n, k, dtype=np.int32):
    """Solves the trivial cases of univariate microaggregation
    The two trivial cases are
    1) only one cluster is possible (2k > n)
    2) there is only one way to soslve the problem with two clusters (2k == n)
    """
    assert k > 0
    assert k <= n
    if 2 * k > n:
        return True, np.zeros(n, dtype=dtype)
    if 2 * k == n:
        out = np.empty(n, dtype=dtype)
        out[:k] = 0
        out[k:] = 1
        return True, out
    return False, np.empty(0, dtype=dtype)


@njit
def compute_cluster_cost_sorted(clusters_sorted, calculator):
    """Given an explicit cluster assignment compute the cluster cost"""
    s = 0.0
    i = 0
    j = 1
    n = len(clusters_sorted)
    while j < n:
        while j < n and clusters_sorted[j] == clusters_sorted[i]:
            j += 1
        s += calculator.calc(i, j)
        i = j
        j = i + 1
    return s
