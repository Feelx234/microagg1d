import numpy as np

from microagg1d._core import microaggregation_sorted

_METHODS = ("auto", "simple", "wilber", "galil_park", "staggered")
_COSTS = ("sse", "sae", "roundup", "rounddown", "maxdist")


def undo_argsort(sorted_arr, sort_order):
    """Puts the sorted_array which was sorted with sort_order back into the original order"""
    revert = np.empty_like(sort_order)
    revert[sort_order] = np.arange(len(sorted_arr))
    return sorted_arr[revert]


def univariate_microaggregation(x, k, method="auto", stable=1, cost="sse"):
    """Performs optimal 1d univariate microaggregation

    Parameters
    ----------
    x : array_like
        The 1d values to cluster.
    k : int
        Minimum cluster size.
    method : {"auto", "simple", "wilber", "galil_park", "staggered"}
        Algorithm used. "auto" picks "simple" for small and "staggered" for large k.
    stable : int
        How the SSE cost is evaluated: 0 fastest, 1 numerically stable (default),
        2 plain prefix sums (3: no precomputation, only for method "simple").
        Ignored for other costs.
    cost : {"sse", "sae", "roundup", "rounddown", "maxdist"}
        The cost function to minimize.

    Returns
    -------
    numpy.ndarray of int, the cluster label of every point.
    """
    x = np.squeeze(np.asarray(x))
    assert len(x.shape) == 1, "provided array is not 1d"
    assert k > 0, f"negative or zero values for k({k}) are not supported"
    assert k <= len(
        x
    ), f"values of k({k}) larger than the length of the provided array ({len(x)}) are not supported"
    assert method in _METHODS, "invalid method supplied"
    if cost not in _COSTS:
        raise NotImplementedError(f"cost {cost!r} is not supported")
    if method == "auto":
        method = "simple" if k <= 21 else "staggered"  # 21 determined empirically

    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=np.float64)[order]
    if cost == "sse":
        # SSE is translation invariant; if the offset dominates the spread, the prefix
        # sums suffer catastrophic cancellation, so center the data in that case
        # (not always, as centering changes rounding and thereby tie breaking)
        mean = x_sorted.mean()
        if abs(mean) > 10 * (x_sorted[-1] - x_sorted[0]):
            x_sorted = x_sorted - mean
    x_sorted = np.ascontiguousarray(x_sorted)
    clusters = microaggregation_sorted(x_sorted, int(k), method, cost, int(stable))
    return undo_argsort(clusters, order)
