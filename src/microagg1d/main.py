import numpy as np

from microagg1d.common import trivial_cases
from microagg1d.user_facing import (
    _maxdist_user,
    _rounddown_user,
    _roundup_user,
    _sae_user,
    _sse_galil_park2,
    _sse_simple_dynamic_program2,
    _sse_staggered2,
    _sse_wilber2,
)

USE_CACHE = True


def undo_argsort(sorted_arr, sort_order):
    """Puts the sorted_array which was sorted with sort_order back into the original order"""
    revert = np.empty_like(sort_order)
    revert[sort_order] = np.arange(len(sorted_arr))
    return sorted_arr[revert]


def univariate_microaggregation(x, k, method="auto", stable=1, cost="sse"):
    """Performs optimal 1d univariate microaggregation"""
    x = np.squeeze(np.asarray(x))
    assert len(x.shape) == 1, "provided array is not 1d"
    assert k > 0, f"negative or zero values for k({k}) are not supported"
    assert k <= len(
        x
    ), f"values of k({k}) larger than the length of the provided array ({len(x)}) are not supported"

    assert method in (
        "auto",
        "simple",
        "wilber",
        "galil_park",
        "staggered",
    ), "invalid method supplied"
    if method == "auto":
        if k <= 21:  # 21 determined empirically
            method = "simple"
        else:
            method = "staggered"

    order = np.argsort(x)
    x = np.array(x, dtype=np.float64)[order]

    is_trivial, trivial_result = trivial_cases(len(x), k)
    if is_trivial:
        return trivial_result

    if cost == "sse":
        if method == "simple":
            clusters = _sse_simple_dynamic_program2(x, k, stable=stable)
        elif method == "wilber":
            clusters = _sse_wilber2(x, k, stable=stable)
        elif method == "galil_park":
            clusters = _sse_galil_park2(x, k, stable=stable)
        elif method == "staggered":
            clusters = _sse_staggered2(x, k, stable=stable)
        else:
            raise NotImplementedError("Should not be reachable")
    elif cost == "sae":
        clusters = _sae_user(x, k, method)
    elif cost == "roundup":
        clusters = _roundup_user(x, k, method)
    elif cost == "rounddown":
        clusters = _rounddown_user(x, k, method)
    elif cost == "maxdist":
        clusters = _maxdist_user(x, k, method)
    else:
        raise NotImplementedError("Should not be reachable")
    return undo_argsort(clusters, order)
