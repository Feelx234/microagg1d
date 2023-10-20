import numpy as np
from numba import njit

from microagg1d.common import convert_implicit_to_explicit_clustering


@njit(cache=False)  # no caching as otherwise it would be recompiled often
def __simple_dynamic_program(n, k, calculator):
    """Solves the univariate microaggregation problem for the calculator
    Does NOT assume the cost function is concave
    """
    back_tracks = np.zeros(n, dtype=np.int64)
    min_vals = np.zeros(n)
    for i in range(0, k - 1):
        min_vals[i] = np.inf
        back_tracks[i] = -1
    for i in range(k - 1, 2 * k - 1):
        min_vals[i] = calculator.calc(0, i + 1)
        back_tracks[i] = -1

    for i in range(2 * k - 1, n):
        # print("i", i)
        min_index = i - 2 * k + 1
        # print("min", min_index)
        prev_min_val = min_vals[min_index] + calculator.calc(min_index + 1, i + 1)
        for j in range(i - 2 * k + 2, i - k + 1):
            # print(j, min_vals[j], prev_min_val)
            new_val = min_vals[j] + calculator.calc(j + 1, i + 1)
            if new_val < prev_min_val:
                min_index = j
                prev_min_val = new_val
        # print("result", min_index, prev_min_val)

        back_tracks[i] = min_index
        min_vals[i] = prev_min_val
        # print(back_tracks)
    return convert_implicit_to_explicit_clustering(back_tracks + 1)


@njit(cache=False)  # no caching as otherwise it would be recompiled often
def __simple_dynamic_program2(n, k, calculator):
    """Solves the univariate microaggregation problem for the calculator
    Assumes the cost function in calculator is CONCAVE
    """
    back_tracks = np.zeros(n, dtype=np.int64)
    min_vals = np.zeros(n)
    for i in range(0, k - 1):
        min_vals[i] = np.inf
        back_tracks[i] = -1
    for i in range(k - 1, min(2 * k - 1, n)):
        min_vals[i] = calculator.calc(0, i + 1)
        back_tracks[i] = -1

    prev_min_index = 0
    for right in range(2 * k - 1, n):  # right=i
        # print("i", i)
        min_index = right - 2 * k + 1
        # print("min", min_index)
        prev_min_val = min_vals[min_index] + calculator.calc(min_index + 1, right + 1)
        for left in range(max(right - 2 * k + 2, prev_min_index), right - k + 1):
            # print(j, min_vals[j], prev_min_val)
            new_val = min_vals[left] + calculator.calc(left + 1, right + 1)
            if new_val < prev_min_val:
                min_index = left
                prev_min_val = new_val
        # print("result", min_index, prev_min_val)

        back_tracks[right] = min_index
        prev_min_index = max(min_index, 0)
        min_vals[right] = prev_min_val
        # print(back_tracks)
    return convert_implicit_to_explicit_clustering(back_tracks + 1)
