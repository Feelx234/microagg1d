import numpy as np
from numba import njit

from microagg1d.cost_sse import compute_sse_sorted_stable


def _create_sol(n, k, d):
    solution = np.empty(n, np.int32)
    arange_len, remainder = divmod(n, k)
    if remainder == 0:
        solution[:] = np.repeat(np.arange(arange_len, dtype=np.int32), k)
    solution[:-remainder] = np.repeat(np.arange(arange_len, dtype=np.int32), k)
    total = remainder + d * k
    if total > n:
        assert False

    left = -total
    for i in range(d - 1):
        delta = abs(left) // (d - i)
        solution[left : left + delta] = arange_len + i - d  # arange_len-i-1
        left = left + delta
    solution[left:] = arange_len - 1
    return solution


def solution_for_arange(n, k):
    if k > n // 2:
        return np.zeros(n, dtype=np.int32)
    arange_len, remainder = divmod(n, k)
    arr = np.arange(n, dtype=np.float64)
    solution = np.empty(n, np.int32)
    if remainder == 0:
        solution[:] = np.repeat(np.arange(arange_len, dtype=np.int32), k)
    elif remainder < k:
        solution[:-remainder] = np.repeat(np.arange(arange_len, dtype=np.int32), k)
        d_max = 2 * remainder
        costs = np.inf * np.ones(d_max, dtype=np.float64)
        for d in range(2, d_max):
            total = remainder + d * k
            if total > n:
                break

            left = -total
            for i in range(d - 1):
                delta = abs(left) // (d - i)
                # print("LR", l, l+delta)
                solution[left : left + delta] = arange_len + i - d
                left = left + delta
            solution[left:] = arange_len - 1
            costs[d] = compute_sse_sorted_stable(
                arr[-remainder + d * k :], solution[-remainder + d * k :]
            )

        d_opt = np.argmin(costs)
        solution = _create_sol(n, k, d_opt)
    else:
        raise NotImplementedError()
    return solution


def create_pair_arange(n, k):
    arr = np.arange(n, dtype=np.float64)
    solution = solution_for_arange(n, k)
    return arr, solution


@njit(cache=True)
def _apply_epsilon(arr, solution, epsilon):
    assert len(arr) > 0
    d = -1
    prev_sol = solution[0]
    for i in range(len(arr)):  # pylint:disable=consider-using-enumerate
        if epsilon * d >= 1:
            raise ValueError("epsilon is to large")
        if solution[i] == prev_sol:
            d += 1
        else:
            d = 0
            prev_sol = solution[i]
        arr[i] += epsilon * d


def create_pair_known_sizes(sizes, k, epsilon=0):
    if (min_size := np.min(sizes)) < k:
        raise ValueError(f"Min of sizes is {min_size} but k is {k}. Need min_size >= k")

    if ((max_size := np.max(sizes)) + 1) * epsilon >= 1:
        raise ValueError(
            f"epsilon ({epsilon}) is to large because max(size)={max_size} and we need (max(size)+1)*epsilon < 1"
        )

    arr = np.repeat(np.arange(len(sizes), dtype=np.float64), sizes)
    solution = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)

    if epsilon > 0:
        _apply_epsilon(arr, solution, epsilon)

    return arr, solution


def create_pair_const_size(n, c, epsilon):
    num_full_clusters, R = divmod(n, c)

    sizes = np.full(num_full_clusters, c, dtype=np.int32)
    sizes[-1] += R
    return create_pair_known_sizes(sizes, c, epsilon)
