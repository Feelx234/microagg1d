import numpy as np
from numba import float64, int64, njit

from microagg1d.common import (
    calc_cumsum,
    convert_implicit_to_explicit_clustering,
    trivial_cases,
)
from microagg1d.cost_sse import (
    AdaptedSSECostCalculator,
    FasterAdaptedSSECostCalculator,
    StableAdaptedSSECostCalculator,
)
from microagg1d.smawk_old import _smawk_iter

USE_CACHE = True


@njit
def __wilber(n, wil_calculator):
    """Solves Univariate Microaggregation problem in O(n)
    this is an implementation of the proposed algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    F = np.empty(n, dtype=np.int32)
    F_vals = wil_calculator.F_vals
    H = np.empty(n, dtype=np.int32)
    H_vals = np.empty(n + 1, dtype=np.float64)
    F_vals[0] = 0
    c = 0  # columns [0,c] have correct F_vals
    r = 0  # rows [r,c] may contain column minima

    while c < n:
        p = min(2 * c - r + 1, n)
        # print("F_input", r, c+1, c, p)
        _smawk_iter(np.arange(c, p), np.arange(r, c + 1), wil_calculator, F)
        # print("F", F)
        for j in range(c, p):
            F_vals[j + 1] = wil_calculator.calc(j, F[j])

        # print("H", c+1, p, c+1,p)
        _smawk_iter(np.arange(c + 1, p), np.arange(c + 1, p), wil_calculator, H)
        for j in range(c + 1, p):
            H_vals[j + 1] = wil_calculator.calc(j, H[j])

        j0 = p + 1
        for j in range(c + 2, p + 1):
            if H_vals[j] < F_vals[j]:
                F[j - 1] = H[j - 1]
                j0 = j
                break
        if j0 == p + 1:  # we were right all along
            # F_vals up to p (inclusive) are correct
            r = F[p - 1]
            c = p
        else:  # our guessing strategy failed
            F_vals[j0] = H_vals[j0]
            r = c + 1
            c = j0

    return F


@njit
def __galil_park(n, wil_calculator):
    """Solves the dynamic problem in O(n)
    This is an implementation of the proposed algorithm
    from "A Linear-Time Algorithm for Concave One-Dimensional Dynamic Programming" by Zvi Galil and Kunsoo Park 1989
    """
    F = np.empty(n, dtype=np.int32)
    F_vals = wil_calculator.F_vals
    N = np.empty(n, dtype=np.int32)
    N_vals = np.inf * np.ones(n + 1, dtype=np.float64)
    F_vals[0] = 0
    c = 0  # columns [0,c] have correct F_vals
    r = 0  # rows [r,c] may contain column minima

    while c < n:
        p = min(2 * c - r + 1, n)
        _smawk_iter(np.arange(c, p), np.arange(r, c + 1), wil_calculator, F)
        for j in range(c, p):
            val = wil_calculator.calc(j, F[j])
            if val < N_vals[j + 1]:
                F_vals[j + 1] = val
            else:
                F_vals[j + 1] = N_vals[j + 1]
                F[j] = N[j]
        j0 = p + 2
        for j in range(c + 2, p + 1):
            if wil_calculator.calc(j - 1, j - 1) < F_vals[j]:
                # the H value considered was smaller, may not continue
                F[j - 1] = j - 1
                j0 = j
                F_vals[j0] = wil_calculator.calc(j - 1, j - 1)
                r = c
                c = j0
                break

            if F_vals[p] <= wil_calculator.calc(p - 1, j - 1):
                # we did just eliminate row j entries c:p
                # => may continue as usual
                pass
            else:
                # print("B")
                # need to break because it is not guaranteed that the following
                # F values, (F[j+1:]) are correct as well, they might lie in row j
                j0 = j
                # F[j-1]=j
                N[j - 1 : p + 1] = F[j - 1 : p + 1]
                N_vals[j - 1 : p] = F_vals[j - 1 : p]
                r = c + 1
                c = j0
                break
        if j0 == p + 2:
            # F_vals up to p (inclusive) are correct
            r = max(r, F[p - 1])
            c = p
        else:  # our guessing strategy failed
            pass
            # r = c
            # c = j0

    return F


def __execute_linear_internal(method, v, k, stable):
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if stable == 2:
        cost_calculator = FasterAdaptedSSECostCalculator(
            calc_cumsum(v), k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    if stable == 1:
        cost_calculator = StableAdaptedSSECostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64), 3 * k
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    elif stable == 0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        cost_calculator = AdaptedSSECostCalculator(
            cumsum, cumsum2, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    else:
        raise NotImplementedError("Only stable in (0,1) supported")


def execute_linear(method, arr, k: int, stable=1):
    assert k > 0
    assert k <= len(arr)
    valid, res = trivial_cases(len(arr), k)
    if valid:
        return res
    return __execute_linear_internal(method, arr, k, stable=stable)


# @njit([(float64[:], int64, int64)], cache=USE_CACHE)
def wilber(arr, k: int, stable=1):
    """Solves the dynamic problem in O(n)
    This is an implementation of the proposed algorithm
    from "The concave least weight subsequence problem revisited" by Robert wilber 1987
    """
    valid, res = trivial_cases(len(arr), k)
    if valid:
        return res
    return execute_linear(__wilber, arr, k, stable)


# @njit([(float64[:], int64, int64)], cache=USE_CACHE)
def galil_park(arr, k: int, stable=1):
    """Solves the dynamic problem in O(n)
    This is an implementation of the proposed algorithm
    from "A Linear-Time Algorithm for Concave One-Dimensional Dynamic Programming" by Zvi Galil and Kunsoo Park 1989
    """
    valid, res = trivial_cases(len(arr), k)
    if valid:
        return res
    return execute_linear(__galil_park, arr, k, stable)


@njit([(float64[:], int64, int64)], cache=USE_CACHE)
def _wilber(v, k, stable=1):
    method = __wilber
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if stable == 2:
        cost_calculator = FasterAdaptedSSECostCalculator(
            calc_cumsum(v), k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    if stable == 1:
        cost_calculator = StableAdaptedSSECostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64), 3 * k
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    elif stable == 0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        cost_calculator = AdaptedSSECostCalculator(
            cumsum, cumsum2, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    else:
        raise NotImplementedError("Only stable in (0,1) supported")


@njit([(float64[:], int64, int64)], cache=USE_CACHE)
def _galil_park(v, k, stable=1):
    method = __galil_park
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if stable == 2:
        cost_calculator = FasterAdaptedSSECostCalculator(
            calc_cumsum(v), k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    if stable == 1:
        cost_calculator = StableAdaptedSSECostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64), 3 * k
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    elif stable == 0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        cost_calculator = AdaptedSSECostCalculator(
            cumsum, cumsum2, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    else:
        raise NotImplementedError("Only stable in (0,1) supported")


def execute_linear2(method, arr, k: int, stable=1):
    assert k > 0
    assert k <= len(arr)
    res = trivial_cases(len(arr), k)
    if res is not None:
        return res
    return __execute_linear_internal2(method, arr, k, stable=stable)


@njit
def __execute_linear_internal2(method, v, k, stable):
    """This function should have been able to provide a unified way to execute a single algorithms on different
    cost functions, unfortunately numba doesn't allow that yet"""
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if stable == 2:
        cost_calculator = FasterAdaptedSSECostCalculator(
            calc_cumsum(v), k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    if stable == 1:
        cost_calculator = StableAdaptedSSECostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64), 3 * k
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    elif stable == 0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        cost_calculator = AdaptedSSECostCalculator(
            cumsum, cumsum2, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator))
    else:
        raise NotImplementedError("Only stable in (0,1) supported")
