import numpy as np
from numba import float64, int64, njit, types

from microagg1d.algorithms_linear import __galil_park2, __staggered2, __wilber2
from microagg1d.algorithms_other import (
    __simple_dynamic_program,
    __simple_dynamic_program2,
)
from microagg1d.common import calc_cumsum, convert_implicit_to_explicit_clustering
from microagg1d.cost_maxdist import AdaptedMaxDistCostCalculator, MaxDistCostCalculator
from microagg1d.cost_round import (
    AdaptedRoundDownCostCalculator,
    AdaptedRoundUpCostCalculator,
    RoundDownCostCalculator,
    RoundUpCostCalculator,
)
from microagg1d.cost_sae import AdaptedSAECostCalculator, SAECostCalculator
from microagg1d.cost_sse import (
    AdaptedSSECostCalculator,
    FasterAdaptedSSECostCalculator,
    FasterSSECostCalculator,
    NoPrecomputeSSECostCalculator,
    SSECostCalculator,
    StableAdaptedSSECostCalculator,
    StableSSECostCalculator,
)

USE_CACHE = True


@njit([(float64[:], int64, int64)], cache=USE_CACHE)
def _sse_staggered2(v, k, stable=1):
    """Computes the univariate microaggregation of data vector v and min size k
    different choices of stable choose different methods how the costs are computed
    """
    method = __staggered2
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if stable == 2:
        cost_calculator = FasterAdaptedSSECostCalculator(
            calc_cumsum(v), k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator, k))
    if stable == 1:
        cost_calculator = StableAdaptedSSECostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64), 3 * k
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator, k))
    elif stable == 0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        cost_calculator = AdaptedSSECostCalculator(
            cumsum, cumsum2, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(method(n, cost_calculator, k))
    else:
        raise NotImplementedError("Only stable in (0,1) supported")


@njit([(float64[:], int64, int64)], cache=USE_CACHE)
def _sse_galil_park2(v, k, stable=1):
    method = __galil_park2
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
def _sse_wilber2(v, k, stable=1):
    method = __wilber2
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


@njit(cache=False)
def _sse_simple_dynamic_program(x, k, stable=1):
    n = len(x)
    assert k > 0
    if n // 2 < k:  # there can only be one cluster
        return np.zeros(n, dtype=np.int64)
    if k == 1:  # each node has its own cluster
        return np.arange(n)

    n = len(x)
    assert k > 0
    if n // 2 < k:  # there can only be one cluster
        return np.zeros(n, dtype=np.int64)
    if k == 1:  # each node has its own cluster
        return np.arange(n)
    if stable == 3:
        calculator = NoPrecomputeSSECostCalculator(x)
        return __simple_dynamic_program(n, k, calculator)
    if stable == 2:
        calculator = SSECostCalculator(x)
        return __simple_dynamic_program(n, k, calculator)
    if stable == 1:
        calculator = StableSSECostCalculator(x, k)
        return __simple_dynamic_program(n, k, calculator)
    elif stable == 0:
        calculator = FasterSSECostCalculator(x)
        return __simple_dynamic_program(n, k, calculator)
    else:
        assert False


@njit(cache=USE_CACHE)
def _sse_simple_dynamic_program2(x, k, stable=1):
    n = len(x)
    assert k > 0
    if n // 2 < k:  # there can only be one cluster
        return np.zeros(n, dtype=np.int64)
    if k == 1:  # each node has its own cluster
        return np.arange(n)
    if stable == 3:
        calculator = NoPrecomputeSSECostCalculator(x)
        return __simple_dynamic_program2(n, k, calculator)
    if stable == 2:
        calculator = SSECostCalculator(x)
        return __simple_dynamic_program2(n, k, calculator)
    if stable == 1:
        calculator = StableSSECostCalculator(x, k)
        return __simple_dynamic_program2(n, k, calculator)
    elif stable == 0:
        calculator = FasterSSECostCalculator(x)
        return __simple_dynamic_program2(n, k, calculator)
    else:
        assert False


@njit([(float64[:], int64, types.unicode_type)], cache=USE_CACHE)
def _sae_user(v, k, algorithm):
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if algorithm == "galil_park":
        cost_calculator = AdaptedSAECostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(
            __galil_park2(n, cost_calculator)
        )
    elif algorithm == "wilber":
        cost_calculator = AdaptedSAECostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(__wilber2(n, cost_calculator))
    elif algorithm == "staggered":
        cost_calculator = AdaptedSAECostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(
            __staggered2(n, cost_calculator, k)
        )
    elif algorithm == "simple":
        cost_calculator = SAECostCalculator(v)
        return __simple_dynamic_program2(n, k, cost_calculator)
    else:
        raise NotImplementedError("Wrong algorithm string provided")


@njit([(float64[:], int64, types.unicode_type)], cache=USE_CACHE)
def _roundup_user(v, k, algorithm):
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if algorithm == "galil_park":
        cost_calculator = AdaptedRoundUpCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(
            __galil_park2(n, cost_calculator)
        )
    elif algorithm == "wilber":
        cost_calculator = AdaptedRoundUpCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(__wilber2(n, cost_calculator))
    elif algorithm == "staggered":
        cost_calculator = AdaptedRoundUpCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(
            __staggered2(n, cost_calculator, k)
        )
    elif algorithm == "simple":
        cost_calculator = RoundUpCostCalculator(v)
        return __simple_dynamic_program2(n, k, cost_calculator)
    else:
        raise NotImplementedError("Wrong algorithm string provided")


@njit([(float64[:], int64, types.unicode_type)], cache=USE_CACHE)
def _rounddown_user(v, k, algorithm):
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if algorithm == "galil_park":
        cost_calculator = AdaptedRoundDownCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(
            __galil_park2(n, cost_calculator)
        )
    elif algorithm == "wilber":
        cost_calculator = AdaptedRoundDownCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(__wilber2(n, cost_calculator))
    elif algorithm == "staggered":
        cost_calculator = AdaptedRoundDownCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(
            __staggered2(n, cost_calculator, k)
        )
    elif algorithm == "simple":
        cost_calculator = RoundDownCostCalculator(v)
        return __simple_dynamic_program2(n, k, cost_calculator)
    else:
        raise NotImplementedError("Wrong algorithm string provided")


@njit([(float64[:], int64, types.unicode_type)], cache=USE_CACHE)
def _maxdist_user(v, k, algorithm):
    # unfortunately a lot of copy pasta as numba can't handle it yet
    n = len(v)
    if algorithm == "galil_park":
        cost_calculator = AdaptedMaxDistCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(
            __galil_park2(n, cost_calculator)
        )
    elif algorithm == "wilber":
        cost_calculator = AdaptedMaxDistCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(__wilber2(n, cost_calculator))
    elif algorithm == "staggered":
        cost_calculator = AdaptedMaxDistCostCalculator(
            v, k, np.empty(n + 1, dtype=np.float64)
        )
        return convert_implicit_to_explicit_clustering(
            __staggered2(n, cost_calculator, k)
        )
    elif algorithm == "simple":
        cost_calculator = MaxDistCostCalculator(v)
        return __simple_dynamic_program2(n, k, cost_calculator)
    else:
        raise NotImplementedError("Wrong algorithm string provided")
