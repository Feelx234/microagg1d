import numpy as np
from numba import bool_, float64, int64, njit
from numba.experimental import jitclass

from microagg1d.algorithms_old import __wilber
from microagg1d.common import calc_cumsum, convert_implicit_to_explicit_clustering
from microagg1d.cost_sse import RestrictedCalculator, calc_objective_upper_inclusive

USE_CACHE = True


@jitclass(
    [
        ("cumsum", float64[:]),
        ("cumsum2", float64[:]),
        ("k", int64),
        ("F_vals", float64[:]),
        ("G", float64[:, :]),
        ("SMALL_VAL", float64),
        ("LARGE_VAL", float64),
    ]
)
class MicroaggWilberCalculator_edu:
    """An educational variant of the microagg calculator which keeps track of all the states visited in matrix G"""

    def __init__(self, cumsum, cumsum2, k, F_vals):
        self.cumsum = cumsum
        self.cumsum2 = cumsum2
        self.k = k
        self.F_vals = F_vals
        n = len(cumsum) - 1
        self.G = np.empty((n, n))
        self.SMALL_VAL = calc_objective_upper_inclusive(cumsum, cumsum2, 0, n - 1) + 1
        self.LARGE_VAL = self.SMALL_VAL * (1 + n)

    def calc(self, j, i):  # i <-> j interchanged is not a bug!
        # print(j, i)
        if j < i:
            self.G[i, j] = np.inf
            # print(i, j, np.inf)
            return np.inf

        if not (j + 1 - i >= self.k):
            # print("A", i, j, self.LARGE_VAL + self.SMALL_VAL*i)
            self.G[i, j] = self.LARGE_VAL + self.SMALL_VAL * i
            return self.LARGE_VAL + self.SMALL_VAL * i
        if not (j + 1 - i <= 2 * self.k - 1):
            # print("B", i, j)
            self.G[i, j] = self.LARGE_VAL - self.SMALL_VAL * i
            return self.LARGE_VAL - self.SMALL_VAL * i
        # if self.F_vals[i] >= self.SMALL_VAL: # bogus value
        #    #print("C", i, j, self.LARGE_VAL + self.SMALL_VAL*i)
        #    if j > i:
        #        self.G[i,j] = self.LARGE_VAL +  self.SMALL_VAL*i
        #    return self.LARGE_VAL + self.SMALL_VAL*i
        # print(i, j, self.calculator.calc(i, j) + self.F_vals[i])
        self.G[i, j] = (
            calc_objective_upper_inclusive(self.cumsum, self.cumsum2, i, j)
            + self.F_vals[i]
        )
        # print(" ", i, j, calc_objective_1(self.cumsum, self.cumsum2, i, j) + self.F_vals[i])
        return (
            calc_objective_upper_inclusive(self.cumsum, self.cumsum2, i, j)
            + self.F_vals[i]
        )


def wilber_edu(v, k, should_print=True):
    result, G = _wilber_edu(v, k)
    if should_print:
        with np.printoptions(linewidth=300, precision=3, suppress=True):
            print(G.T)
    return convert_implicit_to_explicit_clustering(result)


@njit([(float64[:], int64)], cache=USE_CACHE)
def _wilber_edu(v, k):
    n = len(v)
    cumsum = calc_cumsum(v)
    cumsum2 = calc_cumsum(np.square(v))
    wil_calculator = MicroaggWilberCalculator_edu(
        cumsum, cumsum2, k, np.empty(n + 1, dtype=np.float64)
    )

    result = __wilber(n, wil_calculator)
    return result, wil_calculator.G


@njit([(int64, float64[:], int64, bool_)], cache=True)
def _conventional_algorithm(n, vals, k, full):
    """Solves the univariate microaggregation problem in O(n^2)
    this is an implementation of the conventional algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    if n > 1000:
        raise ValueError(
            "Probably not intended to allocate such a large array, use other algorithm"
        )
    calculator = RestrictedCalculator(vals, k)
    g = np.zeros((n, n + 1))
    g[0, 0] = 0
    min_cost = np.empty(n + 1)
    min_cost[0] = 0
    best_pred = np.zeros(n, dtype=np.int32)
    for col in range(1, n + 1):
        lb = 0
        ub = col
        if not full:
            lb = max(col - 2 * k + 1, 0)
            ub = max(col - k + 1, 0)

        for row in range(lb, ub):
            # print(i, j, calculator.calc(i, j))
            g[row, col] = min_cost[row] + calculator.calc(row, col)
        if lb == ub:
            best_pred[col - 1] = 0
            min_cost[col] = np.inf
        else:
            # print(g)
            best_pred[col - 1] = np.argmin(g[lb:ub, col]) + lb
            # print(j,  F[j-1])
            # print()
            min_cost[col] = g[best_pred[col - 1], col]

    return best_pred, g


def conventional_algorithm(vals, k: int, full: bool = False, should_print: bool = True):
    """Solves the univariate microaggregation problem in O(n^2)
    this is an implementation of the conventional algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    n = len(vals)
    F, g = _conventional_algorithm(n, vals, k, full)  # pylint: disable=unused-variable
    if should_print:
        with np.printoptions(linewidth=200, precision=3, suppress=True):
            print(g[:, 1:].T)
    return convert_implicit_to_explicit_clustering(F)
