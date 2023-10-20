import numpy as np
from numba import float64, int64, njit
from numba.experimental import jitclass


@njit
def calc_maxdist_from_array(arr, i, j):
    return arr[j - 1] - arr[i]


@jitclass(
    [
        ("arr", float64[:]),
        ("k", int64),
        ("F_vals", float64[:]),
        ("SMALL_VAL", float64),
        ("LARGE_VAL", float64),
    ]
)
class AdaptedMaxDistCostCalculator:
    """The standard calculator the microaggregation-adapted SSE cost"""

    def __init__(self, arr, k, F_vals):
        self.arr = arr
        self.k = k
        self.F_vals = F_vals  # F_vals[i] is min_l w_li
        n = len(arr) - 1
        # the largest cluster cost is 2 * n * the maximum difference
        self.SMALL_VAL = (arr[-1] - arr[0]) * n
        self.LARGE_VAL = self.SMALL_VAL * (1 + n)

    def calc(self, i, j):  # i <-> j interchanged is not a bug!
        """This function computes the w_{ij} values introduced"""
        if j <= i:
            return np.inf

        if not (j - i >= self.k):
            return self.LARGE_VAL + self.SMALL_VAL * i
        if not (j - i <= 2 * self.k - 1):
            return self.LARGE_VAL - self.SMALL_VAL * i
        return calc_maxdist_from_array(self.arr, i, j) + self.F_vals[i]


@jitclass([("arr", float64[:])])
class MaxDistCostCalculator:
    """The standard calculator the microaggregation-adapted SSE cost"""

    def __init__(self, arr):
        self.arr = arr

    def calc(self, i, j):
        """This function computes the w_{ij} values introduced"""
        if j <= i:
            return np.inf
        return calc_maxdist_from_array(self.arr, i, j)
