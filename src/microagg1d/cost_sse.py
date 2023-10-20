import numpy as np
from numba import float64, int64, njit
from numba.experimental import jitclass

from microagg1d.common import _calc_objective, calc_cumsum

USE_CACHE = True


@njit([(float64[:], float64[:], int64, int64)], cache=USE_CACHE)
def calc_objective_upper_exclusive(cumsum, cumsum2, i, j):
    """Compute the cluster cost of clustering points including i excluding j"""
    if j <= i:
        return 0.0
    return _calc_objective(cumsum[j] - cumsum[i], cumsum2[j] - cumsum2[i], j - i)


#    mu = (cumsum[j]-cumsum[i])/(j-i)
#    result = cumsum2[j] - cumsum2[i]
#    result += (j - i) * (mu * mu)
#    result -= (2 * mu) * (cumsum[j] - cumsum[i])
#    return max(result, 0)


@njit([(float64[:], float64[:], int64, int64)], cache=USE_CACHE)
def calc_objective_upper_inclusive(cumsum, cumsum2, i, j):
    """Compute the cluster cost of clustering points including both i and j"""
    if j <= i:
        return 0.0
    return _calc_objective(
        cumsum[j + 1] - cumsum[i], cumsum2[j + 1] - cumsum2[i], j + 1 - i
    )
    # mu = (cumsum[j + 1]-cumsum[i])/(j + 1-i)
    # result = cumsum2[j + 1] - cumsum2[i]
    # result += (j - i + 1) * (mu * mu)
    # result -= (2 * mu) * (cumsum[j + 1] - cumsum[i])
    # return max(result, 0)


@njit([(float64[:], int64)], cache=True)
def calc_cumsum_cell(v, cell_size):
    """Computes cumsums in cells of size cell_size
    instead of cumsum([1,2,3,4]) = [0,1,3,6,10]
    cumsum_cell([1,2,3,4], 2) = [[0,1,3], [0, 3, 7]]
    This has a numeric advantage as the numbers stored grow slower
    """
    quotient, remainder = divmod(len(v), cell_size)
    num_cells = quotient
    if remainder != 0:
        num_cells += 1
    out = np.zeros((num_cells, cell_size + 1), dtype=np.float64)

    for i in range(quotient):
        curr_out = out[i, :]
        curr_out[0] = 0.0
        offset = i * cell_size
        for j in range(cell_size):
            x = v[offset + j]
            curr_out[j + 1] = curr_out[j] + x
    if remainder != 0:
        i = quotient
        curr_out = out[i, :]
        curr_out[0] = 0.0
        offset = i * cell_size
        for j in range(remainder):
            x = v[offset + j]
            curr_out[j + 1] = curr_out[j] + x
    return out


@njit([(float64, float64, float64, float64, int64, int64, int64)], cache=USE_CACHE)
def calc_objective_upper_inclusive_2(
    i_cumsum, i_cumsum2, j_cumsum, j_cumsum2, i, j, cell_size
):
    """Compute the cluster cost of clustering points including both i and j across two cells"""
    val1 = j_cumsum + i_cumsum
    # print("\t", val1)
    mu = (val1) / (j + 1 + cell_size - i)
    result = j_cumsum2 + i_cumsum2
    # print("\t", result)
    result -= (2 * mu) * val1
    result += (j - i + 1 + cell_size) * (mu * mu)
    # print("\t", result)
    return max(result, 0.0)


@njit([(float64[:, :], float64[:, :], int64, int64, int64)], cache=USE_CACHE)
def calc_objective_cell(cumsum, cumsum2, cell_size, i, j):
    # assert j>=i
    # assert j - i < 2 * cell_size

    cell_i, remainder_i = divmod(i, cell_size)
    cell_j, remainder_j = divmod(j, cell_size)
    if cell_i == cell_j:  # both are in one cell
        return calc_objective_upper_inclusive(
            cumsum[cell_i, :], cumsum2[cell_i, :], remainder_i, remainder_j
        )
    else:
        return calc_objective_upper_inclusive_2(
            cumsum[cell_i, cell_size] - cumsum[cell_i, remainder_i],
            cumsum2[cell_i, cell_size] - cumsum2[cell_i, remainder_i],
            cumsum[cell_j, remainder_j + 1],
            cumsum2[cell_j, remainder_j + 1],
            remainder_i,
            remainder_j,
            cell_size,
        )


@jitclass(
    [
        ("cumsum", float64[:]),
        ("cumsum2", float64[:]),
        ("k", int64),
        ("F_vals", float64[:]),
        ("SMALL_VAL", float64),
        ("LARGE_VAL", float64),
    ]
)
class AdaptedSSECostCalculator:
    """The standard calculator the microaggregation-adapted SSE cost"""

    def __init__(self, cumsum, cumsum2, k, F_vals):
        self.cumsum = cumsum
        self.cumsum2 = cumsum2
        self.k = k
        self.F_vals = F_vals  # F_vals[i] is min_l w_li
        n = len(cumsum) - 1
        self.SMALL_VAL = calc_objective_upper_exclusive(cumsum, cumsum2, 0, n)
        self.LARGE_VAL = self.SMALL_VAL * (1 + n)

    def calc(self, i, j):
        """This function computes the w_{ij} values introduced"""
        if j <= i:
            return np.inf

        if not (j - i >= self.k):
            return self.LARGE_VAL + self.SMALL_VAL * i
        if not (j - i <= 2 * self.k - 1):
            return self.LARGE_VAL - self.SMALL_VAL * i
        return (
            calc_objective_upper_exclusive(self.cumsum, self.cumsum2, i, j)
            + self.F_vals[i]
        )


@jitclass(
    [
        ("cumsum", float64[:, :]),
        ("cumsum2", float64[:, :]),
        ("k", int64),
        ("F_vals", float64[:]),
        ("SMALL_VAL", float64),
        ("LARGE_VAL", float64),
        ("cell_size", int64),
    ]
)
class StableAdaptedSSECostCalculator:
    """A stable variant of the calculator for microaggregation-adapted SSE cost"""

    def __init__(self, x, k, F_vals, cell_size):
        self.cumsum = calc_cumsum_cell(x, cell_size)
        x_square = np.square(x)
        self.cumsum2 = calc_cumsum_cell(x_square, cell_size)
        self.k = k
        self.F_vals = F_vals
        n = len(x)
        self.SMALL_VAL = _calc_objective(np.sum(x), np.sum(x_square), n)
        self.LARGE_VAL = self.SMALL_VAL * (1 + n)
        self.cell_size = cell_size

    def calc(self, i, j):
        if j <= i:
            return np.inf

        if not (j - i >= self.k):
            return self.LARGE_VAL + self.SMALL_VAL * i
        if not (j - i <= 2 * self.k - 1):
            return self.LARGE_VAL - self.SMALL_VAL * i

        return (
            calc_objective_cell(self.cumsum, self.cumsum2, self.cell_size, i, j - 1)
            + self.F_vals[i]
        )


@jitclass(
    [
        ("cumsum", float64[:]),
        ("k", int64),
        ("F_vals", float64[:]),
        ("SMALL_VAL", float64),
        ("LARGE_VAL", float64),
    ]
)
class FasterAdaptedSSECostCalculator:
    """The faster calculator for wilbers method
    Instead of the "normal" cost function this uses the faster to compute cost function
    """

    def __init__(self, cumsum, k, F_vals):
        self.cumsum = cumsum
        self.k = k
        self.F_vals = F_vals
        n = len(cumsum) - 1
        x_bar = (cumsum[n] - cumsum[0]) / n
        self.SMALL_VAL = (n + 1) * x_bar * x_bar
        self.LARGE_VAL = self.SMALL_VAL * (1 + n)

    def calc(self, i, j):
        if j <= i:
            return np.inf

        if not (j - i >= self.k):
            return self.LARGE_VAL + self.SMALL_VAL * i
        if not (j - i <= 2 * self.k - 1):
            return self.LARGE_VAL - self.SMALL_VAL * i
        n = j - i
        x_bar = (self.cumsum[j] - self.cumsum[i]) / n
        return -n * x_bar * x_bar + self.F_vals[i]


@jitclass([("cumsum", float64[:])])
class FasterSSECostCalculator:
    def __init__(self, v):
        self.cumsum = calc_cumsum(v)

    def calc(self, i, j):
        n = j - i
        delta = self.cumsum[j] - self.cumsum[i]
        return -delta * delta / n


@jitclass([("cumsum", float64[:]), ("cumsum2", float64[:]), ("k", int64)])
class RestrictedCalculator:
    def __init__(self, v, k):
        self.cumsum = calc_cumsum(v)
        self.cumsum2 = calc_cumsum(np.square(v))
        self.k = k

    def calc(self, i, j):
        # print(i, j)
        if not (j - i >= self.k):
            # print("A", i, j, self.k)
            return np.inf
        if not (j - i <= 2 * self.k - 1):
            # print("B", i, j, self.k)
            return np.inf
        # print("C", i, j)
        return calc_objective_upper_exclusive(self.cumsum, self.cumsum2, i, j)


@njit
def sse_stable(v):
    mean = 0
    for x in v:
        mean += x
    mean /= len(v)
    s = 0
    for x in v:
        s += (x - mean) ** 2
    return s


@njit
def sse_from_mean_stable(v, mean):
    s = 0
    for x in v:
        s += (x - mean) ** 2
    return s


@njit(cache=True)
def compute_sse_sorted_stable(v, clusters_sorted):
    s = 0.0
    left = 0
    r = 0
    while r < len(v):
        r = left
        while clusters_sorted[left] == clusters_sorted[r] and r < len(v):
            r += 1
        # r-=1
        mean = np.mean(v[left:r])
        sse = sse_from_mean_stable(v[left:r], mean)
        s += sse
        left = r
    return s


@jitclass(
    [
        ("v", float64[:]),
    ]
)
class NoPrecomputeSSECostCalculator:
    def __init__(self, v):
        self.v = v

    def calc(self, i, j):
        return sse_stable(self.v[i:j])
        # mean = np.mean(self.v[i:j+1])
        # return np.sum(np.square(self.v[i:j+1]-mean))


@jitclass([("cumsum", float64[:]), ("cumsum2", float64[:])])
class SSECostCalculator:
    def __init__(self, v):
        self.cumsum = calc_cumsum(v)
        self.cumsum2 = calc_cumsum(np.square(v))

    def calc(self, i, j):
        if i >= j:
            return np.inf
        return calc_objective_upper_exclusive(self.cumsum, self.cumsum2, i, j)


@jitclass([("cumsum", float64[:, :]), ("cumsum2", float64[:, :]), ("cell_size", int64)])
class StableSSECostCalculator:
    def __init__(self, v, cell_size):
        self.cumsum = calc_cumsum_cell(v, cell_size)
        self.cumsum2 = calc_cumsum_cell(np.square(v), cell_size)
        self.cell_size = cell_size

    def calc(self, i, j):
        if j == i:
            return 0
        return calc_objective_cell(self.cumsum, self.cumsum2, self.cell_size, i, j - 1)
