import numpy as np
from numba import njit, float64, int64, bool_
from numba.experimental import jitclass
from microagg1d.smawk_iter import _smawk_iter
from microagg1d.common import calc_cumsum, calc_objective_upper_exclusive, calc_objective_upper_inclusive, calc_objective_cell, calc_cumsum_cell, _calc_objective
USE_CACHE=True

@njit(cache=USE_CACHE)
def calc_num_clusters_plus_one(result):
    """Compute the number of clusters encoded in results
    Can be used on e.g. the result of _conventional_algorithm, Weber
    """
    num_clusters = 0
    curr_pos = len(result)-1
    while result[curr_pos]>0:
        curr_pos = result[curr_pos]-1
        num_clusters+=1
    return num_clusters+1


@njit(cache=USE_CACHE)
def relabel_clusters_plus_one(result):
    num_clusters = calc_num_clusters_plus_one(result)-1
    out = np.empty_like(result)
    curr_pos = len(result)-1
    while result[curr_pos]>0:
        out[result[curr_pos]:curr_pos+1] = num_clusters
        curr_pos = result[curr_pos]-1
        num_clusters-=1
    out[0:curr_pos+1] = num_clusters
    return out




@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:]), ('k', int64)])
class RestrictedCalculator:
    def __init__(self, v, k):
        self.cumsum = calc_cumsum(v)
        self.cumsum2 = calc_cumsum(np.square(v))
        self.k = k

    def calc(self, i, j):
        #print(i, j)
        if not (j - i >= self.k):
            #print("A", i, j, self.k)
            return np.inf
        if not (j - i <= 2*self.k -1):
            #print("B", i, j, self.k)
            return np.inf
        #print("C", i, j)
        return calc_objective_upper_exclusive(self.cumsum, self.cumsum2, i, j)


@njit([(int64, float64[:], int64, bool_)], cache=True)
def _conventional_algorithm(n, vals, k, full):
    """Solves the univariate microaggregation problem in O(n^2)
    this is an implementation of the conventional algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    if n > 1000:
        raise ValueError("Probably not intended to allocate such a large array, use other algorithm")
    calculator = RestrictedCalculator(vals, k)
    g = np.zeros((n,n+1))
    g[0,0]=0
    min_cost = np.empty(n+1)
    min_cost[0]=0
    best_pred = np.zeros(n, dtype=np.int32)
    for col in range(1,n+1):
        lb = 0
        ub = col
        if not full:
            lb = max(col-2*k+1, 0)
            ub = max(col-k+1, 0)

        for row in range(lb, ub):
            #print(i, j, calculator.calc(i, j))
            g[row, col] = min_cost[row] + calculator.calc(row, col)
        if lb == ub:
            best_pred[col-1]=0
            min_cost[col]=np.inf
        else:
            #print(g)
            best_pred[col-1] = np.argmin(g[lb:ub, col])+lb
            #print(j,  F[j-1])
            #print()
            min_cost[col] = g[best_pred[col-1],col]

    return best_pred, g


def conventional_algorithm(vals, k : int, full : bool=False, should_print : bool=True):
    """Solves the univariate microaggregation problem in O(n^2)
    this is an implementation of the conventional algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    n = len(vals)
    F, g = _conventional_algorithm(n, vals, k, full) #pylint: disable=unused-variable
    if should_print:
        with np.printoptions(linewidth=200, precision=3, suppress=True):
            print(g[:, 1:].T)
    return relabel_clusters_plus_one(F)







@njit()
def __wilber(n, wil_calculator):
    """Solves Univariate Microaggregation problem in O(n)
    this is an implementation of the proposed algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    F = np.empty(n, dtype=np.int32)
    F_vals = wil_calculator.F_vals
    H = np.empty(n, dtype=np.int32)
    H_vals = np.empty(n+1, dtype=np.float64)
    F_vals[0]=0
    c = 0 # columns [0,c] have correct F_vals
    r = 0 # rows [r,c] may contain column minima


    while c < n:
        p = min(2*c-r+1, n)
        #print("F_input", r, c+1, c, p)
        _smawk_iter(np.arange(c, p), np.arange(r, c+1), wil_calculator, F)
        #print("F", F)
        for j in range(c, p):
            F_vals[j+1] = wil_calculator.calc(j, F[j])

        #print("H", c+1, p, c+1,p)
        _smawk_iter(np.arange(c+1, p), np.arange(c+1, p), wil_calculator, H)
        for j in range(c+1, p):
            H_vals[j+1] = wil_calculator.calc(j, H[j])

        j0=p+1
        for j in range(c+2, p+1):
            if H_vals[j] < F_vals[j]:
                F[j-1] = H[j-1]
                j0 = j
                break
        if j0==p+1: # we were right all along
            # F_vals up to p (inclusive) are correct
            r = F[p-1]
            c = p
        else: # our guessing strategy failed
            F_vals[j0] = H_vals[j0]
            r = c+1
            c = j0

    return F

def trivial_cases(n, k, dtype=np.int64):
    if k == 0:
        return np.arange(n, dtype=dtype)
    if 2*k > n:
        return np.zeros(n, dtype=dtype)
    if 2*k == n:
        out = np.empty(n, dtype=dtype)
        out[:k]=0
        out[k:]=1
        return out
    return None






@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:]), ('k', int64), ("F_vals", float64[:]), ("G", float64[:,:]),("SMALL_VAL", float64), ("LARGE_VAL", float64)])
class MicroaggWilberCalculator_edu:
    """An educational variant of the microagg calculator which keeps track of all the states visited in matrix G"""
    def __init__(self, cumsum, cumsum2, k, F_vals):
        self.cumsum = cumsum
        self.cumsum2 = cumsum2
        self.k = k
        self.F_vals = F_vals
        n = len(cumsum) - 1
        self.G = -np.ones((n, n))
        self.SMALL_VAL = calc_objective_upper_inclusive(cumsum, cumsum2, 0, n-1) + 1
        self.LARGE_VAL = self.SMALL_VAL * (1 + n)

    def calc(self, j, i): # i <-> j interchanged is not a bug!
        #print(j, i)
        if j < i:
            self.G[i,j]=np.inf
            #print(i, j, np.inf)
            return np.inf

        if not (j+1 - i >= self.k):
            #print("A", i, j, self.LARGE_VAL + self.SMALL_VAL*i)
            self.G[i,j] = self.LARGE_VAL +  self.SMALL_VAL*i
            return self.LARGE_VAL + self.SMALL_VAL*i
        if not (j+1 - i <= 2 * self.k - 1):
            #print("B", i, j)
            self.G[i,j] = self.LARGE_VAL - self.SMALL_VAL*i
            return self.LARGE_VAL - self.SMALL_VAL*i
        #if self.F_vals[i] >= self.SMALL_VAL: # bogus value
        #    #print("C", i, j, self.LARGE_VAL + self.SMALL_VAL*i)
        #    if j > i:
        #        self.G[i,j] = self.LARGE_VAL +  self.SMALL_VAL*i
        #    return self.LARGE_VAL + self.SMALL_VAL*i
        #print(i, j, self.calculator.calc(i, j) + self.F_vals[i])
        self.G[i,j] = calc_objective_upper_inclusive(self.cumsum, self.cumsum2, i, j) + self.F_vals[i]
        #print(" ", i, j, calc_objective_1(self.cumsum, self.cumsum2, i, j) + self.F_vals[i])
        return calc_objective_upper_inclusive(self.cumsum, self.cumsum2, i, j) + self.F_vals[i]


@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:]), ('k', int64), ("F_vals", float64[:]), ("SMALL_VAL", float64), ("LARGE_VAL", float64)])
class MicroaggWilberCalculator:
    """The standard microagg calculator for wilbers method"""
    def __init__(self, cumsum, cumsum2, k, F_vals):
        self.cumsum = cumsum
        self.cumsum2 = cumsum2
        self.k = k
        self.F_vals = F_vals
        n = len(cumsum) - 1
        self.SMALL_VAL = calc_objective_upper_inclusive(cumsum, cumsum2, 0, n-1)
        self.LARGE_VAL = self.SMALL_VAL * (1 + n)

    def calc(self, j, i): # i <-> j interchanged is not a bug!
        if j < i:
            return np.inf

        if not (j+1 - i >= self.k):
            return self.LARGE_VAL + self.SMALL_VAL*i
        if not (j+1 - i <= 2 * self.k - 1):
            return self.LARGE_VAL - self.SMALL_VAL*i
        return calc_objective_upper_inclusive(self.cumsum, self.cumsum2, i, j) + self.F_vals[i]



@jitclass([('cumsum', float64[:,:]), ('cumsum2', float64[:,:]), ('k', int64), ("F_vals", float64[:]), ("SMALL_VAL", float64), ("LARGE_VAL", float64), ("cell_size", int64)])
class StableMicroaggWilberCalculator:
    """A stable variant of the microagg calculator for wilbers method"""
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

    def calc(self, j, i): # i <-> j interchanged is not a bug!
        if j < i:
            return np.inf

        if not (j+1 - i >= self.k):
            return self.LARGE_VAL + self.SMALL_VAL*i
        if not (j+1 - i <= 2 * self.k - 1):
            return self.LARGE_VAL - self.SMALL_VAL*i

        return calc_objective_cell(self.cumsum, self.cumsum2, self.cell_size, i, j) + self.F_vals[i]


def wilber_edu(v, k, should_print=True):
    result, G  = _wilber_edu(v, k)
    if should_print:
        with np.printoptions(linewidth=300, precision=3, suppress=True):
            print(G.T)
    return relabel_clusters_plus_one(result)

@njit([(float64[:], int64)], cache=USE_CACHE)
def _wilber_edu(v, k):
    n = len(v)
    cumsum = calc_cumsum(v)
    cumsum2 = calc_cumsum(np.square(v))
    wil_calculator = MicroaggWilberCalculator_edu(cumsum, cumsum2, k, np.empty(n+1, dtype=np.float64))

    result = __wilber(n, wil_calculator)
    return result, wil_calculator.G

@njit([(float64[:], int64, int64)], cache=USE_CACHE)
def _wilber(v, k, stable=1):
    n = len(v)
    if stable==1:
        wil_calculator = StableMicroaggWilberCalculator(v, k, np.empty(n+1, dtype=np.float64), k)
        return relabel_clusters_plus_one(__wilber(n, wil_calculator))
    elif stable==0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        wil_calculator = MicroaggWilberCalculator(cumsum, cumsum2, k, np.empty(n+1, dtype=np.float64))
        return relabel_clusters_plus_one(__wilber(n, wil_calculator))
    else:
        raise NotImplementedError("Only stable in (0,1) supported")






def wilber(arr, k : int, stable=1):
    """Solves the REGULARIZED 1d kmeans problem in O(n)
    this is an implementation of the proposed algorithm
    from "The concave least weight subsequence problem revisited" by Robert wilber 1987
    """

    assert k > 0
    assert k <= len(arr)
    res = trivial_cases(len(arr), k)
    if not res is None:
        return res
    return _wilber(arr, k, stable=stable)