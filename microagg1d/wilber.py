import numpy as np
from numba import njit, float64, int64

from microagg1d.main import calc_cumsum, calc_cumsum2


@njit([(float64[:], float64[:], int64, int64)], cache=False)
def calc_objective(cumsum, cumsum2, i, j):
    if j <= i:
        return 0.0
#            raise ValueError("j should never be larger than i")
    mu = (cumsum[j]-cumsum[i])/(j-i)
    result = cumsum2[j] - cumsum2[i]
    result += (j - i) * (mu * mu)
    result -= (2 * mu) * (cumsum[j] - cumsum[i])
    return max(result, 0)

@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:]), ('k', int64)])
class RestrictedCalculator:
    def __init__(self, v, k):
        self.cumsum = calc_cumsum(v)
        self.cumsum2 = calc_cumsum2(v)
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
        return calc_objective(self.cumsum, self.cumsum2, i, j)

#@njit([(int64, float64[:], int64)], cache=USE_CACHE)
def _conventional_algorithm(n, vals, k):
    """Solves the univariate microaggregation problem in O(n^2)
    this is an implementation of the conventional algorithm
    from "The concave least weight subsequence problem revisited" by Robert Wilber 1987
    """
    calculator = RestrictedCalculator(vals, k)
    g = np.zeros((n,n+1))
    g[0,0]=0
    min_cost = np.empty(n+1)
    min_cost[0]=0
    best_pred = np.zeros(n, dtype=np.int32)
    for col in range(1,n+1):

        lb = max(col-2*k+1, 0)
        ub = max(col-k+1, 0)
        #for i in range(0, lb):
        #    g[i,j] = 999
        for row in range(lb,ub):
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