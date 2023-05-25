import numpy as np
from numba import njit, int64, float64

from microagg1d.smawk_iter2 import __smawk_iter
from microagg1d.wilber import SimpleWilberCalculator

USE_CACHE=True

@njit()
def __wilber2(n, wil_calculator, k):
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

    col_buffer = np.empty(2*n+2, dtype=np.int64)
    col_starts = np.empty(2*n+2, dtype=np.int64)


    while c < n:
        p = min(2*c-r+1, n)
        #print("F_input", r, c+1, c, p)
        __smawk_iter(c, p, r, c+1, wil_calculator, F, col_starts, col_buffer)
        #print("F", F)
        for j in range(c, p):
            F_vals[j+1] = wil_calculator.calc(j, F[j])

        #print("H", c+1, p, c+1,p)
        __smawk_iter(c+1, p, c+1, p, wil_calculator, H, col_starts, col_buffer)
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
            r = F[j0-1]
            c = j0

    return F

from microagg1d.wilber import StableMicroaggWilberCalculator, relabel_clusters_plus_one, calc_cumsum, MicroaggWilberCalculator



@njit()
def __galil_park2(n, wil_calculator, k):
    """ Solves the dynamic problem in O(n)
    This is an implementation of the proposed algorithm
    from "A Linear-Time Algorithm for Concave One-Dimensional Dynamic Programming" by Zvi Galil and Kunsoo Park 1989
    """
    F = np.empty(n, dtype=np.int32)
    F_vals = wil_calculator.F_vals
    N = np.empty(n, dtype=np.int32)
    N_vals =  np.inf * np.ones(n+1, dtype=np.float64)
    F_vals[0]=0
    c = 0 # columns [0,c] have correct F_vals
    r = 0 # rows [r,c] may contain column minima

    col_buffer = np.empty(2*n+2, dtype=np.int64)
    col_starts = np.empty(2*n+2, dtype=np.int64)

    while c < n:
        p = min(2*c-r+1, n)
        __smawk_iter(c, p, r, c+1, wil_calculator, F, col_starts, col_buffer)
        for j in range(c, p):
            val = wil_calculator.calc(j, F[j])
            if val < N_vals[j+1]:
                F_vals[j+1] = val
            else:
                F_vals[j+1] = N_vals[j+1]
                F[j] = N[j]
        j0=p+2
        for j in range(c+2, p+1):
            if wil_calculator.calc(j-1, j-1) < F_vals[j]:
                # the H value considered was smaller, may not continue
                F[j-1] = j-1
                j0 = j
                F_vals[j0] = wil_calculator.calc(j-1, j-1)
                r = c
                c = j0
                break

            if F_vals[p] <= wil_calculator.calc(p-1, j-1):
                # we did just eliminate row j entries c:p
                # => may continue as usual
                pass
            else:
                # print("B")
                # need to break because it is not guaranteed that the following
                # F values, (F[j+1:]) are correct as well, they might lie in row j
                j0=j
                #F[j-1]=j
                N[j-1:p+1] = F[j-1:p+1]
                N_vals[j-1:p] = F_vals[j-1:p]
                r = c+1
                c = j0
                break
        if j0==p+2:
            # F_vals up to p (inclusive) are correct
            r = max(r, F[p-1])
            c = p
        else: # our guessing strategy failed
            pass
            #r = c
            #c = j0

    return F





@njit()
def __staggered2(n, wil_calculator, k):
    """ Solves the dynamic problem in O(n)
    """
    F = np.empty(n, dtype=np.int32)
    F_vals = wil_calculator.F_vals
    F_vals[0]=0
    col_buffer = np.empty(2*n+2, dtype=np.int64)
    col_starts = np.empty(2*n+2, dtype=np.int64)

    # initial values
    for i in range(k-1, min(2*k-1,n)):
        F[i] = 0
        F_vals[i+1] = wil_calculator.calc(i, 0)
    if n<=2*k-1:
        return F

    def one_round(l, r, u, b):
        f_min = np.min(F_vals[u:b])
        for j in range(u, b):
            F_vals[j]-=f_min

        __smawk_iter(l, r, u, b, wil_calculator, F, col_starts, col_buffer)
        for j in range(l, r):
            F_vals[j+1] = wil_calculator.calc(j, F[j])

    # first block
    max_col = min(3*k-1,n)
    one_round(2*k-1, max_col, k, 2*k)
    if n<=max_col:
        return F

    # remaining blocks
    f, R = divmod(n-3*k+1,k)
    j=3
    for i in range(3,3+f): # do the main blocks
        one_round(i*k-1, (i+1)*k-1, max(F[i*k-2], (i-2)*k), i*k)
        j=i+1

    if R > 0: # deal with the remainder
        one_round(j*k-1, (j)*k-1+R, max(F[j*k-2], (j-2)*k), (j-1)*k+R)

    return F









# def wilber2(arr, k : int, stable=1):
#     """Solves the REGULARIZED 1d kmeans problem in O(n)
#     this is an implementation of the proposed algorithm
#     from "The concave least weight subsequence problem revisited" by Robert wilber 1987
#     """
#     return execute_linear2(__wilber2, arr, k, stable)



@njit([(float64[:], int64, int64)], cache=USE_CACHE)
def _staggered2(v, k, stable=1):
    method=__staggered2
    n = len(v)
    if stable==2:
        wil_calculator = SimpleWilberCalculator(calc_cumsum(v), k, -np.ones(n+1, dtype=np.float64))
        return relabel_clusters_plus_one(method(n, wil_calculator, k))
    if stable==1:
        wil_calculator = StableMicroaggWilberCalculator(v, k, -np.ones(n+1, dtype=np.float64), 3*k)
        return relabel_clusters_plus_one(method(n, wil_calculator, k))
    elif stable==0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        wil_calculator = MicroaggWilberCalculator(cumsum, cumsum2, k, -np.ones(n+1, dtype=np.float64))
        out = method(n, wil_calculator, k)
        return relabel_clusters_plus_one(out)
    else:
        raise NotImplementedError("Only stable in (0,1) supported")


@njit([(float64[:], int64, int64)], cache=USE_CACHE)
def _galil_park2(v, k, stable=1):
    method=__galil_park2
    n = len(v)
    if stable==2:
        wil_calculator = SimpleWilberCalculator(calc_cumsum(v), k, -np.ones(n+1, dtype=np.float64))
        return relabel_clusters_plus_one(method(n, wil_calculator, k))
    if stable==1:
        wil_calculator = StableMicroaggWilberCalculator(v, k, -np.ones(n+1, dtype=np.float64), 3*k)
        return relabel_clusters_plus_one(method(n, wil_calculator, k))
    elif stable==0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        wil_calculator = MicroaggWilberCalculator(cumsum, cumsum2, k, -np.ones(n+1, dtype=np.float64))
        out = method(n, wil_calculator, k)
        return relabel_clusters_plus_one(out)
    else:
        raise NotImplementedError("Only stable in (0,1) supported")


@njit([(float64[:], int64, int64)], cache=USE_CACHE)
def _wilber2(v, k, stable=1):
    method=__wilber2
    n = len(v)
    if stable==2:
        wil_calculator = SimpleWilberCalculator(calc_cumsum(v), k, -np.ones(n+1, dtype=np.float64))
        return relabel_clusters_plus_one(method(n, wil_calculator, k))
    if stable==1:
        wil_calculator = StableMicroaggWilberCalculator(v, k, -np.ones(n+1, dtype=np.float64), 3*k)
        return relabel_clusters_plus_one(method(n, wil_calculator, k))
    elif stable==0:
        cumsum = calc_cumsum(v)
        cumsum2 = calc_cumsum(np.square(v))
        wil_calculator = MicroaggWilberCalculator(cumsum, cumsum2, k, -np.ones(n+1, dtype=np.float64))
        out = method(n, wil_calculator, k)
        return relabel_clusters_plus_one(out)
    else:
        raise NotImplementedError("Only stable in (0,1) supported")