import numpy as np
from numba import njit, float64, int64
from numba.experimental import jitclass
from microagg1d.wilber import wilber
from microagg1d.common import calc_cumsum, calc_objective_upper_inclusive, calc_objective_cell, calc_cumsum_cell


USE_CACHE=True







@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:])])
class CumsumCalculator:
    def __init__(self, v):
        self.cumsum = calc_cumsum(v)
        self.cumsum2 = calc_cumsum(np.square(v))

    def calc(self, i, j):
        return calc_objective_upper_inclusive(self.cumsum, self.cumsum2, i, j)

@njit
def compute_cluster_cost_sorted(v, clusters_sorted):
    calculator = CumsumCalculator(v)
    s = 0.0
    i = 0
    j = 1
    while j < len(v):
        while clusters_sorted[j]==clusters_sorted[i] and  j < len(v):
            j+=1
        s+=calculator.calc(i, j-1)
        i=j
        j=i+1
    return s




@njit(cache=USE_CACHE)
def calc_num_clusters(result):
    """Compute the number of clusters encoded in results
    Can be used on e.g. _optimal_univariate_microaggregation
    """
    num_clusters = 0
    curr_pos = len(result)-1
    while result[curr_pos]>=0:
        curr_pos = result[curr_pos]
        num_clusters+=1
    return num_clusters+1


@njit(cache=USE_CACHE)
def relabel_clusters(result):
    num_clusters = calc_num_clusters(result)-1

    out = np.empty_like(result)
    curr_pos = len(result)-1
    while result[curr_pos]>=0:
        out[result[curr_pos]:curr_pos+1] = num_clusters
        curr_pos = result[curr_pos]
        num_clusters-=1
    out[0:curr_pos+1] = num_clusters
    return out



@jitclass([('cumsum', float64[:,:]), ('cumsum2', float64[:,:]), ('cell_size', int64)])
class StableCumsumCalculator:
    def __init__(self, v, cell_size):
        self.cumsum = calc_cumsum_cell(v, cell_size)
        self.cumsum2 = calc_cumsum_cell(np.square(v), cell_size)
        self.cell_size = cell_size

    def calc(self, i, j):
        if j==i:
            return 0
        return calc_objective_cell(self.cumsum, self.cumsum2, self.cell_size, i, j)



@njit(cache=USE_CACHE)
def _simple_dynamic_program(x, k, stable=1):
    n = len(x)
    assert k > 0
    if n//2 < k: # there can only be one cluster
        return np.zeros(n, dtype=np.int64)
    if k==1: # each node has its own cluster
        return np.arange(n)

    if stable==1:
        calculator = StableCumsumCalculator(x, k)
        return __simple_dynamic_program(n, k, calculator)
    elif stable==0:
        calculator = CumsumCalculator(x)
        return __simple_dynamic_program(n, k, calculator)
    else:
        assert False

@njit(cache=False) # no caching as otherwise it would be recompiled often
def __simple_dynamic_program(n, k, calculator):
    back_tracks =  np.zeros(n, dtype=np.int64)
    min_vals = np.zeros(n)
    for i in range(0, k-1):
        min_vals[i] = np.inf
        back_tracks[i]=-1
    for i in range(k-1, 2*k-1):
        min_vals[i] = calculator.calc(0,i)
        back_tracks[i]=-1

    for i in range(2*k-1, n):
        #print("i", i)
        min_index = i-2*k+1
        #print("min", min_index)
        prev_min_val = min_vals[min_index] + calculator.calc(min_index+1, i)
        for j in range(i-2*k + 2, i-k+1):
            #print(j, min_vals[j], prev_min_val)
            new_val = min_vals[j] + calculator.calc(j+1, i)
            if  new_val < prev_min_val:
                min_index = j
                prev_min_val = new_val
        #print("result", min_index, prev_min_val)

        back_tracks[i] = min_index
        min_vals[i] = prev_min_val
        #print(back_tracks)
    return relabel_clusters(back_tracks)



def undo_argsort(sorted_arr, sort_order):
    """Puts the sorted_array which was sorted with sort_order back into the original order"""
    revert = np.empty_like(sort_order)
    revert[sort_order]=np.arange(len(sorted_arr))
    return sorted_arr[revert]



def optimal_univariate_microaggregation_1d(x, k, method="auto", stable=1):
    """Performs optimal 1d univariate microaggregation"""
    x = np.squeeze(np.asarray(x))
    assert len(x.shape)==1, "provided array is not 1d"
    assert k > 0, f"negative or zero values for k({k}) are not supported"
    assert k <= len(x), f"values of k({k}) larger than the length of the provided array ({len(x)}) are not supported"

    assert method in ("auto", "simple", "wilber"), "invalid method supplied"
    if method == "auto":
        if k <= 21: # 21 determined emperically
            method = "simple"
        else:
            method = "wilber"

    order = np.argsort(x)
    x = np.array(x, dtype=np.float64)[order]

    if method=="simple":
        clusters = _simple_dynamic_program(x, k, stable=stable)
    elif method=="wilber":
        clusters = wilber(x, k)
    else:
        raise NotImplementedError("Should not be reachable")
    return undo_argsort(clusters, order)
