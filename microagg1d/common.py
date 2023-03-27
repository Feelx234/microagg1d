import numpy as np
from numba import njit, float64, int64


USE_CACHE=True

@njit([(float64[:],)], cache=USE_CACHE)
def calc_cumsum(v):
    cumsum = np.empty(len(v)+1, dtype=np.float64)
    cumsum[0]=0
    cumsum[1:] = np.cumsum(v)
    return cumsum


@njit([(float64, float64, int64)], cache=USE_CACHE)
def _calc_objective(val1, val2, n):
    """Compute the cluster cost of clustering points including i excluding j"""
    mu = val1/n
    result = val2
    result += n * (mu * mu)
    result -= (2 * mu) * (val1)
    return max(result, 0)


@njit([(float64[:], float64[:], int64, int64)], cache=USE_CACHE)
def calc_objective_upper_exclusive(cumsum, cumsum2, i, j):
    """Compute the cluster cost of clustering points including i excluding j"""
    if j <= i:
        return 0.0
    return _calc_objective(cumsum[j]-cumsum[i], cumsum2[j] - cumsum2[i], j-i)
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
    return _calc_objective(cumsum[j+1]-cumsum[i], cumsum2[j+1] - cumsum2[i], j+1-i)
    #mu = (cumsum[j + 1]-cumsum[i])/(j + 1-i)
    #result = cumsum2[j + 1] - cumsum2[i]
    #result += (j - i + 1) * (mu * mu)
    #result -= (2 * mu) * (cumsum[j + 1] - cumsum[i])
    #return max(result, 0)


@njit([(float64[:],int64)], cache=True)
def calc_cumsum_cell(v, cell_size):
    """Computes cumsums in cells of size cell_size
    instead of cumsum([1,2,3,4]) = [0,1,3,6,10]
    cumsum_cell([1,2,3,4], 2) = [[0,1,3], [0, 3, 7]]
    This has a numeric advantage as the numbers stored grow slower
    """
    quotient, remainder = divmod(len(v), cell_size)
    num_cells = quotient
    if remainder!=0:
        num_cells+=1
    out = np.zeros((num_cells, cell_size+1), dtype=np.float64)

    for i in range(quotient):
        curr_out = out[i,:]
        curr_out[0] = 0.0
        offset = i*cell_size
        for j in range(cell_size):
            x = v[offset+j]
            curr_out[j+1] = curr_out[j] + x
    if remainder != 0:
        i=quotient
        curr_out = out[i,:]
        curr_out[0] = 0.0
        offset = i*cell_size
        for j in range(remainder):
            x = v[offset+j]
            curr_out[j+1] = curr_out[j] + x
    return out


@njit([(float64[:], float64[:], float64[:], float64[:], int64, int64)], cache=USE_CACHE)
def calc_objective_upper_inclusive_2(i_cumsum, i_cumsum2, j_cumsum, j_cumsum2,  i, j):
    """Compute the cluster cost of clustering points including both i and j across two cells"""
    cell_size = len(i_cumsum)-1
    val1 = j_cumsum[j + 1] + i_cumsum[cell_size] - i_cumsum[i]
    #print("\t", val1)
    mu = (val1)/(j + 1 + cell_size - i)
    result = j_cumsum2[j + 1] + i_cumsum2[cell_size] - i_cumsum2[i]
    #print("\t", result)
    result -= (2 * mu) * val1
    result += (j - i + 1+ cell_size) * (mu * mu)
    #print("\t", result)
    return max(result, 0.0)

@njit([(float64[:,:], float64[:,:], int64, int64, int64)], cache=USE_CACHE)
def calc_objective_cell(cumsum, cumsum2, cell_size, i, j):
    assert j>=i
    assert j - i < 2 * cell_size

    cell_i, remainder_i = divmod(i, cell_size)
    cell_j, remainder_j = divmod(j, cell_size)
    if cell_i  == cell_j: # both are in one cell
        return calc_objective_upper_inclusive(cumsum[cell_i,:], cumsum2[cell_i,:], remainder_i, remainder_j)
    else:
        return calc_objective_upper_inclusive_2(cumsum[cell_i,:], cumsum2[cell_i,:], cumsum[cell_j,:], cumsum2[cell_j,:], remainder_i, remainder_j)