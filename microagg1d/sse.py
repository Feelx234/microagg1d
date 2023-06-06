import numpy as np
from numba import njit

@njit
def sse_stable(v):
    mean = 0
    for x in v:
        mean+=x
    mean/=len(v)
    s = 0
    for x in v:
        s+=(x-mean)**2
    return s
@njit
def sse_from_mean_stable(v, mean):
    s = 0
    for x in v:
        s+=(x-mean)**2
    return s

@njit(cache=True)
def compute_sse_sorted_stable(v, clusters_sorted):
    s = 0.0
    l = 0
    r = 0
    while r < len(v):
        r=l
        while clusters_sorted[l]==clusters_sorted[r] and  r < len(v):
            r+=1
        #r-=1
        mean = np.mean(v[l:r])
        sse = sse_from_mean_stable(v[l:r], mean)
        s+=sse
        l=r
    return s