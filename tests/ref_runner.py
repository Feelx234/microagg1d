"""Runs the ORIGINAL numba microagg1d on pickled cases. Executed with REFERENCE_PYTHON."""
import pickle
import sys

import numpy as np
from microagg1d import univariate_microaggregation

cases = pickle.load(open(sys.argv[1], "rb"))
out = []
for x, k, method, cost, stable in cases:
    try:
        out.append(np.asarray(univariate_microaggregation(x, k, method=method, cost=cost, stable=stable)))
    except Exception as e:  # noqa
        out.append(f"EXC {type(e).__name__}: {e}")
pickle.dump(out, open(sys.argv[2], "wb"))
