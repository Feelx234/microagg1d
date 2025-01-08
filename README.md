[![build](https://github.com/Feelx234/microagg1d/actions/workflows/ci.yaml/badge.svg)](https://github.com/Feelx234/microagg1d/actions)
[![Coverage Status](https://coveralls.io/repos/github/Feelx234/microagg1d/badge.svg)](https://coveralls.io/github/Feelx234/microagg1d)
[![arXiv:2401.02381](https://img.shields.io/badge/arXiv-2401.02381-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2401.02381)

microagg1d
========

A Python library which implements different techniques for optimal univariate microaggregation. The two main parameters that determine the runtime are the length n of the input array and minimal class size k. This package offers both O(n) (fast for large k) and O(kn) (fast for small k) algorithms.

The code is written in Python and relies on the [numba](https://numba.pydata.org/) compiler for speed.

Requirements
------------

*microagg1d* relies on `numpy` and `numba` which currently support python 3.8-3.11.

Installation
------------

microagg1d is available on [PyPI](https://pypi.python.org/pypi/microagg1d), the Python Package Index.

```sh
$ pip3 install microagg1d
```

Example Usage
-------------

```python
from microagg1d import univariate_microaggregation

x = [5, 1, 1, 1.1, 5, 1, 5.1]

clusters = univariate_microaggregation(x, k=3)

print(clusters)   # [1 0 0 0 1 0 1]

# explicitly choose method / algorithm
clusters2 = univariate_microaggregation(x, k=3, method="wilber")

print(clusters2)   # [1 0 0 0 1 0 1]

# choose a different cost (sae / sse / roundup / rounddown / maxdist)
clusters3 = univariate_microaggregation(x, k=3, cost="sae")

print(clusters3)   # [1 0 0 0 1 0 1]
```

**Important notice**: On first import the the code is compiled once which may take about 30s. On subsequent imports this is no longer necessary and imports are almost instant.

Tests
-----

Tests are in [tests/](https://github.com/Feelx234/microagg1d/tree/main/tests).

```sh
# Run tests
$ python3 -m pytest .
```

Method Details
--------------

Currently the package implements the following methods:
- `"simple"` [O(nk), faster for small k]
- `"wilber"` [O(n), faster for larger k]
- `"galil_park"` [O(n), fewer calls to SMAWK]
- `"staggered"` [fastest O(n)]

By default, the package switches between the simple and wilber method depending on the size of k.

Both methods rely on a prefix sum approach to compute the cluster cost. As the prefix sums tend to become very large quite quickly, a slightly slower but numerically more robust method is chosen by default. If your data is small, or you don't need the numeric stability then you may choose to also opt out of stable.



License
-------

The code in this repository has an BSD 2-Clause "Simplified" License.

See [LICENSE](https://github.com/Feelx234/microagg1d/blob/master/LICENSE).



Citation
-----------

This code was created as part of the research for the following publication. If you use this package please cite:

```
@article{stamm2024faster,
	title = {Faster optimal univariate microaggregation},
	url = {https://openreview.net/forum?id=s5lEUtyVly},
	journal = {Transactions on Machine Learning Research},
	author = {Stamm, Felix I. and Schaub, Michael T},
	year = {2024},
}
```


References
----------

- Hansen, S.L. and Mukherjee, S., 2003. A polynomial algorithm for optimal univariate microaggregation. IEEE Transactions on Knowledge and Data Engineering
- Wilber, R., 1988. The concave least-weight subsequence problem revisited. Journal of Algorithms, 9(3), pp.418-425.
- Galil, Z. and Park, K., 1989. A linear-time algorithm for concave one-dimensional dynamic programming.
