[![build](https://github.com/Feelx234/microagg1d/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/Feelx234/microagg1d/actions)

microagg1d
========

A Python library which implements different techniques for optimal univariate microaggregation. The two main parameters that determine the runtime are the length n of the input array and minimal class size k.

Currently the package implements the following methods:
- `"simple"` [O(nk), faster for small k]
- `"wilber"` [O(n), faster for larger k]
By default, the package switches between the two methods depending on the size of k.


Both methods rely on a prefix sum approach to compute the cluster cost. As the prefix sums tend to become very large quite quickly, a slightly slower but numerically more robust method is chosen by default. If your data is small, or you don't need the numeric stability then you may choose to also opt out of stable.

The code is written in Python and relies on the [numba](https://numba.pydata.org/) compiler for speed.

Requirements
------------

*microagg1d* relies on `numpy` and `numba` which currently support python 3.8-3.10.

Installation
------------

[microagg1d](https://pypi.python.org/pypi/microagg1d) is available on PyPI, the Python Package Index.

```sh
$ pip3 install microagg1d
```

Example Usage
-------------

```python
import microagg1d

x = [5, 1, 1, 1.1, 5, 1, 5.1]
k = 3

clusters = microagg1d.optimal_univariate_microaggregation_1d(x, k) # automatically choose method

print(clusters)   # [1 0 0 0 1 0 1]

clusters2 = microagg1d.optimal_univariate_microaggregation_1d(x, k=2, method="wilber") # explicitly choose method

print(clusters2)   # [1 0 0 0 1 0 1]

# may opt to get increased speed at cost of stability, this is usually not a problem on small datasets like the one used here
# stable works with both wilber and the simple method
clusters3 = microagg1d.optimal_univariate_microaggregation_1d(x, k=2, stable=False)

print(clusters3)   # [1 0 0 0 1 0 1]
```

Important notice: On first usage the the code is compiled once which may take about 30s. On subsequent usages this is no longer necessary and execution is much faster.

Tests
-----

Tests are in [tests/](https://github.com/Feelx234/microagg1d/blob/master/tests).

```sh
# Run tests
$ python3 -m pytest .
```

License
-------

The code in this repository has an BSD 2-Clause "Simplified" License.

See [LICENSE](https://github.com/Feelx234/microagg1d/blob/master/LICENSE).

