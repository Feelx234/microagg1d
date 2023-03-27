[![build](https://github.com/Feelx234/microagg1d/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/Feelx234/microagg1d/actions)

microagg1d
========

A Python library which implements different techniques for optimal univariate microaggregation. The two main parameters that determine the runtime are the length n of the input array and minimal class size k.

Currently the package implements the following methods:
- `"simple"` [O(nk), faster for small k]
- `"wilber"` [O(n), faster for larger k]
By default, the package switches between the two methods depending on the size of k.


Both methods rely on a prefix sum approach to compute the cluster cost. As the prefix sums tend to become very large quite quickly, a slightly slower but numerically more robust method is offered as well.

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

x = [5, 1, 1, 1.1, 5, 1, 5]
k = 3

clusters = microagg1d.optimal_univariate_microaggregation_1d(x, k) # automatically choose method

print(clusters)   # [1 0 0 0 1 0 1]

# for large datasets, should increase numeric stability, but increases runtime
clusters_large = microagg1d.optimal_univariate_microaggregation_1d(np.arange(500_000), k=2, stable=True)

print(clusters_large)   # [     0      0      1 ... 249998 249999 249999]
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

