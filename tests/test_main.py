# pylint: disable=missing-function-docstring
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from microagg1d.generation import create_pair_arange
from microagg1d.main import undo_argsort, univariate_microaggregation
from microagg1d.utils_for_test import remove_numba_from_class, restore_to_class

# use with k=5
interesting_arr = np.array(
    [
        1.14374817e-04,
        2.73875932e-02,
        9.23385948e-02,
        1.46755891e-01,
        1.86260211e-01,
        2.04452250e-01,
        3.02332573e-01,
        3.45560727e-01,
        3.96767474e-01,
        4.17022005e-01,
        4.19194514e-01,
        5.38816734e-01,
        6.85219500e-01,
        7.20324493e-01,
        8.78117436e-01,
    ]
)


def get_random_arr(seed, n):
    np.random.seed(seed)
    x = np.random.rand(n)
    order = np.argsort(x)
    x_sorted = x[order]
    return x, x_sorted, order


class TestMainLarger(unittest.TestCase):
    def test_microagg_larger_input(self):
        arr, solution = create_pair_arange(150, 22)
        result = univariate_microaggregation(arr, 22)
        np.testing.assert_array_equal(solution, result)


class TestMain(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.array([1, 1, 1, 1.1, 5, 5, 5])
        self.solutions = [
            [0, 1, 2, 3, 4, 5, 6],
            [0, 0, 1, 1, 2, 2, 2],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]

    def test_undo_argsort_random(self):
        for seed in range(20):
            x, x_sorted, order = get_random_arr(seed, n=100)
            x_undone = undo_argsort(x_sorted, order)
            assert_array_equal(x, x_undone)

    def test_microagg_raises(self):
        with self.assertRaises(AssertionError):
            univariate_microaggregation(np.random.rand(4, 4), 1)

        with self.assertRaises(AssertionError):
            univariate_microaggregation(self.arr, 0)

    def test_microagg_combinations(self):
        for method in ("auto", "wilber", "galil_park", "staggered"):
            for cost in ("sse", "sae", "roundup", "rounddown", "maxdist"):
                with self.subTest(msg=f"{method}_{cost}"):
                    for k, solution in zip(range(1, len(self.arr) + 1), self.solutions):
                        result = univariate_microaggregation(
                            self.arr.copy(), k, method=method, cost=cost
                        )
                        np.testing.assert_array_equal(solution, result, f"k={k}")

    def test_microagg_no_arguments(self):
        for k, solution in zip(range(1, len(self.arr) + 1), self.solutions):
            result = univariate_microaggregation(self.arr.copy(), k)
            np.testing.assert_array_equal(solution, result, f"k={k}")

    def test_example_usage(self):
        # pylint: disable=redefined-outer-name,reimported,import-outside-toplevel
        from microagg1d import univariate_microaggregation

        x = [6, 2, 0.7, 1.1, 5, 1, 5.1]

        clusters = univariate_microaggregation(x, k=3)

        print(clusters)  # [1 0 0 0 1 0 1]

        # explicitly choose method / algorithm
        clusters2 = univariate_microaggregation(x, k=3, method="wilber")

        print(clusters2)  # [1 0 0 0 1 0 1]

        # choose a different cost (sae / sse / roundup / rounddown / maxdist)
        clusters3 = univariate_microaggregation(x, k=3, cost="sae")

        print(clusters3)  # [1 0 0 0 1 0 1]

        np.testing.assert_array_equal(clusters, [1, 0, 0, 0, 1, 0, 1], f"k={3}")

        np.testing.assert_array_equal(clusters2, [1, 0, 0, 0, 1, 0, 1], f"k={3}")

        np.testing.assert_array_equal(clusters3, [1, 0, 0, 0, 1, 0, 1], f"k={3}")


class TestMainNonCompiled(TestMain):
    def setUp(self):
        self.cleanup = remove_numba_from_class(
            self.__class__.__bases__[0], allowed_packages=["microagg1d"]
        )

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)


if __name__ == "__main__":
    unittest.main()
