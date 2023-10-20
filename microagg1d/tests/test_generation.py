# pylint: disable=missing-function-docstring
import unittest
import numpy as np
from numpy.testing import assert_array_equal

from microagg1d.generation import create_pair_arange, create_pair_known_sizes

class TestArangeGeneration(unittest.TestCase):
    def test_arange_generation_simple(self):
        solutions = [
            np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]),
            np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ]

        for k, solution in zip(range(1, 10), solutions):
            arr, res = create_pair_arange(17, k)
            assert_array_equal(arr, np.arange(17))
            assert_array_equal(res, solution, f"k={k}")



    def test_create_pair_known_sizes_epsilon0(self):
        inputs = [
            ([2,2], 2),
            ([2,3,4,5], 2),
            ([5,4,3,2], 2),
            ([5,4,5,4], 4),
        ]
        solutions = [
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]),
        ]
        for alg_input, sol_solution in zip(inputs, solutions):
            arr, solution = create_pair_known_sizes(*alg_input, 0)
            assert_array_equal(arr, np.float64(sol_solution), f"input={alg_input}")
            assert_array_equal(solution, sol_solution, f"input={alg_input}")



    def test_create_pair_known_sizes_epsilon1(self):
        inputs = [
            ([2,2], 2),
            ([2,3,4,5], 2),
            ([5,4,3,2], 2),
            ([5,4,5,4], 4),
        ]
        solutions = [
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]),
        ]
        arrs = [
            np.array([0. , 0.1, 1. , 1.1]),
            np.array([0. , 0.1, 1. , 1.1, 1.2, 2. , 2.1, 2.2, 2.3, 3. , 3.1, 3.2, 3.3,  3.4]),
            np.array([0. , 0.1, 0.2, 0.30000000000000004, 0.4, 1. , 1.1, 1.2, 1.3, 2. , 2.1, 2.2, 3. ,  3.1]),
            np.array([0. , 0.1, 0.2, 0.30000000000000004, 0.4, 1. , 1.1, 1.2, 1.3, 2. , 2.1, 2.2, 2.3, 2.4, 3. , 3.1, 3.2, 3.3]),
        ]
        for alg_input, sol_solution, sol_arr in zip(inputs, solutions, arrs):
            arr, solution = create_pair_known_sizes(*alg_input, 0.1)
            assert_array_equal(arr, sol_arr, f"input={alg_input}")
            assert_array_equal(solution, sol_solution, f"input={alg_input}")

    def test_create_pair_known_sizes_raises_large_k(self):
        # assert raises sizes to small for k
        with self.assertRaises(ValueError):
            create_pair_known_sizes([5,5], k=6, epsilon=0.1)


    def test_create_pair_known_sizes_raises_large_epsilon(self):
        # assert raises for epsilon to large
        with self.assertRaises(ValueError):
            create_pair_known_sizes([5,5], k=4, epsilon=0.2)

        # this also needs to be forbidden as we would have rescaled arange
        with self.assertRaises(ValueError):
            create_pair_known_sizes([4,4], k=4, epsilon=0.2)


if __name__ == '__main__':
    unittest.main()