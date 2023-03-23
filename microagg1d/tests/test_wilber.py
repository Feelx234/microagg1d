
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from microagg1d.wilber import conventional_algorithm, Wilber, _Wilber, Wilber_edu
from microagg1d.main import optimal_univariate_microaggregation_1d, _simple_dynamic_program, compute_cluster_cost_sorted
from functools import partial

def my_test_algorithm(self, algorithm):
    for k, solution in self.solutions.items():
        result = algorithm(self.arr, k)
        np.testing.assert_array_equal(solution, result, f"k={k}")

class Test8Elements(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.array([1.1, 1.2, 1.3, 1.4, 5, 5, 5, 5])
        self.solutions = {
            1 : [0, 1, 2, 3, 4, 5, 6, 7],
            2 : [0, 0, 1, 1, 2, 2, 3, 3],
            3 : [0, 0, 0, 0, 1, 1, 1, 1],
            4 : [0, 0, 0, 0, 1, 1, 1, 1],
            5 : [0, 0, 0, 0, 0, 0, 0, 0],
        }

    def test_conventional_algorithm(self):
        my_test_algorithm(self, partial(conventional_algorithm, should_print=False))

    def test_conventional_algorithm_full(self):
        my_test_algorithm(self, partial(conventional_algorithm, full=True, should_print=False))

    def test_Wilber(self):
        my_test_algorithm(self, Wilber)

    def test__Wilber(self):
        my_test_algorithm(self, _Wilber)

    def test_Wilber_edu(self):
        my_test_algorithm(self, partial(Wilber_edu, should_print=False))

    def test_optimal_univariate_microaggregation_simple(self):
        my_test_algorithm(self, partial(optimal_univariate_microaggregation_1d, method="simple"))

    def test_optimal_univariate_microaggregation_wilber(self):
        my_test_algorithm(self, partial(optimal_univariate_microaggregation_1d, method="wilber"))



class Test7Elements(Test8Elements):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.array([1.1, 1.2, 1.3, 1.4, 5, 5, 5])
        self.solutions = {
            1 : [0, 1, 2, 3, 4, 5, 6],
            2 : [0, 0, 1, 1, 2, 2, 2],
            3 : [0, 0, 0, 0, 1, 1, 1],
            4 : [0, 0, 0, 0, 0, 0, 0],
            5 : [0, 0, 0, 0, 0, 0, 0],
        }




class TestArray(Test8Elements):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.array([1.14374817e-04, 2.73875932e-02, 9.23385948e-02, 1.46755891e-01,
       1.86260211e-01, 2.04452250e-01, 3.02332573e-01, 3.45560727e-01,
       3.96767474e-01, 4.17022005e-01, 4.19194514e-01, 5.38816734e-01,
       6.85219500e-01, 7.20324493e-01, 8.78117436e-01])
        self.solutions = {
            1 : np.arange(15),
            2 : np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6]),
            3 : np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
            4 : np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]),
            5 : np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            6 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
            7 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
            8 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        }


class TestAgreement(Test8Elements):
    def test_1(self):
        np.random.seed(0)
        arr = np.random.rand(1_000_000)
        arr.sort()
        result1 = Wilber(arr, 2)
        result2 = _simple_dynamic_program(arr, 2)

        cost1 = compute_cluster_cost_sorted(arr, result1)
        cost2 = compute_cluster_cost_sorted(arr, result2)
        self.assertEqual(cost1, cost2)
        #assert_array_equal(result1, result2)




if __name__ == '__main__':
    unittest.main()