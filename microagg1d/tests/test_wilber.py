
import unittest
from functools import partial
import numpy as np
from numpy.testing import assert_array_equal
from microagg1d.wilber import conventional_algorithm, wilber, _wilber, wilber_edu
from microagg1d.main import optimal_univariate_microaggregation_1d, _simple_dynamic_program, compute_cluster_cost_sorted


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

    def test_wilber(self):
        my_test_algorithm(self, wilber)

    def test__wilber(self):
        my_test_algorithm(self, partial(_wilber, stable=False))

    def test__wilber_stable(self):
        my_test_algorithm(self, partial(_wilber, stable=True))

    def test__simple_dynamic_program(self):
        my_test_algorithm(self, _simple_dynamic_program)

    def test__simple_dynamic_program_stable(self):
        my_test_algorithm(self, partial(_simple_dynamic_program, stable=True))

    def test_wilber_edu(self):
        my_test_algorithm(self, partial(wilber_edu, should_print=False))

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

# Testing ranges because they sometimes have no unique solution
class TestRange5(Test8Elements):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.arange(5, dtype=np.float64)
        self.solutions = {
            1 : np.arange(5),
            2 : np.array([0, 0, 1, 1, 1]),
        }

class TestRange6(Test8Elements):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.arange(6, dtype=np.float64)
        self.solutions = {
            1 : np.arange(6),
            2 : np.array([0, 0, 1, 1, 2, 2]),
        }

class TestRange7(Test8Elements):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.arange(7, dtype=np.float64)
        self.solutions = {
            1 : np.arange(7),
            2 : np.array([0, 0, 1, 1, 2, 2, 2]),
        }




class TestAgreement(Test8Elements):
    """Tests to ensure that _simple_dynamic_program and wilber produce the same clusterings
    but clusterings were not the same!
    """

    def assert_agreement(self, arr, k):
        arr.sort()

        result1 = wilber(arr.copy(), k)
        result2 = _simple_dynamic_program(arr.copy(), k)

        cost1 = compute_cluster_cost_sorted(arr, result1)
        cost2 = compute_cluster_cost_sorted(arr, result2)
        self.assertEqual(cost1, cost2)

        assert_array_equal(result1, result2)

        #print(result1)
        #print(result2)
    def test_1(self):
        np.random.seed(0)
        arr = np.random.rand(1_000_000)
        self.assert_agreement(arr, k=2)


    def test_2(self):
        arr = np.arange(1000001, dtype=np.float64)
        self.assert_agreement(arr, k=2)

    def test_3(self):
        result = _simple_dynamic_program(np.arange(500_000, dtype=np.float64), 2, stable=True)
        expected_result = np.repeat(np.arange(250_000), 2)
        assert_array_equal(result, expected_result)

        with self.assertRaises(AssertionError): # weird test, but it makes sure that the stable version is still needed ...
            # if this issue is resolved for the default algorithm, the stable version might be cut
            result2 = _simple_dynamic_program(np.arange(500_000, dtype=np.float64), 2, False)
            assert_array_equal(result2, expected_result)

        result3 = _wilber(np.arange(500_000, dtype=np.float64), 2, stable=True)
        assert_array_equal(result3, expected_result)

# n=1000000 seed=0 k=5 does not agree!


if __name__ == '__main__':
    unittest.main()