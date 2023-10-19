import unittest
from functools import partial
import numpy as np
from microagg1d.cost_maxdist import MaxDistCostCalculator, AdaptedMaxDistCostCalculator
from microagg1d.common import compute_cluster_cost_sorted
from microagg1d.user_facing import _maxdist_user
from microagg1d.main import optimal_univariate_microaggregation_1d



class TestMedianCosts(unittest.TestCase):
    def test_maxdist(self):
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        F_vals = np.zeros_like(arr)
        calculators = [
            MaxDistCostCalculator(arr),
            AdaptedMaxDistCostCalculator(arr, 1, F_vals)
        ]
        for calculator in calculators:
            for ub, value in zip(range(1, len(arr)), [0,1,2,3,4,5]):
                if hasattr(calculator, 'k'):
                    calculator.k = ub
                self.assertAlmostEqual(calculator.calc(0, ub), value)



def my_test_algorithm(self, algorithm):
    for k, solution in self.solutions.items():
        result = algorithm(self.arr, k)
        calculator = MaxDistCostCalculator(self.arr)
        c_sol = compute_cluster_cost_sorted(np.array(solution,dtype=np.int64), calculator)
        c_res = compute_cluster_cost_sorted(result, calculator)
        np.testing.assert_array_equal(result, solution, f"k={k} C_sol={c_sol} c_res={c_res}")



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

    def test_maxdist_wilber(self):
        my_test_algorithm(self, partial(_maxdist_user, algorithm="wilber"))

    def test_maxdist_galil(self):
        my_test_algorithm(self, partial(_maxdist_user, algorithm="galil_park"))

    def test_maxdist_simple(self):
        my_test_algorithm(self, partial(_maxdist_user, algorithm="simple"))

    def test_maxdist_staggered(self):
        my_test_algorithm(self, partial(_maxdist_user, algorithm="staggered"))

    def test_maxdist_main(self):
        my_test_algorithm(self, partial(optimal_univariate_microaggregation_1d, cost="maxdist"))


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
    """This is a test where maxdist and SSE disagree on the optimal clustering"""
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
            5 : np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            6 : np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            7 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
            8 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        }



if __name__ == '__main__':
    unittest.main()