import unittest
from functools import partial

import numpy as np

from microagg1d.common import compute_cluster_cost_sorted
from microagg1d.cost_sae import AdaptedSAECostCalculator, SAECostCalculator
from microagg1d.user_facing import _sae_user
from microagg1d.utils_for_test import remove_from_class, restore_to_class


class TestMedianCosts(unittest.TestCase):
    def test_sae(self):
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        F_vals = np.zeros_like(arr)
        calculators = [SAECostCalculator(arr), AdaptedSAECostCalculator(arr, 1, F_vals)]
        for calculator in calculators:
            for ub, value in zip(range(1, len(arr)), [0, 1, 2, 4, 6, 9]):
                if hasattr(calculator, "k"):
                    calculator.k = ub
                self.assertAlmostEqual(calculator.calc(0, ub), value)


def my_test_algorithm(self, algorithm):
    for k, solution in self.solutions.items():
        result = algorithm(self.arr, k)
        calculator = SAECostCalculator(self.arr)
        c_sol = compute_cluster_cost_sorted(
            np.array(solution, dtype=np.int64), calculator
        )
        c_res = compute_cluster_cost_sorted(result, calculator)
        np.testing.assert_array_equal(
            result, solution, f"k={k} C_sol={c_sol} c_res={c_res}"
        )


class Test8Elements(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.array([1.1, 1.2, 1.3, 1.4, 5, 5, 5, 5])
        self.solutions = {
            1: [0, 1, 2, 3, 4, 5, 6, 7],
            2: [0, 0, 1, 1, 2, 2, 3, 3],
            3: [0, 0, 0, 0, 1, 1, 1, 1],
            4: [0, 0, 0, 0, 1, 1, 1, 1],
            5: [0, 0, 0, 0, 0, 0, 0, 0],
        }

    def test_sae_wilber(self):
        my_test_algorithm(self, partial(_sae_user, algorithm="wilber"))

    def test_sae_galil(self):
        my_test_algorithm(self, partial(_sae_user, algorithm="galil_park"))

    def test_sae_simple(self):
        my_test_algorithm(self, partial(_sae_user, algorithm="simple"))

    def test_sae_staggered(self):
        my_test_algorithm(self, partial(_sae_user, algorithm="staggered"))


class Test7Elements(Test8Elements):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.array([1.1, 1.2, 1.3, 1.4, 5, 5, 5])
        self.solutions = {
            1: [0, 1, 2, 3, 4, 5, 6],
            2: [0, 0, 1, 1, 2, 2, 2],
            3: [0, 0, 0, 0, 1, 1, 1],
            4: [0, 0, 0, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 0, 0],
        }


class TestArray(Test8Elements):
    """This is a test where SAE and SSE disagree on the optimal clustering"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arr = np.array(
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
        self.solutions = {
            1: np.arange(15),
            2: np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6]),
            3: np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
            4: np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]),
            5: np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            6: np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            7: np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            8: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        }


class Test8ElementsNonCompiled(Test8Elements):
    def setUp(self):
        self.cleanup = remove_from_class(
            self.__class__.__bases__[0], allowed_packages=["microagg1d"]
        )

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)


class TestArrayElementsNonCompiled(TestArray):
    def setUp(self):
        self.cleanup = remove_from_class(
            self.__class__.__bases__[0], allowed_packages=["microagg1d"]
        )

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)


if __name__ == "__main__":
    unittest.main()
