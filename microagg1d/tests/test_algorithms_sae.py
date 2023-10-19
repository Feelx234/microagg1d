import unittest
import numpy as np
from microagg1d.sae_cost import SAECostCalculator, calc_sorted_median
from microagg1d.sse_cost import SSECostCalculator, AdaptedSSECostCalculator
from microagg1d.common import calc_cumsum
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
calculator = SSECostCalculator(arr)
print(calculator.calc(0, len(arr)))
print(calculator.calc(0, 3))
print(calculator.calc(0, 2))
print(calculator.calc(0, 1))

cumsum = calc_cumsum(arr)
cumsum2 = calc_cumsum(np.square(arr))
calculator = AdaptedSSECostCalculator(cumsum, cumsum2, 2, np.zeros(len(arr)+1, dtype=np.float64))
print(calculator.calc(3, 0))
print(calculator.calc(2, 0))
print(calculator.calc(1, 0))

class MedianCalculation(unittest.TestCase):
    def test_sae(self):
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        F_vals = np.zeros_like(arr)
        calculator = SAECostCalculator(arr, F_vals)
        self.assertAlmostEqual(calculator.calc(5, 0), 6.0)
        self.assertAlmostEqual(calculator.calc(4, 0), 4.0)
        self.assertAlmostEqual(calculator.calc(5, 1), 4.0)
        self.assertAlmostEqual(calculator.calc(4, 1), 2.0)
        self.assertAlmostEqual(calculator.calc(2, 4), 1.0)



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


if __name__ == '__main__':
    unittest.main()