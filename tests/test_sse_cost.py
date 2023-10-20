import unittest

import numpy as np

from microagg1d.common import calc_cumsum
from microagg1d.cost_sse import (
    AdaptedSSECostCalculator,
    FasterAdaptedSSECostCalculator,
    FasterSSECostCalculator,
    NoPrecomputeSSECostCalculator,
    SSECostCalculator,
    StableAdaptedSSECostCalculator,
    StableSSECostCalculator,
)


def get_sse_calculators(v, k):
    cumsum = calc_cumsum(v)
    cumsum2 = calc_cumsum(np.square(v))
    F_vals = np.zeros_like(v)
    return [
        SSECostCalculator(v),
        NoPrecomputeSSECostCalculator(v),
        StableSSECostCalculator(v, 2 * k),
        AdaptedSSECostCalculator(cumsum, cumsum2, k, F_vals),
        StableAdaptedSSECostCalculator(v, k, F_vals, 2 * k),
    ], [FasterSSECostCalculator(v), FasterAdaptedSSECostCalculator(cumsum, k, F_vals)]


class RegularizedKmeans(unittest.TestCase):
    def test_size1_cluster(self):
        faster_results = -np.square(np.arange(10))
        self.standard_procedure(
            cluster_size=1, normal_result=0.0, faster_results=faster_results
        )

    def test_size2_cluster(self):
        faster_results = [-0.5, -4.5, -12.5, -24.5, -40.5, -60.5, -84.5, -112.5]
        self.standard_procedure(
            cluster_size=2, normal_result=0.5, faster_results=faster_results
        )

    def test_size3_cluster(self):
        faster_results = [-3.0, -12.0, -27.0, -48.0, -75.0, -108.0, -147.0]
        self.standard_procedure(
            cluster_size=3, normal_result=2.0, faster_results=faster_results
        )

    def test_size4_cluster(self):
        faster_results = [-9.0, -25.0, -49.0, -81.0, -121.0, -169.0]
        self.standard_procedure(
            cluster_size=4, normal_result=5.0, faster_results=faster_results
        )

    def standard_procedure(self, cluster_size, normal_result, faster_results):
        v = np.arange(10, dtype=np.float64)
        calculators, faster_calculators = get_sse_calculators(v, k=cluster_size)

        for i in range(len(v) - cluster_size):
            for calculator in calculators:
                self.assertAlmostEqual(
                    calculator.calc(i, i + cluster_size),
                    normal_result,
                    msg=f"{i} {calculator.__class__}",
                )

        l = []
        for i, result in zip(range(len(v) - cluster_size), faster_results):
            for calculator in faster_calculators:
                l.append(calculator.calc(i, i + cluster_size))
                #        break
                # print(l)
                self.assertAlmostEqual(
                    calculator.calc(i, i + cluster_size),
                    result,
                    msg=f"{i} {calculator.__class__}",
                )


if __name__ == "__main__":
    unittest.main()
