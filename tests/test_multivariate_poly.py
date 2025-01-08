import unittest

import numpy as np
from numpy.testing import assert_array_equal

from microagg1d.multivariate_poly import forest2, set_both_seeds


class TestMultivariatePoly(unittest.TestCase):
    def get_example_1(self):
        return np.array(
            [[1, 2], [2, 0], [1, 0], [4, 5], [5, 3], [3, 4]], dtype=np.int64
        )

    def test_forest2_simple1(self):
        set_both_seeds(2)
        closest_neighbors = self.get_example_1()
        partitions, edges = forest2(closest_neighbors)
        self.assertEqual(len(partitions), 2)
        assert_array_equal(partitions[0], [1, 0, 2])
        assert_array_equal(partitions[1], [3, 5, 4])

        assert_array_equal(edges, [[0, 1], [5, 3], [2, 1], [4, 5]])

    def test_forest2_simple2(self):
        set_both_seeds(3)
        closest_neighbors = self.get_example_1()
        partitions, edges = forest2(closest_neighbors)
        self.assertEqual(len(partitions), 2)
        assert_array_equal(partitions[0], [1, 2, 0])
        assert_array_equal(partitions[1], [4, 3, 5])

        assert_array_equal(edges, [[2, 1], [0, 1], [3, 4], [5, 3]])


if __name__ == "__main__":
    unittest.main()
