import math
import unittest

import numpy as np

from verification.step01_static_stiffness import PortalFrameParameters
from verification.step04_three_story_multibay_frame import (
    MultiBayParameters,
    run_multibay_verification,
    validate_results,
)


class ThreeStoryMultiBayFrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = PortalFrameParameters()
        cls.parameters = MultiBayParameters()
        cls.results = run_multibay_verification(cls.frame, cls.parameters)

    def test_model_counts(self):
        for bays, result in self.results.items():
            expected_nodes = 4 * (bays + 1)
            expected_columns = 3 * (bays + 1)
            expected_beams = 3 * bays
            self.assertEqual(result.node_count, expected_nodes)
            self.assertEqual(result.column_count, expected_columns)
            self.assertEqual(result.beam_count, expected_beams)
            self.assertEqual(result.element_count, expected_columns + expected_beams)

    def test_condensed_and_opensees_results_match(self):
        validate_results(self.results)

    def test_mass_is_fixed_for_all_bay_counts(self):
        expected_mass = sum(self.parameters.floor_weights) / self.parameters.gravity
        for result in self.results.values():
            self.assertTrue(
                math.isclose(result.total_free_mass_x, expected_mass, rel_tol=1.0e-13)
            )

    def test_condensed_stiffness_is_positive_definite(self):
        for result in self.results.values():
            eigenvalues = np.linalg.eigvalsh(
                np.asarray(result.condensed_stiffness_matrix)
            )
            self.assertTrue(np.all(eigenvalues > 0.0))

    def test_more_bays_increase_stiffness_and_reduce_first_period(self):
        stiffnesses = [self.results[bays].roof_lateral_stiffness for bays in (1, 2, 3)]
        periods = [self.results[bays].modes[0].opensees_period for bays in (1, 2, 3)]
        self.assertLess(stiffnesses[0], stiffnesses[1])
        self.assertLess(stiffnesses[1], stiffnesses[2])
        self.assertGreater(periods[0], periods[1])
        self.assertGreater(periods[1], periods[2])


if __name__ == "__main__":
    unittest.main()
