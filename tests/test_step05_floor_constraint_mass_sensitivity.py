import math
import unittest

import numpy as np

from verification.step01_static_stiffness import PortalFrameParameters
from verification.step05_floor_constraint_mass_sensitivity import (
    Step05Parameters,
    run_step05_verification,
    validate_results,
)


class FloorConstraintMassSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = PortalFrameParameters()
        cls.parameters = Step05Parameters()
        cls.results = run_step05_verification(cls.frame, cls.parameters)

    def test_all_cases_pass_validation(self):
        validate_results(self.results, self.parameters)

    def test_total_mass_is_unchanged(self):
        expected = sum(self.parameters.floor_weights) / self.parameters.gravity
        for result in self.results.values():
            self.assertTrue(
                math.isclose(result.total_free_mass_x, expected, rel_tol=1.0e-12)
            )

    def test_top_floor_eccentricity_and_mass_fractions(self):
        centered = self.results["untied_centered"]
        eccentric = self.results["untied_top_eccentric"]
        self.assertAlmostEqual(centered.top_mass_eccentricity_ratio, 0.0, places=13)
        self.assertAlmostEqual(
            eccentric.top_mass_eccentricity_ratio,
            self.parameters.top_eccentricity_ratio,
            places=13,
        )
        self.assertTrue(
            np.allclose(
                eccentric.mass_fractions_by_floor[-1],
                [0.07, 0.19, 0.31, 0.43],
                atol=1.0e-13,
            )
        )

    def test_tied_periods_ignore_nodal_mass_location(self):
        centered = self.results["tied_centered"]
        eccentric = self.results["tied_top_eccentric"]
        self.assertTrue(np.allclose(centered.periods, eccentric.periods, rtol=1.0e-11))

    def test_eccentric_mass_distribution_affects_untied_periods(self):
        centered = np.asarray(self.results["untied_centered"].periods)
        eccentric = np.asarray(self.results["untied_top_eccentric"].periods)
        relative_change = np.abs((eccentric - centered) / centered)
        self.assertGreater(float(np.max(relative_change)), 1.0e-4)

    def test_tied_and_untied_compatibility(self):
        tied = self.results["tied_centered"].static
        untied = self.results["untied_centered"].static
        self.assertLess(max(tied.floor_spreads), 1.0e-13)
        self.assertGreater(untied.floor_spreads[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
