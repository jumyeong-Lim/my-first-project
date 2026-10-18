import math
import unittest

import numpy as np

from verification.step01_static_stiffness import PortalFrameParameters
from verification.step06_linear_pdelta_sensitivity import (
    Step06Parameters,
    run_step06_verification,
    validate_results,
)


class LinearPDeltaSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = PortalFrameParameters()
        cls.parameters = Step06Parameters()
        cls.results = run_step06_verification(cls.frame, cls.parameters)

    def test_all_cases_pass_validation(self):
        validate_results(self.results, self.parameters)

    def test_gravity_reactions_match_applied_weight(self):
        for result in self.results.values():
            self.assertLess(abs(result.gravity_reaction_error_percent), 1.0e-8)

    def test_mass_is_unchanged(self):
        expected = sum(self.parameters.floor_weights) / self.parameters.gravity
        for result in self.results.values():
            self.assertTrue(
                math.isclose(result.total_free_mass_x, expected, rel_tol=1.0e-12)
            )

    def test_transformations_match_without_gravity(self):
        linear = self.results["linear_g0.00"]
        pdelta = self.results["pdelta_g0.00"]
        self.assertTrue(np.allclose(linear.periods, pdelta.periods, rtol=1.0e-11))
        self.assertTrue(
            math.isclose(
                linear.roof_tangent_stiffness,
                pdelta.roof_tangent_stiffness,
                rel_tol=1.0e-10,
            )
        )

    def test_reference_gravity_pdelta_softens_frame(self):
        linear = self.results["linear_g1.00"]
        pdelta = self.results["pdelta_g1.00"]
        self.assertLess(pdelta.roof_tangent_stiffness, linear.roof_tangent_stiffness)
        self.assertGreater(pdelta.periods[0], linear.periods[0])

    def test_pdelta_effect_increases_with_gravity(self):
        periods = [
            self.results[f"pdelta_g{factor:.2f}"].periods[0]
            for factor in self.parameters.gravity_factors
        ]
        stiffnesses = [
            self.results[f"pdelta_g{factor:.2f}"].roof_tangent_stiffness
            for factor in self.parameters.gravity_factors
        ]
        self.assertTrue(np.all(np.diff(periods) > 0.0))
        self.assertTrue(np.all(np.diff(stiffnesses) < 0.0))


if __name__ == "__main__":
    unittest.main()
