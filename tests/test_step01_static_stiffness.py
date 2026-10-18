import math
import unittest

from verification.step01_static_stiffness import (
    PortalFrameParameters,
    flexible_beam_limit,
    rigid_beam_limit,
    run_benchmark_cases,
    validate_results,
)


class StaticStiffnessVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = PortalFrameParameters()
        cls.results = run_benchmark_cases(cls.params)

    def test_section_properties_are_positive(self):
        self.assertGreater(self.params.beam.area, 0.0)
        self.assertGreater(self.params.beam.second_moment_of_area, 0.0)
        self.assertGreater(self.params.column.area, 0.0)
        self.assertGreater(self.params.column.second_moment_of_area, 0.0)

    def test_textbook_limits_have_expected_ratio(self):
        self.assertTrue(
            math.isclose(
                rigid_beam_limit(self.params),
                4.0 * flexible_beam_limit(self.params),
                rel_tol=1.0e-14,
            )
        )

    def test_openSees_matches_closed_form_benchmark(self):
        validate_results(self.results)

    def test_assumed_section_axial_deformation_reduces_stiffness(self):
        ideal = self.results[1]
        assumed = self.results[3]
        self.assertLess(assumed.opensees_stiffness, ideal.opensees_stiffness)
        self.assertGreater(assumed.opensees_stiffness, 0.99 * ideal.opensees_stiffness)


if __name__ == "__main__":
    unittest.main()
