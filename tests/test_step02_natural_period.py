import math
import unittest

from verification.step01_static_stiffness import PortalFrameParameters
from verification.step02_natural_period import (
    NaturalPeriodParameters,
    run_natural_period_verification,
    validate_result,
)


class NaturalPeriodVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = PortalFrameParameters()
        cls.dynamic = NaturalPeriodParameters()
        cls.result = run_natural_period_verification(cls.frame, cls.dynamic)

    def test_weight_is_converted_to_mass(self):
        self.assertTrue(
            math.isclose(
                self.result.floor_mass,
                self.dynamic.floor_weight / self.dynamic.gravity,
                rel_tol=1.0e-14,
            )
        )

    def test_openSees_mass_equals_input_floor_mass(self):
        self.assertTrue(
            math.isclose(
                self.result.total_free_mass_x,
                self.result.floor_mass,
                rel_tol=1.0e-13,
            )
        )

    def test_openSees_period_matches_static_sdof_period(self):
        validate_result(self.result)

    def test_axial_deformation_lengthens_period(self):
        self.assertGreater(self.result.opensees_period, self.result.textbook_period)


if __name__ == "__main__":
    unittest.main()

