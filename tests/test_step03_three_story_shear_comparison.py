import math
import unittest

import numpy as np

from verification.step01_static_stiffness import PortalFrameParameters
from verification.step03_three_story_shear_comparison import (
    ThreeStoryParameters,
    assemble_shear_matrices,
    run_three_story_comparison,
    validate_result,
)


class ThreeStoryShearComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = PortalFrameParameters()
        cls.parameters = ThreeStoryParameters()
        cls.result = run_three_story_comparison(cls.frame, cls.parameters)

    def test_uniform_shear_matrix_is_assembled_correctly(self):
        stiffness, mass = assemble_shear_matrices(
            np.array([10.0, 10.0, 10.0]), np.array([2.0, 2.0, 2.0])
        )
        np.testing.assert_allclose(
            stiffness,
            np.array([[20.0, -10.0, 0.0], [-10.0, 20.0, -10.0], [0.0, -10.0, 10.0]]),
        )
        np.testing.assert_allclose(mass, np.diag([2.0, 2.0, 2.0]))

    def test_three_modes_and_constraints_are_valid(self):
        validate_result(self.result)

    def test_openSees_mass_equals_three_floor_masses(self):
        self.assertTrue(
            math.isclose(
                self.result.total_free_mass_x,
                sum(self.result.floor_masses),
                rel_tol=1.0e-13,
            )
        )

    def test_periods_descend_with_mode_number(self):
        shear = [mode.shear_period for mode in self.result.modes]
        frame = [mode.frame_period for mode in self.result.modes]
        self.assertGreater(shear[0], shear[1])
        self.assertGreater(shear[1], shear[2])
        self.assertGreater(frame[0], frame[1])
        self.assertGreater(frame[1], frame[2])


if __name__ == "__main__":
    unittest.main()
