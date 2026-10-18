"""STEP08의 질량 보존, 강체 극한, 바닥판 변형 및 모드 대응을 검증한다."""
import unittest

import numpy as np

from verification.step07_3d_reference_frame import build_model
from verification.step08_diaphragm_inplane_sensitivity import (
    PARAMETERS, build_finite_case, run_verification,
)


class DiaphragmInplaneSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_verification()

    def test_same_gravity_and_horizontal_mass(self):
        expected_weight = PARAMETERS.floor_weight*PARAMETERS.stories
        expected_mass = PARAMETERS.floor_mass*PARAMETERS.stories
        for case in self.results.values():
            self.assertAlmostEqual(case["modal"]["gravity"]["reaction_kN"], expected_weight, places=7)
            np.testing.assert_allclose(case["modal"]["opensees_total_free_mass"][:2], expected_mass)
            np.testing.assert_allclose(case["modal"]["opensees_center_of_mass"][:2], 0.0, atol=1e-10)

    def test_actual_shells_exist_only_in_finite_diaphragm(self):
        shell = build_finite_case("semirigid")
        self.assertEqual(len(shell["shell_elements"]),
                         PARAMETERS.stories*PARAMETERS.bays_x*PARAMETERS.bays_y)
        beams_only = build_finite_case("floor_beams_only")
        self.assertEqual(len(beams_only["shell_elements"]), 0)
        self.assertEqual(shell["edges"], beams_only["edges"])
        baseline = build_model(PARAMETERS, transformation="PDelta")
        self.assertEqual(shell["edges"], baseline["edges"])
        self.assertEqual(shell["coords"],
                         {tag: baseline["coords"][tag] for tag in shell["coords"]})

    def test_large_shell_stiffness_converges_to_rigid_constraint(self):
        reference = np.array(self.results["rigid"]["modal"]["periods"])
        matched = self.results["rigid_limit"]["match_to_rigid"]
        np.testing.assert_allclose([x["period_s"] for x in matched], reference, rtol=2e-5)
        self.assertGreater(min(x["mac"] for x in matched), 0.99999)

    def test_inplane_roof_distortion_increases_as_shell_softens(self):
        for direction in ("static_x", "static_y"):
            deformation = [self.results[name][direction]["roof_rms_inplane_distortion_mm"]
                           for name in ("rigid", "semirigid", "flexible", "floor_beams_only")]
            self.assertLess(deformation[0], 1e-10)
            self.assertTrue(all(a < b for a, b in zip(deformation, deformation[1:])))

    def test_torsional_modes_change_and_modal_order_is_tracked(self):
        flexible = self.results["flexible"]["match_to_rigid"]
        # 다섯 번째 강체모드는 유연 조건의 아홉 번째 모드와 대응한다.
        fifth = flexible[4]
        self.assertEqual(fifth["finite_mode"], 9)
        self.assertGreater(fifth["period_change_pct"], 5.0)
        self.assertGreater(fifth["roof_inplane_mode_ratio"], 0.4)
        self.assertLess(fifth["mac"], 0.9)


if __name__ == "__main__":
    unittest.main()
