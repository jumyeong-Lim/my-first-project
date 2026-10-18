"""3D 모델의 질량변환·모드·중력효과 및 독립적인 탄성 해를 검증한다."""
import unittest
import numpy as np
import openseespy.opensees as ops

from verification.step01_static_stiffness import PortalFrameParameters, run_static_case
from verification.step07_3d_reference_frame import (
    Parameters3D, layout, run_verification, validate_case, configure_static,
)


class ThreeDimensionalReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_verification()

    def test_gravity_mass_constraints_and_modal_consistency(self):
        for result in self.results.values():
            validate_case(result)

    def test_three_modes_per_family(self):
        families = self.results["reference_PDelta_g1"]["family"]
        for family in ("X", "Y", "Rz"):
            self.assertEqual(families.count(family), 3)

    def test_polar_inertia_for_four_corner_masses(self):
        # 한 변 6 m인 바닥 네 모서리 질량의 해석해이다.
        p = Parameters3D(stories=1, bays_x=1, bays_y=1)
        _, mass = layout(p)
        self.assertAlmostEqual(mass[2], p.floor_mass*(3**2+3**2))

    def test_small_frame_x_stiffness_equals_two_parallel_2d_portals(self):
        # 대칭 X 병진에서 직교 보의 추가 변형이 없으므로 두 2D 골조가 병렬 작용한다.
        f = PortalFrameParameters()
        reference = run_static_case("step07_parallel_portals", f,
                                    beam_inertia_factor=1.0, axial_area_factor=1.0)
        stiffness = self.results["small_Linear_g0"]["condensed_stiffness"][0][0]
        self.assertAlmostEqual(stiffness/(2*reference.opensees_stiffness), 1.0, places=10)

    def test_column_local_axes_against_cantilever_closed_form(self):
        # 로컬축 오류는 고유치 비교만으로 발견하기 어렵다. 외팔보의 독립 이론해와 비교한다.
        p, f = Parameters3D(), PortalFrameParameters()
        length = p.story_height
        shear = f.elastic_modulus/(2*(1+p.poisson))
        for dof, expected in ((1, length**3/(3*f.elastic_modulus*f.column.second_moment_of_area)),
                              (2, length**3/(3*f.elastic_modulus*p.column_iy)),
                              (6, length/(shear*p.column_j))):
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            ops.node(1, 0, 0, 0)
            ops.node(2, 0, 0, length)
            ops.fix(1, 1, 1, 1, 1, 1, 1)
            ops.geomTransf("Linear", 1, 0, 1, 0)
            ops.element("elasticBeamColumn", 1, 1, 2, f.column.area, f.elastic_modulus,
                        shear, p.column_j, p.column_iy, f.column.second_moment_of_area, 1)
            ops.timeSeries("Linear", 1)
            ops.pattern("Plain", 1, 1)
            force = [0.0]*6
            force[dof-1] = 1.0
            ops.load(2, *force)
            configure_static(1.0)
            self.assertEqual(ops.analyze(1), 0)
            self.assertAlmostEqual(ops.nodeDisp(2, dof)/expected, 1.0, places=10)
        ops.wipe()

    def test_gravity_changes_only_pdelta_periods(self):
        for label in ("small", "reference"):
            zero = self.results[f"{label}_Linear_g0"]["periods"]
            np.testing.assert_allclose(self.results[f"{label}_PDelta_g0"]["periods"], zero, rtol=1e-10)
            np.testing.assert_allclose(self.results[f"{label}_Linear_g1"]["periods"], zero, rtol=1e-10)
            self.assertTrue(np.all(np.array(self.results[f"{label}_PDelta_g1"]["periods"]) > zero))


if __name__ == "__main__":
    unittest.main()
