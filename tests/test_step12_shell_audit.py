"""STEP12 독립 검증의 핵심 물리·수치 조건을 반복 검사한다."""
import unittest
import numpy as np
from verification.step12_shell_audit import tension, shear, mass_check, zero_plate_check, drilling_check, cantilever
from verification.step13_diaphragm_validity import diagnostic_builder
import openseespy.opensees as ops


class ShellAuditTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_affine_fields_and_modifier(self):
        for e in (2e7,2e4):
            for m in (0.,1.):
                for result in (tension(e,m),tension(e,m,-1),shear(e,m)):
                    self.assertLess(abs(result["error_pct"]),1e-7)

    def test_zero_plate_and_density(self):
        for e in (2e7,2e4):
            self.assertEqual(zero_plate_check(e)["max_out_of_plane_reaction"],0.)
        for rho in (0.,1.):
            result=mass_check(rho)
            self.assertLess(result["max_matrix_error"],1e-12)
            self.assertEqual(result["max_rotational_mass"],0.)

    def test_drilling_is_not_plate_bending(self):
        a=drilling_check(2e7,0.)["drilling_effective_stiffness_kNm"]
        b=drilling_check(2e7,1.)["drilling_effective_stiffness_kNm"]
        c=drilling_check(2e4,0.)["drilling_effective_stiffness_kNm"]
        np.testing.assert_allclose([a/b,a/c],[1.,1000.],rtol=1e-10)

    def test_plate_bending_convergence(self):
        coarse=cantilever(2e7,2,True,1.)
        refined=cantilever(2e7,8,True,1.)
        self.assertLess(abs(refined["error_pct"]),abs(coarse["error_pct"])/10)

    def test_diagnostic_topology(self):
        # 작은 모델에서 셸 제거 후 무연결 내부 절점이 없는지 확인한다.
        from verification.step07_3d_reference_frame import Parameters3D
        p=Parameters3D(stories=1,bays_x=1,bays_y=1)
        model=diagnostic_builder("flexible",p,2,variant="no_shell")
        linked={t for edge in model["edges"] for t in edge}
        self.assertEqual(set(ops.getNodeTags()),linked)
        self.assertFalse(model["shell_elements"])
        # 면외 판 진단에서는 내부 절점 고정을 실제로 해제했는지 확인한다.
        model=diagnostic_builder("flexible",p,2,variant="full_plate")
        linked={t for edge in model["edges"] for t in edge}
        interior=set(model["shell_levels"][0])-linked
        self.assertEqual(len(interior),1)
        self.assertFalse(set(ops.getFixedNodes()) & interior)


if __name__=="__main__":
    unittest.main()
