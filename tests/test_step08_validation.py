"""메시 확대 시 추가한 수치 절차를 독립적인 조건으로 검증한다."""
import unittest
import numpy as np
import openseespy.opensees as ops
from verification.step08_validation import floor_metrics, modal, configure_sparse


class DiaphragmValidationTest(unittest.TestCase):
    def test_rigid_translation_and_rotation_have_zero_distortion(self):
        xy = np.array([[-3,-3],[3,-3],[3,3],[-3,3]], dtype=float)
        motion = np.column_stack((.02-.001*xy[:,1], -.03+.001*xy[:,0]))
        self.assertLess(floor_metrics(xy, motion)["distortion_rms"], 1e-15)

    def test_membrane_patch_matches_uniaxial_stress(self):
        # 포아송 수축을 허용하는 일정한 축변형의 경계력은 E*h*L*변형률이다.
        ops.wipe();ops.model("basic","-ndm",3,"-ndf",6)
        xy=[(0,0),(6,0),(6,6),(0,6)]
        for tag,(x,y) in enumerate(xy,1):
            ops.node(tag,x,y,0);ops.fix(tag,0,0,1,1,1,1)
        ops.section("ElasticMembranePlateSection",1,2e4,.2,.12,0,0)
        ops.element("ShellMITC4",1,1,2,3,4,1)
        ops.timeSeries("Linear",1);ops.pattern("Plain",1,1)
        for tag,(x,y) in enumerate(xy,1):
            ops.sp(tag,1,1e-4*x);ops.sp(tag,2,-.2e-4*y)
        configure_sparse(1)
        self.assertEqual(ops.analyze(1),0)
        ops.reactions()
        self.assertAlmostEqual(ops.nodeReaction(2,1)+ops.nodeReaction(3,1),2e4*.12*6*1e-4,places=9)
        ops.wipe()

    def test_refined_mesh_preserves_mass_and_solver_eigenvalues(self):
        a=modal(2e4,2,count=24)
        b=modal(2e4,2,count=24,solver="-fullGenLapack")
        np.testing.assert_allclose(a["periods"],b["periods"],rtol=1e-7)
        self.assertLess(a["mass_orthogonality_error"],1e-7)
        self.assertGreater(min(a["cumulative_participation"][:2]),.99)


if __name__ == "__main__":
    unittest.main()
