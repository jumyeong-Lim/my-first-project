"""물성에서 유도한 막 구성식과 모델 교체 범위를 검증한다."""
import unittest
import numpy as np
import openseespy.opensees as ops
from verification.step14_material_based_diaphragm import (
    SLABS, Slab, axial_test, constitutive_test, material_builder, use_material,
)
from verification import step10_diaphragm_mass_interaction as original
from verification.step07_3d_reference_frame import Parameters3D


class MaterialDiaphragmTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_independent_closed_form_targets(self):
        spec = SLABS["RC150_a30"]
        self.assertAlmostEqual(spec.E*spec.thickness_m, 1350000.)
        self.assertAlmostEqual(spec.A[2,2], 562500.)
        self.assertAlmostEqual(spec.A[0,0], 1406250.)
        for d, target in ((0,675000.),(1,2700000.)):
            result = axial_test(spec,2,d)
            self.assertAlmostEqual(result["FE_kN_m"]/target,1.,places=8)
            self.assertLess(abs(result["equilibrium_error_kN"]),1e-8)

    def test_full_constitutive_matrix(self):
        for spec in SLABS.values():
            result = constitutive_test(spec,2)
            np.testing.assert_allclose(result["A_FE_kN_m"],spec.A,rtol=1e-8,atol=1e-7)

    def test_mass_and_connection_preserved(self):
        p=Parameters3D(stories=1,bays_x=1,bays_y=1)
        model=material_builder(SLABS["RC150_a10"],"flexible",p,2)
        beam_nodes={t for edge in model["edges"] for t in edge}
        shell_nodes={t for elem in model["shell_elements"] for t in elem}
        self.assertEqual(len(beam_nodes & shell_nodes),8)
        self.assertAlmostEqual(sum(ops.nodeMass(t,1) for t in ops.getNodeTags()),p.floor_mass)
        saved=original.build_finite_case
        with use_material("RC150_a10"):
            self.assertIn("RC150_a10",original.DIAPHRAGMS)
        self.assertIs(original.build_finite_case,saved)
        self.assertNotIn("RC150_a10",original.DIAPHRAGMS)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError): Slab("bad",0.)
        with self.assertRaises(ValueError): Slab("bad",.3,thickness_m=-1.)


if __name__ == "__main__":
    unittest.main()
