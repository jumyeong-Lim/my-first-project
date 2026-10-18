"""STEP10의 질량 대조·질량중심 모드 분해·강체 기준 재현 검증."""
import unittest
import numpy as np
from verification.step10_diaphragm_mass_interaction import mass_cases,XY,P,characterize,modal
from verification.step09_mass_eccentricity import mass_matrix,solve_case


class InteractionTest(unittest.TestCase):
    def test_same_center_controls_change_inertia(self):
        cases=mass_cases();inertia=[]
        for key in ("C50_Jlo","C50","C50_Jhi"):
            weights=cases[key]
            self.assertAlmostEqual(sum(weights),1)
            self.assertTrue(np.all(weights>0))
            _,center,jc=mass_matrix(weights,XY,P.floor_mass)
            np.testing.assert_allclose(center,[4.5,3],atol=1e-10)
            inertia.append(jc)
        self.assertTrue(inertia[0]<inertia[1]<inertia[2])

    def test_mass_center_rotation_has_no_translation_or_distortion_energy(self):
        weights=mass_cases()["C50"]
        _,center,jc=mass_matrix(weights,XY,P.floor_mass)
        xy=XY-np.asarray(center)
        vector=np.tile(np.column_stack((-xy[:,1],xy[:,0])).ravel(),P.stories)
        vector/=np.sqrt(P.stories*jc)
        result=characterize(vector,weights)
        np.testing.assert_allclose(result["energy_X_Y_Rz_distortion"],[0,0,1,0],atol=1e-10)

    def test_rigid_corner_matches_step09(self):
        result=modal("R",mass_cases()["C99"])
        prior=solve_case("corner",.99)
        np.testing.assert_allclose(result["periods_s"],prior["periods_s"],rtol=1e-8)
        np.testing.assert_allclose(result["character"][0]["energy_X_Y_Rz_distortion"][:3],
                                   prior["energy_fraction_X_Y_Rz"][0],atol=1e-8)


if __name__=="__main__":unittest.main()
