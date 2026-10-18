"""STEP09의 질량 보존, 편심 관성결합 및 대칭 극한을 확인한다."""
import unittest
import numpy as np

from verification.step09_mass_eccentricity import (
    P, mass_weights, mass_matrix, solve_case,
)


class MassEccentricityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xy=np.array([(i*P.bay_length-P.lengths[0]/2,j*P.bay_length-P.lengths[1]/2)
                         for j in range(P.bays_y+1) for i in range(P.bays_x+1)])

    def test_exact_edge_and_near_corner_centers(self):
        for direction,alpha,expected in (("x_edge",1,(9,0)),("y_edge",1,(0,6)),
                                          ("corner",.99,(8.91,5.94))):
            w=mass_weights(self.xy,direction,alpha)
            self.assertAlmostEqual(sum(w),1)
            m,center,jc=mass_matrix(w,self.xy,P.floor_mass)
            np.testing.assert_allclose(center,expected,atol=1e-12)
            self.assertGreater(jc,0)
            self.assertGreater(min(np.linalg.eigvalsh(m)),0)
            self.assertAlmostEqual(m[0,2],-P.floor_mass*center[1])
            self.assertAlmostEqual(m[1,2],P.floor_mass*center[0])
        with self.assertRaises(ValueError):
            mass_weights(self.xy,"corner",1)

    def test_openSees_mass_and_baseline_are_consistent(self):
        baseline=solve_case("corner",0)
        eccentric=solve_case("corner",.99)
        np.testing.assert_allclose(baseline["periods_s"][:3],
                                   [.3319769918853128,.30667971078484557,.2735348160863736],rtol=1e-8)
        self.assertLess(max(abs(x) for x in baseline["center_xy_m"]),1e-12)
        self.assertLess(baseline["roof_rotation_displacement_share"][0],1e-10)
        self.assertGreater(eccentric["roof_rotation_displacement_share"][0],.3)
        self.assertGreater(eccentric["periods_s"][0],baseline["periods_s"][0])
        self.assertAlmostEqual(eccentric["opensees_total_free_mass"][0],3*P.floor_mass)
        self.assertAlmostEqual(eccentric["gravity"]["reaction_kN"],3*P.floor_weight)


if __name__=="__main__":
    unittest.main()
