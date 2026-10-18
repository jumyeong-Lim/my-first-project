"""좌표·부호·평형·대칭을 독립 외력으로 검산한다."""
import unittest
import numpy as np
import openseespy.opensees as ops
from verification.step15_frame_shear_distribution import axis_probe, calculate_case, original


class FrameShearTests(unittest.TestCase):
    def tearDown(self):
        ops.wipe()

    def test_single_column_axes_and_end_order(self):
        for t in ("Linear","PDelta"):
            for d in (0,1):
                for reverse in (False,True):
                    result=axis_probe(t,d,reverse)
                    self.assertAlmostEqual(result["lower_global_cut_shear_kN"],1.,places=10)
                    self.assertAlmostEqual(result["base_reaction_kN"],-1.,places=10)
                    if t=="Linear": self.assertAlmostEqual(result["mapped_local_shear_kN"],1.,places=10)
                    else: self.assertGreater(result["mapped_local_shear_kN"],1.)

    def test_rigid_uniform_equal_line_shares(self):
        for d,count in ((0,3),(1,4)):
            c=calculate_case("R","U",d)
            for r in c["line_results"]:
                self.assertAlmostEqual(r["signed_share_pct"],100/count,places=8)
            np.testing.assert_allclose([r["total_shear_kN"] for r in c["floor_summary"]],
                                       original.P.floor_mass*np.array([3,2,1]),atol=1e-9)

    def test_mirrored_mass_reverses_line_distribution(self):
        w=original.mass_cases()["C99"]
        # XY 배열의 역순은 중심에 대한 점대칭이다. X/Y 양 방향에서 반대 선을 비교한다.
        for d in (0,1):
            a=calculate_case("R","C99",d)
            b=calculate_case("R","C99",d,weights_override=w[::-1])
            for floor in (1,2,3):
                x=[r["V_kN"] for r in a["line_results"] if r["floor"]==floor]
                y=[r["V_kN"] for r in b["line_results"] if r["floor"]==floor]
                np.testing.assert_allclose(x,y[::-1],atol=1e-8)

    def test_finite_shell_and_rc_force_torque_equilibrium(self):
        for code in ("F","RC150_a30"):
            for d in (0,1):
                c=calculate_case(code,"C99",d,mesh=2)
                self.assertLess(max(r["force_error_kN"] for r in c["equilibrium"]),1e-7)
                self.assertLess(max(abs(r["torque_error_kNm"]) for r in c["equilibrium"]),1e-6)
                self.assertLess(c["base_cut_reaction_error_kN"],1e-7)
                for floor in (1,2,3):
                    rows=[r for r in c["line_results"] if r["floor"]==floor]
                    self.assertAlmostEqual(sum(r["signed_share_pct"] for r in rows),100.,places=7)


if __name__ == "__main__": unittest.main()
