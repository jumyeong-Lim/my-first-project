"""STEP11 후처리식의 독립 단위시험."""
import unittest

import numpy as np

from verification.step11_robustness_conclusion import (
    ANCHORS, METRICS, contrasts, first_modes_match, response_map, scenarios,
)


class RobustnessPostprocessingTest(unittest.TestCase):
    def test_interaction_is_double_difference(self):
        records = {}
        for family, level, _, _, _, diaphragms in scenarios():
            for mass_case in ANCHORS:
                for diaphragm in diaphragms:
                    value = 10.0
                    if mass_case != "U":
                        value += 2.0
                    if diaphragm == "F":
                        value += 3.0
                        if mass_case != "U":
                            value += 4.0
                    elif diaphragm == "S":
                        value += 1.0
                    metrics = {metric: value for metric in METRICS}
                    records[(family, level, diaphragm, mass_case)] = {"metrics": metrics}
        output = contrasts(records)
        row = next(row for row in output if row["family"] == "baseline" and
                   row["diaphragm"] == "F" and row["mass_case"] == "C99" and
                   row["metric"] == "T1_s")
        self.assertAlmostEqual(row["interaction"], 4.0)
        self.assertAlmostEqual(row["mass_effect_R"], 2.0)
        self.assertAlmostEqual(row["diaphragm_effect_U"], 3.0)
        self.assertAlmostEqual(row["diaphragm_effect_E"], 7.0)

    def test_only_mesh_rigid_reference_can_fall_back(self):
        records = {("baseline", "reference", d, e): {"metrics": {"T1_s": 1.0}}
                   for d in ("R", "S", "F") for e in ANCHORS}
        for e in ANCHORS:
            for d in ("S", "F"):
                records[("mesh", "4x4", d, e)] = {"metrics": {"T1_s": 2.0}}
        self.assertEqual(response_map(records, "mesh", "4x4")[("R", "U")]["T1_s"], 1.0)
        with self.assertRaises(KeyError):
            response_map(records, "section", "0.75")

    def test_mode_match_handles_order_swap(self):
        # 세 층의 첫 두 정규직교 수평 변형을 서로 바꾸어 넣는다.
        from verification.step10_diaphragm_mass_interaction import P, XY
        width = 2*len(XY)*P.stories
        a = np.zeros(width)
        b = np.zeros(width)
        a[0] = 1
        b[1] = 1
        mass = np.full(len(XY), 1/len(XY))
        norm = np.sqrt(P.floor_mass/len(XY))
        reference = {"modal": {"vectors": [a/norm, b/norm]}}
        comparison = {"modal": {"vectors": [b/norm, a/norm]}}
        match = first_modes_match(reference, comparison, mass)
        self.assertEqual(match["matched_orders_first_9"], [2, 1])
        np.testing.assert_allclose(match["MAC_first_9"], [1, 1])


if __name__ == "__main__":
    unittest.main()
