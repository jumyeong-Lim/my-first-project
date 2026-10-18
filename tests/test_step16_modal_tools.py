"""모드 비교의 정규화·대응·근접 부분공간 처리를 독립 예제로 검증한다."""
import unittest

import numpy as np

from verification.step16_modal_tools import (
    match_modes,
    mode_clusters,
    subspace_similarity,
    weighted_mac,
)


class ModalToolsTests(unittest.TestCase):
    def test_mac_scale_and_sign_invariance(self):
        a = np.array([[1., 2., -1.], [2., 0., 1.]])
        b = np.array([[2., 1., 3.], [0., -1., 2.]])
        weights = np.array([1., 3., 7.])
        expected = weighted_mac(a, b, weights)
        actual = weighted_mac(a * np.array([[-17.], [0.004]]),
                              b * np.array([[0.2], [-31.]]), weights)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_different_original_mass_normalizations_use_fixed_weights(self):
        a = np.array([[1., 2., -1.], [2., -1., 3.]])
        b = np.array([[1., 1., -2.], [3., 0., 1.]])
        first_mass = np.array([1., 4., 9.])
        second_mass = np.array([9., 4., 1.])
        fixed = np.array([2., 5., 3.])
        an = a / np.sqrt(np.sum(a * a * first_mass, axis=1))[:, None]
        bn = b / np.sqrt(np.sum(b * b * second_mass, axis=1))[:, None]
        expected = (a @ np.diag(fixed) @ b.T) ** 2 / np.outer(
            np.sum(a * a * fixed, axis=1), np.sum(b * b * fixed, axis=1)
        )
        np.testing.assert_allclose(weighted_mac(an, bn, fixed), expected, atol=1e-14)

    def test_matching_reordered_and_scaled_candidate_modes(self):
        a = np.eye(3)
        b = a[[2, 0, 1]] * np.array([[-4.], [2.], [7.]])
        result = match_modes(a, b, np.array([1., 2., 3.]))
        np.testing.assert_array_equal(result["indices"], [1, 2, 0])
        np.testing.assert_allclose(result["MAC"], 1.)
        # 후보가 더 많아도 기준 모드 전체에 대한 일대일 대응을 유지한다.
        larger = np.vstack([b, [1., 1., 1.]])
        np.testing.assert_array_equal(match_modes(a, larger, np.ones(3))["indices"],
                                      [1, 2, 0])

    def test_rotated_near_degenerate_subspace_is_identical(self):
        a = np.array([[1., 0., 0.], [0., 1., 0.]])
        angle = np.pi / 4
        rotation = np.array([[np.cos(angle), np.sin(angle)],
                             [-np.sin(angle), np.cos(angle)]])
        b = rotation @ a
        np.testing.assert_allclose(weighted_mac(a, b, np.ones(3)), 0.5, atol=1e-14)
        result = subspace_similarity(a, b, np.array([2., 7., 3.]))
        self.assertAlmostEqual(result["minimum_cosine_squared"], 1., places=14)
        np.testing.assert_allclose(result["principal_angles_rad"], 0., atol=3e-8)

    def test_orthogonal_and_partial_subspaces(self):
        one = subspace_similarity([[1., 0.]], [[0., 1.]], [2., 9.])
        self.assertAlmostEqual(one["minimum_cosine_squared"], 0.)
        self.assertAlmostEqual(one["principal_angles_rad"][0], np.pi / 2)
        partial = subspace_similarity([[1., 0., 0.], [0., 1., 0.]],
                                      [[1., 0., 0.], [0., 0., 1.]], [1., 2., 3.])
        np.testing.assert_allclose(partial["cosines_squared"], [1., 0.])

    def test_period_clusters_use_adjacent_relative_gap(self):
        self.assertEqual(mode_clusters([1., .98, .96, .6, .59, .2]),
                         [[0, 1, 2], [3, 4], [5]])
        self.assertEqual(mode_clusters([1., 1., .5], relative_gap=0), [[0, 1], [2]])
        self.assertEqual(mode_clusters([]), [])

    def test_mac_input_errors(self):
        cases = [
            ([1., 0.], [[1., 0.]], [1., 1.]),
            (np.empty((0, 2)), [[1., 0.]], [1., 1.]),
            ([[1., 0.]], [[1., 0., 0.]], [1., 1.]),
            ([[1., 0.]], [[1., 0.]], [1.]),
            ([[1., 0.]], [[1., 0.]], [[1., 1.]]),
            ([[1., 0.]], [[1., 0.]], [0., 1.]),
            ([[1., 0.]], [[1., 0.]], [-1., 1.]),
            ([[1., 0.]], [[1., 0.]], [1., np.inf]),
            ([[np.nan, 0.]], [[1., 0.]], [1., 1.]),
            ([[0., 0.]], [[1., 0.]], [1., 1.]),
            ([[1j, 0.]], [[1., 0.]], [1., 1.]),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                weighted_mac(*args)

    def test_matching_and_subspace_input_errors(self):
        with self.assertRaises(ValueError):
            match_modes(np.eye(3), np.eye(3)[:2], np.ones(3))
        for a, b in [
            ([[1., 0.]], [[1., 0.], [0., 1.]]),
            ([[1., 0.], [2., 0.]], [[1., 0.], [0., 1.]]),
            (np.ones((3, 2)), np.ones((3, 2))),
        ]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                subspace_similarity(a, b, np.ones(2))
        for periods, gap in [([1., 2.], .03), ([1., 0.], .03),
                             ([1., np.nan], .03), ([[1., .9]], .03),
                             ([1.], -.1), ([1.], np.inf), ([1.], [.03])]:
            with self.subTest(periods=periods, gap=gap), self.assertRaises(ValueError):
                mode_clusters(periods, gap)


if __name__ == "__main__":
    unittest.main()
