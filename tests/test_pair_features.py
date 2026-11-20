from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import pair_features as pf


def table(**columns):
    defaults = dict(speaker=['a', 'a', 'b', 'c'],
        recording=['r1', 'r1', 'r2', 'r3'], component=['g', 'g', 'k', 'g'],
        gender=['female', 'female', 'male', 'unknown'], age=[30, 0, 50, 40],
        duration_ms=[200, 400, 100, 200],
        phones=['b oː m', 'p oː m', 'b oː m ə n', 'ɲ a'],
        left_word=['', '', '', ''], right_word=['', '', '', ''],
        boundary_index=[-1, -1, -1, -1])
    defaults.update(columns)
    return {name: np.asarray(values) for name, values in defaults.items()}


class PairFeatureTests(unittest.TestCase):
    def setUp(self):
        self.features = table()
        self.queries = np.array([0, 1])
        self.others = np.array([[1, 2], [3, 0]])

    def test_identity_features_follow_other_shape(self):
        f, q, o = self.features, self.queries, self.others
        np.testing.assert_array_equal(pf.same_speaker(f, q, o),
            [[True, False], [False, True]])
        np.testing.assert_array_equal(pf.same_recording(f, q, o),
            [[True, False], [False, True]])
        np.testing.assert_array_equal(pf.same_component(f, q, o),
            [[True, False], [True, True]])
        np.testing.assert_array_equal(pf.same_speaker(f, q, o[:, 0]),
            [True, False])

    def test_unknown_gender_and_age_give_nan(self):
        f, q, o = self.features, self.queries, self.others
        np.testing.assert_array_equal(pf.same_gender(f, q, o),
            [[1., 0.], [np.nan, 1.]])
        np.testing.assert_array_equal(pf.age_difference(f, q, o),
            [[np.nan, 20.], [np.nan, np.nan]])
        np.testing.assert_array_equal(pf.duration_ratio(f, q, o),
            [[2., 2.], [2., 2.]])

    def test_phone_feature_and_sonority_distances(self):
        f, q, o = self.features, self.queries, self.others
        np.testing.assert_array_equal(pf.phone_edit_distance(f, q, o),
            [[1., 2.], [3., 1.]])
        features = pf.feature_edit_distance(f, q, o)
        self.assertTrue(0 < features[0, 0] < 1)  # b/p differ in voicing
        self.assertEqual(features[0, 1], 2.)
        sonority = pf.sonority_distance(f, q, o)
        self.assertEqual(sonority[0, 0], 0.)  # b and p are both stops
        self.assertEqual(sonority[0, 1], 2.)
        self.assertTrue(np.isnan(sonority[1, 0]))  # ɲ has no class

    def test_shared_words_require_cross_boundary_table(self):
        with self.assertRaisesRegex(ValueError, 'cross-boundary'):
            pf.shared_left_word(self.features, self.queries, self.others)
        features = table(left_word=['de', 'de', 'een', 'de'],
            right_word=['boom', 'huis', 'boom', 'boom'],
            boundary_index=[1, 1, 2, 1])
        np.testing.assert_array_equal(
            pf.shared_left_word(features, self.queries, self.others),
            [[True, False], [True, True]])
        np.testing.assert_array_equal(
            pf.shared_right_word(features, self.queries, self.others),
            [[False, True], [False, False]])

    def test_random_pairs_exclude_self_and_are_seeded(self):
        queries, others = pf.random_pairs(3, 1000, seed=1)
        self.assertFalse(np.any(queries == others))
        self.assertEqual(set(others.tolist()), {0, 1, 2})
        again = pf.random_pairs(3, 1000, seed=1)
        np.testing.assert_array_equal(queries, again[0])
        labels = np.array(['a', 'a', 'b'])
        queries, others = pf.random_pairs(3, 1000, seed=2, labels=labels)
        self.assertFalse(np.any(labels[queries] == labels[others]))
        fixed = np.array([0, 2, 2, 1] * 100)
        others = pf.random_others(fixed, 3, seed=3, labels=labels)
        self.assertFalse(np.any(labels[fixed] == labels[others]))
        self.assertTrue(np.all(others[fixed == 2] != 2))
        with self.assertRaisesRegex(ValueError, 'shape|must be'):
            pf.same_speaker(self.features, self.queries, np.array([1]))

    def test_context_words_and_absolute_difference(self):
        features = table(left_words=['de groene', '* *', 'een groene', 'de'],
            right_words=['staat *', 'valt', 'staat hier', '* *'],
            articulation_rate=[10., 12., 13., np.nan])
        q, o = np.array([0, 0, 1]), np.array([2, 1, 3])
        np.testing.assert_array_equal(pf.shared_previous_word(features, q, o),
            [1, np.nan, np.nan])
        np.testing.assert_array_equal(pf.shared_next_word(features, q, o),
            [1, 0, np.nan])
        np.testing.assert_array_equal(pf.shared_context_word(features, q, o),
            [True, False, False])
        np.testing.assert_array_equal(pf.absolute_difference(features,
            'articulation_rate', q, o), [3, 2, np.nan])


if __name__ == '__main__':
    unittest.main()
