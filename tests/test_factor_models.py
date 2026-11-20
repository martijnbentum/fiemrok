from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import factor_models as fm


WORDS = {'boom': 'b oː m', 'boon': 'b oː n', 'been': 'b eː n',
    'kat': 'k ɑ t', 'kast': 'k ɑ s t', 'mat': 'm ɑ t'}


def table(n_per_word=8, seed=0):
    generator = np.random.default_rng(seed)
    labels = np.repeat(list(WORDS), n_per_word)
    n = len(labels)
    phones = np.array([WORDS[label] for label in labels])
    rows = dict(label=labels, phones=phones,
        n_phones=np.array([len(p.split()) for p in phones]),
        speaker=np.array([f's{i}' for i in generator.integers(4, size=n)]),
        recording=np.array([f'r{i}' for i in generator.integers(6, size=n)]),
        component=np.array(['a', 'k'])[generator.integers(2, size=n)],
        gender=np.array(['female', 'male'])[generator.integers(2, size=n)],
        age=generator.integers(0, 60, size=n),
        duration_ms=generator.integers(150, 600, size=n),
        articulation_rate=generator.uniform(10, 16, size=n),
        left_words=np.array(['de'] * n), right_words=np.array(['is'] * n),
        left_word=np.array([''] * n), right_word=np.array([''] * n),
        boundary_index=np.full(n, -1),
        word_log_freq=generator.uniform(1, 3, size=n))
    return rows


class FactorModelTests(unittest.TestCase):
    def setUp(self):
        self.features = table()
        self.pairs = fm.sample_pairs(self.features, n_pairs=300, seed=1)

    def test_strata_follow_their_definitions(self):
        f, p = self.features, self.pairs
        q, o, stratum = p['queries'], p['others'], p['stratum']
        self.assertFalse(np.any(q == o))
        same_word = f['label'][q] == f['label'][o]
        same_speaker = f['speaker'][q] == f['speaker'][o]
        for index, name in enumerate(fm.STRATA):
            mask = stratum == index
            self.assertTrue(mask.any(), name)
            expected = name.startswith('same_word')
            self.assertTrue(np.all(same_word[mask] == expected), name)
            if name.endswith('same_speaker'):
                self.assertTrue(np.all(same_speaker[mask]))
        close = stratum == fm.STRATA.index('close_phones')
        design = fm.pair_design(f, q[close], o[close])
        distance = design['phone_edit_distance']
        self.assertTrue(np.all((distance >= 1) & (distance <= 2)))
        np.testing.assert_array_equal(f['n_phones'][q[close]],
            f['n_phones'][o[close]])

    def test_models_recover_a_word_effect(self):
        f, p = self.features, self.pairs
        q, o = p['queries'], p['others']
        design = fm.pair_design(f, q, o)
        self.assertIn('age_difference_unknown', design)
        generator = np.random.default_rng(2)
        similarity = (0.3 + 0.2 * design['same_word']
            + 0.1 * design['same_speaker']
            + generator.normal(0, 0.01, size=len(q)))
        types = np.unique(f['label'], return_inverse=True)[1]
        summary = fm.fit_pair_models(design, similarity, p['stratum'],
            types[q], n_bootstrap=20)
        coefficients = summary['coefficients']
        self.assertAlmostEqual(coefficients['same_word']['estimate'], 0.2,
            delta=0.01)
        self.assertAlmostEqual(coefficients['same_speaker']['estimate'], 0.1,
            delta=0.01)
        word = coefficients['same_word']
        self.assertLess(word['ci_low'], word['estimate'])
        self.assertGreater(summary['unique_r2']['same_word'], 0.1)
        for name in ('same_speaker', 'other_speaker', 'matched_cells'):
            self.assertAlmostEqual(summary['contrasts'][name]['estimate'],
                0.2, delta=0.01)
        text = fm.format_summary(dict(summary, condition='word',
            model='hubert', layer=9, pooling='mean',
            stratum_counts={name: 1 for name in fm.STRATA}))
        self.assertIn('matched_cells', text)

    def test_saved_pairs_are_reused_with_their_settings(self):
        with tempfile.TemporaryDirectory() as workspace:
            first = fm.load_pairs('word', self.features, 50, 0,
                workspace=workspace)
            again = fm.load_pairs('word', self.features, 60, 1,
                workspace=workspace)
            np.testing.assert_array_equal(first['queries'], again['queries'])
            self.assertEqual((again['n_pairs'], again['seed']), (50, 0))
            fm.save_pairs('word', fm.sample_pairs(self.features, 60, 1), 60, 1,
                workspace=workspace)
            resampled = fm.load_pairs('word', self.features,
                workspace=workspace)
            self.assertEqual(resampled['n_pairs'], 60)


if __name__ == '__main__':
    unittest.main()
