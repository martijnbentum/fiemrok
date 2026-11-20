from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import factor_models as fm
from lexicon_validation import word_effect as we


WORDS = {'boom': 'b oː m', 'boon': 'b oː n', 'kat': 'k ɑ t',
    'kast': 'k ɑ s t', 'bom': 'b oː m'}
STRINGS = ('ə b', 'ə k', 'n d ə', 't ɑ', 's x ɪ')


def table(labels, phones, generator, cross=False):
    n = len(labels)
    rows = dict(label=np.array(labels), phones=np.array(phones),
        n_phones=np.array([len(p.split()) for p in phones]),
        speaker=np.array([f's{i}' for i in generator.integers(4, size=n)]),
        recording=np.array([f'r{i}' for i in generator.integers(6, size=n)]),
        component=np.array(['a', 'k'])[generator.integers(2, size=n)],
        gender=np.array(['female', 'male'])[generator.integers(2, size=n)],
        left_words=np.array(['de groene'] * n),
        right_words=np.array(['is'] * n),
        left_word=np.array([''] * n), right_word=np.array([''] * n),
        boundary_index=np.full(n, -1))
    if cross:
        rows.update(boundary_index=np.ones(n, dtype=int),
            left_word=np.array(['de', 'een'])[generator.integers(2, size=n)],
            right_word=np.array(['boom', 'kat'])[generator.integers(2,
                size=n)])
    return rows


def part(condition, features, effect, generator):
    pairs = fm.sample_pairs(features, n_pairs=400, seed=1)
    keep = pairs['stratum'] != fm.STRATA.index('close_phones')
    q, o = pairs['queries'][keep], pairs['others'][keep]
    same_type = features['label'][q] == features['label'][o]
    same_speaker = features['speaker'][q] == features['speaker'][o]
    similarity = (0.3 + effect * same_type + 0.05 * same_speaker
        + generator.normal(0, 0.005, size=len(q)))
    types = np.unique(features['label'], return_inverse=True)[1]
    return dict(condition=condition, features=features, queries=q,
        others=o, stratum=pairs['stratum'][keep], similarity=similarity,
        query_types=types[q])


class WordEffectTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(0)
        labels = np.repeat(list(WORDS), 10)
        words = table(labels, [WORDS[label] for label in labels], generator)
        strings = np.repeat(STRINGS, 10)
        cross = table(strings, strings, generator, cross=True)
        self.parts = [part('word', words, 0.25, generator),
            part('cross_boundary', cross, 0.1, generator)]

    def test_design_codes_dataset_specific_terms(self):
        design, centre = we.joint_design(self.parts)
        self.assertEqual(tuple(design), we.TERMS)
        is_word = design['is_word'] == 1
        self.assertFalse(design['cb_same_left_word'][is_word].any())
        self.assertFalse(design['same_form_different_label'][~is_word].any())
        self.assertTrue(design['same_form_different_label'][is_word].any())
        same_type = design['same_type'] == 1
        self.assertFalse(design['other_type_distance'][same_type].any())
        self.assertGreater(centre, 0)
        np.testing.assert_array_equal(design['same_type_x_is_word'],
            same_type & is_word)

    def test_fit_recovers_the_word_effect(self):
        design, _ = we.joint_design(self.parts)
        similarity = np.concatenate([p['similarity'] for p in self.parts])
        summary = we.fit_word_effect(design, similarity,
            we._type_groups(self.parts), n_bootstrap=20)
        coefficients = summary['coefficients']
        for name, expected in (('same_type', 0.1),
            ('same_type_x_is_word', 0.15), ('same_type_words', 0.25),
            ('same_speaker', 0.05)):
            self.assertAlmostEqual(coefficients[name]['estimate'], expected,
                delta=0.01, msg=name)
        effect = coefficients['same_type_x_is_word']
        self.assertLess(effect['ci_low'], effect['ci_high'])
        text = we.format_summary(dict(summary, model='hubert', layer=9,
            pooling='mean', distance_centre=0.9))
        self.assertIn('same_type_x_is_word', text)


if __name__ == '__main__':
    unittest.main()
