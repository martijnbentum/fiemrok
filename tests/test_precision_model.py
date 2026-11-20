from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import precision_model as pm


def part(condition, n_types, n_tokens, hit_rate, generator):
    n = n_types * n_tokens
    labels = np.repeat([f'{condition}{i}' for i in range(n_types)], n_tokens)
    features = dict(label=labels,
        speaker=np.array([f's{i}' for i in generator.integers(20, size=n)]),
        recording=np.array([f'r{i}' for i in generator.integers(30, size=n)]),
        component=np.array(['a', 'k', 'o'])[generator.integers(3, size=n)],
        gender=np.array(['female', 'male'])[generator.integers(2, size=n)],
        age=generator.choice([0, 25, 40, 60], size=n),
        n_phones=generator.integers(2, 9, size=n),
        articulation_rate=generator.uniform(10, 16, size=n))
    types = np.unique(labels, return_inverse=True)[1]
    neighbours = np.empty((n, 30), dtype=np.int64)
    for i in range(n):
        same = np.flatnonzero(types == types[i])
        same = same[same != i]
        other = np.flatnonzero(types != types[i])
        hits = generator.random(30) < hit_rate
        neighbours[i] = np.where(hits, generator.choice(same, 30),
            generator.choice(other, 30))
    return dict(condition=condition, features=features,
        neighbours=neighbours, types=types)


class PrecisionModelTests(unittest.TestCase):
    def test_precision_skips_same_speaker_and_dropped_tokens(self):
        features = dict(label=np.array(['a', 'a', 'b', 'a']),
            speaker=np.array(['x', 'x', 'y', 'y']))
        neighbours = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3],
            [0, 1, 2]])
        values, n = pm.precision(features, neighbours, top_k=2)
        np.testing.assert_array_equal(n, [2, 2, 2, 2])
        np.testing.assert_allclose(values, [0.5, 0.5, 0, 1])
        values, n = pm.precision(features, neighbours, cross_speaker=False,
            top_k=2)
        np.testing.assert_allclose(values, [0.5, 1, 0, 1])
        keep = np.array([True, False, True, True])
        values, n = pm.precision(features, neighbours, cross_speaker=False,
            keep=keep, top_k=2)
        np.testing.assert_allclose(values, [0.5, 1, 0, 0.5])

    def test_fit_recovers_the_precision_difference(self):
        generator = np.random.default_rng(0)
        parts = [part('word', 60, 20, 0.7, generator),
            part('cross_boundary', 50, 20, 0.4, generator)]
        fit = pm.fit_precision(parts, cross_speaker=False, n_bootstrap=10)
        effect = fit['coefficients']['is_word']
        self.assertAlmostEqual(effect['estimate'], 0.3, delta=0.05)
        self.assertLess(effect['ci_low'], effect['ci_high'])
        control = pm.index_size_control(parts, n_subsamples=3)
        self.assertEqual(control['n_word_types'], 50)
        self.assertAlmostEqual(control['mean'], 0.3, delta=0.08)


if __name__ == '__main__':
    unittest.main()
