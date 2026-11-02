import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import lexicon_helper as helper


def word(label='word', variant='nl', component='b', duration=100):
    audio = SimpleNamespace(filename=f'/cgn/comp-{component}/{variant}/x.wav')
    return SimpleNamespace(label=label, audio=audio, duration=duration,
        overlap=False, syllables=[object()])


class FilterTests(unittest.TestCase):
    def test_duration_outliers_are_removed_in_original_order(self):
        durations = [1000, 200, 180, 220, 190, 210, 10]
        tokens = [word(duration=d) for d in durations]
        self.assertEqual(helper.filter_outlier_tokens_from_type(iter(tokens)),
            tokens[1:-1])

    def test_duration_outliers_support_sequences_and_inclusive_fences(self):
        tokens = [SimpleNamespace(start=100, end=100 + d)
            for d in (50, 200, 250, 300, 450)]
        self.assertEqual(helper.filter_outlier_tokens_from_type(tokens), tokens)

    def test_small_groups_and_zero_iqr(self):
        self.assertEqual(helper.filter_outlier_tokens_from_type([]), [])
        tokens = [word(duration=d) for d in (200, 200, 200, 200, 1000)]
        self.assertEqual(helper.filter_outlier_tokens_from_type(tokens),
            tokens[:4])
        self.assertEqual(helper.filter_outlier_tokens_from_type(tokens,
            min_tokens=6), tokens)

    def test_invalid_outlier_parameters_and_durations(self):
        for value in (-1, float('nan'), float('inf'), True, '1.5'):
            with self.subTest(multiplier=value), self.assertRaises(ValueError):
                helper.filter_outlier_tokens_from_type([], iqr_multiplier=value)
        for value in (0, 3, 5.5, True):
            with self.subTest(min_tokens=value), self.assertRaises(ValueError):
                helper.filter_outlier_tokens_from_type([], min_tokens=value)
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(duration=value), self.assertRaises(ValueError):
                helper.filter_outlier_tokens_from_type([word(duration=value)])

    def test_frequency_counts_only_eligible_dutch_tokens(self):
        words = [word('keep'), word('keep'), word('drop'),
            word('drop', variant='vl'), word('drop', component='a'),
            word('drop', duration=99)]
        eligible = helper.filter_components(words)
        eligible = helper.filter_dutch_variant(eligible)
        eligible = helper.filter_duration(eligible)
        counts = helper.make_frequency_dict(eligible)
        selected = helper.filter_frequency(counts, eligible, min_freq=2)
        self.assertEqual(selected, words[:2])

    def test_duration_boundaries_are_inclusive(self):
        words = [word(duration=d) for d in (99, 100, 1500, 1501)]
        self.assertEqual(helper.filter_duration(words), words[1:3])

    def test_unknown_variant_preserves_caller_policy(self):
        words = [word(variant='unknown')]
        self.assertEqual(helper.filter_dutch_variant(words), [])
        with self.assertRaises(ValueError):
            helper.filter_dutch_variant(words, strict=True)

    def test_optional_frequency_bounds_and_short_labels(self):
        words = [word('a'), word('ab'), word('ab'), word('cd')]
        counts = helper.make_frequency_dict(words)
        self.assertEqual(counts, {'ab': 2, 'cd': 1})
        self.assertEqual(helper.filter_frequency(counts, words, None, 1),
            words[3:])
        self.assertEqual(helper.filter_frequency(counts, words, None, None),
            words[1:])

    def test_optional_word_filters(self):
        words = [word(), word()]
        words[1].overlap = True
        words[1].syllables.append(object())
        self.assertEqual(helper.remove_overlap(words), words[:1])
        self.assertEqual(helper.filter_on_n_syllables(words), words[:1])
        self.assertIs(helper.filter_on_n_syllables(words, None), words)


if __name__ == '__main__':
    unittest.main()
