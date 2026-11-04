import pathlib
import random
import sys
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lexicon_validation import cross_boundary_lexicon_sampling as sampling
from lexicon_validation import word_lexicon as wl


def word(key, label='word', duration=200, variant='nl', component='b',
    pronunciation=('a', 'b')):
    phones = [SimpleNamespace(label=label, start=i * 10)
        for i, label in enumerate(pronunciation)]
    filename = f'/cgn/comp-{component}/{variant}/a.wav'
    return SimpleNamespace(key=key.to_bytes(4, 'big'), label=label,
        duration=duration, phones=phones, syllables=[object(), object()],
        audio=SimpleNamespace(filename=filename))


class WordSelectionTests(unittest.TestCase):
    def test_word_filters_allow_multisyllabic_words(self):
        keep = [word(1), word(2, label='I')]
        reject = [word(11, pronunciation=('a',)),
            word(12, duration=99), word(13, duration=1501)]
        actual = wl.filter_word_tokens(keep + reject, n_tokens=1)
        self.assertEqual({w.key for w in actual}, {w.key for w in keep})

    def test_corpus_filters_are_not_reapplied_to_local_material(self):
        tokens = [word(1, variant='vl')]
        tokens += [word(20 + i, component=c) for i, c in enumerate('acdhm')]
        self.assertEqual(wl.filter_word_tokens(tokens, n_tokens=1), tokens)
        # Selection does not depend on corpus-layout paths in the local store.
        tokens[0].audio.filename = 'local-recording.wav'
        self.assertEqual(wl.filter_word_tokens(tokens, n_tokens=1), tokens)

    def test_duration_bounds_are_inclusive(self):
        tokens = [word(1, duration=100), word(2, duration=1500)]
        self.assertEqual(wl.filter_word_tokens(tokens, n_tokens=2), tokens)

    def test_iqr_filter_is_per_type_and_precedes_frequency(self):
        good = [word(i, duration=200) for i in range(10)]
        high = word(10, duration=1000)
        other_type = [word(20 + i, label='other', duration=1000)
            for i in range(10)]
        actual = wl.filter_word_tokens(good + [high] + other_type, n_tokens=10)
        self.assertEqual(Counter(w.label for w in actual),
            {'word': 10, 'other': 10})
        self.assertNotIn(high.key, [w.key for w in actual])
        self.assertEqual(wl.filter_word_tokens(good + [high], n_tokens=11), [])

    def test_select_is_order_independent_and_does_not_change_global_rng(self):
        tokens = [word(i, label=str(i % 2)) for i in range(20)]
        state = random.getstate()
        first = wl.select_words(tokens, n_tokens=3, random_seed=23)
        second = wl.select_words(reversed(tokens), n_tokens=3, random_seed=23)
        self.assertEqual([w.key for w in first], [w.key for w in second])
        self.assertEqual(Counter(w.label for w in first), {'0': 3, '1': 3})
        self.assertEqual(random.getstate(), state)

    def test_sample_quota_is_not_silently_reduced(self):
        with self.assertRaisesRegex(ValueError, 'fewer than 2'):
            wl.select_words([word(1)], n_tokens=2)
        for n in (0, -1, True, 1.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                wl.filter_word_tokens([], n_tokens=n)

    def test_key_round_trip_preserves_order(self):
        keys = [b'b', b'a', b'c']
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'word_keys.pkl'
            wl.save_selected_word_keys(iter(keys), path)
            self.assertEqual(wl.load_selected_word_keys(path), keys)

    def test_phone_count_targets_come_from_selected_words(self):
        words = [word(1, pronunciation=('b', 'a')),
            word(2, pronunciation=('a', 'b', 'c')),
            word(3, label='second', pronunciation=('d', 'e', 'f'))]
        lengths, phones = sampling.inspect_lexicon_phonemes(words)
        # Tied word pronunciations use lexical order, independent of input.
        self.assertEqual(lengths, {3: 2})
        self.assertEqual(phones, Counter('abcdef'))
        self.assertEqual(sampling.inspect_lexicon_phonemes(reversed(words)),
            (lengths, phones))


if __name__ == '__main__':
    unittest.main()
