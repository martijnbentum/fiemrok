import contextlib
import io
import pathlib
import random
import sys
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace

import phraser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lexicon_validation import cross_boundary_sampling as sampling
from lexicon_validation import word_lexicon as wl


class WordSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.store = phraser.Store(path=self.root / 'store')
        self.addCleanup(self.store.close)
        self.audio = self.store.create(phraser.Audio,
            filename='/cgn/comp-b/nl/a.wav', duration=100000, save=True)
        self.speaker = self.store.create(phraser.Speaker, name='speaker',
            dataset='test', save=True)
        self.start = 0
        self.phrases = []

    def add_word(self, label, duration=200, phones=('a', 'b'),
        flag='within_range', timing_uncertain=False):
        '''Queue a phrase holding one word with the given properties.'''
        identity = dict(audio_id=self.audio.identifier,
            speaker_id=self.speaker.identifier, store=self.store)
        start, end = self.start, self.start + duration
        self.start = end + 100
        phrase = phraser.Phrase(label, start, end, **identity)
        word = phraser.Word(label, start, end, **identity)
        syllable = phraser.Syllable(label, start, end, **identity)
        step = duration // len(phones)
        for index, phone in enumerate(phones):
            phone_start = start + index * step
            syllable.add_child(phraser.Phone(phone, phone_start,
                phone_start + step, **identity))
        word.duration_outlier = flag
        word.timing_uncertain = timing_uncertain
        word.add_child(syllable)
        phrase.add_child(word)
        self.phrases.append(phrase)

    def select(self, sample=True, **kwargs):
        if self.phrases: self.store.save_phrase_trees(self.phrases)
        self.phrases = []
        self.store.refresh_query_roots()
        output = io.StringIO()
        random_seed = kwargs.pop('random_seed', 0)
        with contextlib.redirect_stdout(output):
            lexicon = wl.select_words(self.store, **kwargs)
            if sample:
                lexicon = lexicon.sample(n_tokens=kwargs.get('n_tokens', 100),
                    random_seed=random_seed)
        self.assertIsInstance(lexicon, phraser.Lexicon)
        self.assertEqual(lexicon.n_tokens, len(lexicon.token_keys))
        return lexicon.token_keys, output.getvalue()

    def labels(self, keys):
        return Counter(word.label for word in self.store.load_many(keys))

    def test_each_filter_removes_tokens_before_the_type_quota(self):
        for _ in range(3): self.add_word('keep')
        for _ in range(3): self.add_word('single', phones=('a',))
        self.add_word('short', duration=99)
        for _ in range(2): self.add_word('short')
        self.add_word('outlier', flag='long')
        for _ in range(2): self.add_word('outlier')
        self.add_word('unassessed', flag='unset')
        for _ in range(2): self.add_word('unassessed')
        self.add_word('uncertain', timing_uncertain=True)
        for _ in range(2): self.add_word('uncertain')
        keys, report = self.select(n_tokens=3)
        self.assertEqual(self.labels(keys), {'keep': 3})
        expected = {'At least 2 phones': ['15', '3', '5', '1', '1'],
            'Duration 100–1500 ms': ['14', '1', '5', '0', '1'],
            'Duration flag assessed': ['13', '1', '5', '0', '1'],
            'Duration flag within_range': ['12', '1', '5', '0', '1'],
            'Recorded timing uncertain': ['11', '1', '5', '0', '1'],
            'At least 3 tokens per type': ['3', '8', '1', '4', '0']}
        for condition, counts in expected.items():
            row = next(line for line in report.splitlines()
                if line.startswith(condition))
            self.assertEqual(row[len(condition):].split(), counts)
        self.assertIn('Below quota: types newly below 3 tokens', report)
        self.assertIn('Eligible inventory: 1 types; 3 tokens', report)

    def test_inventory_retains_all_eligible_tokens_before_sampling(self):
        for _ in range(5): self.add_word('word')
        keys, report = self.select(sample=False, n_tokens=3)
        self.assertEqual(len(keys), 5)
        self.assertIn('Eligible inventory: 1 types; 5 tokens', report)

    def test_verbose_false_suppresses_progress_and_report(self):
        self.add_word('word')
        keys, output = self.select(n_tokens=1, verbose=False)
        self.assertEqual(len(keys), 1)
        self.assertEqual(output, '')

    def test_exactly_n_tokens_per_type_in_label_then_key_order(self):
        for label in ('b', 'a'):
            for _ in range(5): self.add_word(label)
        keys, _ = self.select(n_tokens=3)
        self.assertEqual(self.labels(keys), {'a': 3, 'b': 3})
        words = self.store.load_many(keys)
        order = [(word.label, word.key) for word in words]
        self.assertEqual(order, sorted(order))

    def test_duration_bounds_are_inclusive(self):
        self.add_word('word', duration=100)
        self.add_word('word', duration=1500)
        keys, _ = self.select(n_tokens=2)
        self.assertEqual(len(keys), 2)

    def test_selection_is_seeded_and_leaves_global_rng_alone(self):
        for _ in range(10): self.add_word('word')
        state = random.getstate()
        first, _ = self.select(n_tokens=3, random_seed=23)
        second, _ = self.select(n_tokens=3, random_seed=23)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), state)

    def test_unassessed_store_is_rejected(self):
        for _ in range(3): self.add_word('word', flag='unset')
        with self.assertRaisesRegex(ValueError, 'not assessed'):
            self.select(n_tokens=1)

    def test_invalid_quota(self):
        for n in (0, -1, True, 1.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                wl.select_words(self.store, n_tokens=n)

    def test_key_round_trip_preserves_order(self):
        keys = [b'b', b'a', b'c']
        path = self.root / 'word_keys.pkl'
        wl.save_selected_word_keys(iter(keys), path)
        self.assertEqual(wl.load_selected_word_keys(path), keys)



def pronounced_word(key, label='word', pronunciation=('a', 'b')):
    phones = [SimpleNamespace(label=label, start=i * 10)
        for i, label in enumerate(pronunciation)]
    return SimpleNamespace(key=key.to_bytes(4, 'big'), label=label,
        phones=phones)


class PhoneTargetTests(unittest.TestCase):
    def test_phone_count_targets_come_from_selected_words(self):
        words = [pronounced_word(1, pronunciation=('b', 'a')),
            pronounced_word(2, pronunciation=('a', 'b', 'c')),
            pronounced_word(3, label='second', pronunciation=('d', 'e', 'f'))]
        lengths, phones = sampling.inspect_lexicon_phonemes(words)
        # Tied word pronunciations use lexical order, independent of input.
        self.assertEqual(lengths, {3: 2})
        self.assertEqual(phones, Counter('abcdef'))
        self.assertEqual(sampling.inspect_lexicon_phonemes(reversed(words)),
            (lengths, phones))


if __name__ == '__main__':
    unittest.main()
