import pathlib
import sys
import unittest
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cross_boundary_lexicon as cbl


def make_phrase(words, component='b', start=1000):
    '''Make consecutive phones with distinct word keys and audio metadata.'''
    audio = SimpleNamespace(filename=f'/cgn/comp-{component}/nl/test.wav')
    phones = []
    for word_index, labels in enumerate(words):
        word = SimpleNamespace(key=word_index)
        for label in labels:
            phone_start = start + len(phones) * 100
            phones.append(SimpleNamespace(label=label, start=phone_start,
                end=phone_start + 100, word=word))
    return SimpleNamespace(phones=phones, audio=audio)


def find_sequence(ps, start, end):
    '''Find an occurrence by its half-open phone-index range.'''
    return next(sequence for sequence in ps.sequences
        if (sequence.start_index, sequence.end_index) == (start, end))


class ExtractionTests(unittest.TestCase):
    def test_extracts_all_and_only_single_boundary_spans(self):
        phrase = make_phrase([['a', 'b'], ['c', 'd'], ['e', 'f']])
        ps = cbl.PhraseSequences(phrase)
        expected = [(0, 3), (0, 4), (1, 3), (1, 4),
            (2, 5), (2, 6), (3, 5), (3, 6)]
        actual = [(s.start_index, s.end_index) for s in ps.sequences]
        self.assertEqual(actual, expected)
        for sequence in ps.sequences:
            with self.subTest(span=(sequence.start_index, sequence.end_index)):
                phones = phrase.phones[sequence.start_index:sequence.end_index]
                self.assertEqual(sequence.phones, tuple(phones))
                changes = sum(a.word.key != b.word.key
                    for a, b in zip(phones, phones[1:]))
                self.assertEqual(changes, 1)
                self.assertIs(sequence.parent, ps)

    def test_orders_phones_by_time_without_reordering_phrase(self):
        phrase = make_phrase([['a', 'b'], ['c']])
        phrase.phones.reverse()
        original = list(phrase.phones)
        ps = cbl.PhraseSequences(phrase)
        self.assertEqual([s.phoneme_sequence for s in ps.sequences],
            ['a b c', 'b c'])
        self.assertEqual(phrase.phones, original)
        self.assertEqual([(s.start_index, s.end_index) for s in ps.sequences],
            [(0, 3), (1, 3)])

    def test_empty_and_single_word_phrases_have_no_candidates(self):
        for words in ([], [['a']], [['a', 'b', 'c']]):
            with self.subTest(words=words):
                self.assertEqual(cbl.PhraseSequences(
                    make_phrase(words)).sequences, [])

    def test_equal_word_labels_still_have_distinct_word_boundaries(self):
        phrase = make_phrase([['a'], ['a'], ['a']])
        ps = cbl.PhraseSequences(phrase)
        self.assertEqual([(s.start_index, s.end_index) for s in ps.sequences],
            [(0, 2), (1, 3)])

    def test_rebuilding_sets_sequences_and_returns_none(self):
        ps = cbl.PhraseSequences(make_phrase([['a'], ['n']]))
        result = ps.make_cross_word_boundary_phone_sequences()
        self.assertIsNone(result)
        self.assertEqual(len(ps.sequences), 1)
        self.assertEqual(ps.sequences[0].phoneme_sequence, 'a n')


class SequenceTests(unittest.TestCase):
    def test_exposes_parent_phrase_audio_indices_times_and_labels(self):
        phrase = make_phrase([['tʃ', 'aː'], ['n']])
        ps = cbl.PhraseSequences(phrase)
        sequence = find_sequence(ps, 0, 3)
        self.assertIs(sequence.parent, ps)
        self.assertIs(sequence.phrase, phrase)
        self.assertIs(sequence.audio, phrase.audio)
        self.assertEqual(sequence.phoneme_sequence, 'tʃ aː n')
        self.assertEqual(sequence.n_phones, 3)
        self.assertEqual((sequence.start, sequence.end), (1000, 1300))
        self.assertEqual((sequence.start_index, sequence.end_index), (0, 3))
        self.assertEqual(repr(sequence), "cbs('tʃ aː n', 300ms)")

    def test_equality_compares_phone_strings_across_occurrences(self):
        first = cbl.PhraseSequences(make_phrase([['a'], ['n']], start=0))
        second = cbl.PhraseSequences(make_phrase([['a'], ['n']], start=5000))
        different = cbl.PhraseSequences(make_phrase([['a'], ['m']]))
        self.assertEqual(first.sequences[0], second.sequences[0])
        self.assertNotEqual(first.sequences[0], different.sequences[0])
        self.assertIs(first.sequences[0].__eq__('a n'), NotImplemented)

    def test_empty_phone_sequence_is_rejected(self):
        ps = cbl.PhraseSequences(make_phrase([]))
        with self.assertRaisesRegex(ValueError, 'empty'):
            cbl.CrossBoundarySequence([], 0, ps)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ps = cbl.PhraseSequences(
            make_phrase([['a', 'b'], ['c', 'd'], ['e', 'f']]))

    def test_selection_blocks_overlap_but_allows_touching_intervals(self):
        first = find_sequence(self.ps, 0, 3)
        overlapping = find_sequence(self.ps, 1, 4)
        adjacent = find_sequence(self.ps, 3, 5)
        self.assertIs(self.ps.select(first), first)
        self.assertFalse(self.ps.can_select(first))
        self.assertFalse(self.ps.can_select(overlapping))
        with self.assertRaises(ValueError):
            self.ps.select(overlapping)
        self.assertTrue(self.ps.can_select(adjacent))
        self.ps.select(adjacent)
        self.assertEqual([(s.start_index, s.end_index)
            for s in self.ps.selected_sequences], [(0, 3), (3, 5)])

    def test_reselecting_same_occurrence_is_rejected(self):
        sequence = self.ps.sequences[0]
        self.ps.select(sequence)
        with self.assertRaises(ValueError):
            self.ps.select(sequence)
        self.assertEqual(len(self.ps.selected_sequences), 1)

    def test_owner_identity_matters_even_for_the_same_phrase(self):
        other = cbl.PhraseSequences(self.ps.phrase)
        sequence = other.sequences[0]
        self.assertEqual(self.ps, other)
        self.assertFalse(self.ps.can_select(sequence))
        with self.assertRaises(ValueError):
            self.ps.select(sequence)

    def test_non_overlapping_equal_strings_are_both_selectable(self):
        ps = cbl.PhraseSequences(make_phrase([['a'], ['n'], ['a'], ['n']]))
        first = find_sequence(ps, 0, 2)
        second = find_sequence(ps, 2, 4)
        self.assertEqual(first, second)
        ps.select(first)
        ps.select(second)
        self.assertEqual(len(ps.selected_sequences), 2)

    def test_available_sequences_and_reset_follow_selection(self):
        self.ps.select(find_sequence(self.ps, 0, 3))
        self.assertEqual([(s.start_index, s.end_index)
            for s in self.ps.available_sequences()], [(3, 5), (3, 6)])
        self.assertEqual(repr(self.ps), 'ps(sequences=8, selected=1)')
        self.ps.reset_selection()
        self.assertEqual(self.ps.selected_sequences, [])
        self.assertEqual(len(self.ps.available_sequences()), 8)
        self.assertTrue(all(self.ps.can_select(s) for s in self.ps.sequences))

    def test_phrase_container_equality_uses_phrase_object_identity(self):
        same = cbl.PhraseSequences(self.ps.phrase)
        equivalent = cbl.PhraseSequences(
            make_phrase([['a', 'b'], ['c', 'd'], ['e', 'f']]))
        self.assertIsNone(self.ps.parent)
        self.assertEqual(self.ps.phrase, equivalent.phrase)
        self.assertEqual(self.ps, same)
        self.assertNotEqual(self.ps, equivalent)
        self.assertIs(self.ps.__eq__(self.ps.phrase), NotImplemented)


class LexiconTests(unittest.TestCase):
    def test_aggregated_counts_track_selection_and_reset(self):
        phrases = [make_phrase([['a'], ['n'], ['a'], ['n']]),
            make_phrase([['a'], ['n']])]
        lexicon = cbl.CrossBoundaryLexicon(phrases)
        expected = Counter({'a n': 3, 'n a': 1})
        self.assertEqual(lexicon.sequence_counts, expected)
        self.assertEqual(lexicon.available_sequence_counts(), expected)
        self.assertEqual(lexicon.selected_sequence_counts, Counter())
        self.assertEqual(repr(lexicon), 'cbl(phrases=2, sequences=4)')
        for phrase, ps in zip(phrases, lexicon.phrase_sequences):
            self.assertIs(ps.parent, lexicon)
            self.assertIs(ps.phrase, phrase)
            self.assertIs(ps.sequences[0].parent.parent, lexicon)
            ps.select(ps.sequences[0])
        self.assertEqual(lexicon.selected_sequence_counts, Counter({'a n': 2}))
        self.assertEqual(lexicon.available_sequence_counts(),
            Counter({'a n': 1}))
        self.assertEqual(lexicon.sequence_counts, expected)
        lexicon.reset_selection()
        self.assertEqual(lexicon.selected_sequence_counts, Counter())
        self.assertEqual(lexicon.available_sequence_counts(), expected)

    def test_empty_lexicon(self):
        lexicon = cbl.CrossBoundaryLexicon([])
        self.assertEqual(lexicon.phrase_sequences, [])
        self.assertEqual(lexicon.sequence_counts, Counter())
        self.assertEqual(lexicon.selected_sequence_counts, Counter())
        self.assertEqual(lexicon.available_sequence_counts(), Counter())
        lexicon.reset_selection()
        self.assertEqual(repr(lexicon), 'cbl(phrases=0, sequences=0)')


class FilteringTests(unittest.TestCase):
    def test_component_is_read_from_audio_path(self):
        phrase = make_phrase([], component='h')
        self.assertEqual(cbl.phrase_to_component(phrase), 'h')

    def test_default_filters_match_historical_component_exclusions(self):
        components = 'abcdefghijklmno'
        phrases = [make_phrase([], component=c) for c in components]
        original = list(phrases)
        for function in (cbl.filter_phrases_by_component,
            cbl.filter_phrases_for_cross_boundary_lexicon):
            with self.subTest(function=function.__name__):
                filtered = function(phrases)
                self.assertIsInstance(filtered, list)
                self.assertEqual([cbl.phrase_to_component(p) for p in filtered],
                    list('befgijklno'))
                for phrase in filtered:
                    self.assertTrue(any(phrase is p for p in phrases))
        self.assertEqual(phrases, original)

    def test_custom_and_empty_component_exclusions(self):
        phrases = [make_phrase([], component=c) for c in 'abc']
        filtered = cbl.filter_phrases_for_cross_boundary_lexicon(
            phrases, remove_components=['b'])
        self.assertEqual([cbl.phrase_to_component(p) for p in filtered],
            ['a', 'c'])
        self.assertEqual(cbl.filter_phrases_by_component(
            phrases, remove_components=''), phrases)
        self.assertEqual(cbl.filter_phrases_for_cross_boundary_lexicon([]), [])


if __name__ == '__main__':
    unittest.main()
