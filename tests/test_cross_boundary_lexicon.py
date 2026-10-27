import pathlib
import random
import sys
import unittest
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cross_boundary_lexicon as cbl


def make_phrase(words, component='b', start=1000, variant='nl'):
    '''Make consecutive phones with distinct word keys and audio metadata.'''
    filename = f'/cgn/comp-{component}/{variant}/test.wav'
    audio = SimpleNamespace(filename=filename)
    phones = []
    phrase_words = []
    for word_index, labels in enumerate(words):
        word = SimpleNamespace(key=word_index)
        phrase_words.append(word)
        for label in labels:
            phone_start = start + len(phones) * 100
            phone = SimpleNamespace(label=label, start=phone_start,
                end=phone_start + 100, word=word)
            phones.append(phone)
    return SimpleNamespace(phones=phones, words=phrase_words, audio=audio)


def find_sequence(ps, start, end):
    '''Find an occurrence by its half-open phone-index range.'''
    return next(sequence for sequence in ps.sequences
        if (sequence.start_index, sequence.end_index) == (start, end))


def make_sampling_lexicon(type_counts, tokens_per_type=2):
    '''Build independent occurrences, keeping one full candidate per phrase.'''
    phrases = []
    lengths = []
    for n_phones, n_types in sorted(type_counts.items()):
        for type_index in range(n_types):
            labels = [f'p{n_phones}_{type_index}_{i}' for i in range(n_phones)]
            for token_index in range(tokens_per_type):
                phrases.append(make_phrase([['edge'] + labels[:-1],
                    [labels[-1], 'edge']], start=token_index * 1000))
                lengths.append(n_phones)
    lexicon = cbl.CrossBoundaryLexicon(phrases)
    for ps, n_phones in zip(lexicon.phrase_sequences, lengths):
        ps.sequences = [s for s in ps.sequences if s.n_phones == n_phones]
    return lexicon


class ExtractionTests(unittest.TestCase):
    def test_extracts_all_and_only_single_boundary_spans(self):
        phrase = make_phrase([['a', 'b'], ['c', 'd'], ['e', 'f']])
        ps = cbl.PhraseSequences(phrase)
        expected = [(1, 3), (3, 5)]
        actual = [(s.start_index, s.end_index) for s in ps.sequences]
        self.assertEqual(actual, expected)
        for sequence in ps.sequences:
            with self.subTest(span=(sequence.start_index, sequence.end_index)):
                phones = phrase.phones[sequence.start_index:sequence.end_index]
                self.assertEqual(sequence.phones, tuple(phones))
                changes = sum(a.word.key != b.word.key
                    for a, b in zip(phones, phones[1:]))
                self.assertEqual(changes, 1)
                left = [p for p in phrase.phones if p.word is phones[0].word]
                right = [p for p in phrase.phones if p.word is phones[-1].word]
                self.assertIsNot(phones[0], left[0])
                self.assertIsNot(phones[-1], right[-1])
                self.assertIs(sequence.parent, ps)

    def test_orders_phones_by_time_without_reordering_phrase(self):
        phrase = make_phrase([['a', 'b'], ['c', 'd']])
        phrase.phones.reverse()
        original = list(phrase.phones)
        ps = cbl.PhraseSequences(phrase)
        self.assertEqual([s.phoneme_sequence for s in ps.sequences],
            ['b c'])
        self.assertEqual(phrase.phones, original)
        self.assertEqual([(s.start_index, s.end_index) for s in ps.sequences],
            [(1, 3)])

    def test_empty_and_single_word_phrases_have_no_candidates(self):
        for words in ([], [['a']], [['a', 'b', 'c']]):
            with self.subTest(words=words):
                self.assertEqual(cbl.PhraseSequences(
                    make_phrase(words)).sequences, [])

    def test_one_phone_words_cannot_supply_partial_word_spans(self):
        for words in ([['a'], ['n']], [['a'], ['n', 'd']],
            [['a', 'b'], ['n']]):
            with self.subTest(words=words):
                self.assertEqual(cbl.PhraseSequences(
                    make_phrase(words)).sequences, [])

    def test_equal_word_labels_still_have_distinct_word_boundaries(self):
        phrase = make_phrase([['a', 'a'], ['a', 'a'], ['a', 'a']])
        ps = cbl.PhraseSequences(phrase)
        self.assertEqual([(s.start_index, s.end_index) for s in ps.sequences],
            [(1, 3), (3, 5)])

    def test_rebuilding_sets_sequences_and_returns_none(self):
        ps = cbl.PhraseSequences(make_phrase([['x', 'a'], ['n', 'x']]))
        result = ps.make_cross_word_boundary_phone_sequences()
        self.assertIsNone(result)
        self.assertEqual(len(ps.sequences), 1)
        self.assertEqual(ps.sequences[0].phoneme_sequence, 'a n')


class SequenceTests(unittest.TestCase):
    def test_words_exposes_the_two_linked_word_objects_in_order(self):
        phrase = make_phrase([['a', 'b'], ['c', 'd'], ['e']])
        ps = cbl.PhraseSequences(phrase)
        for sequence in ps.sequences:
            with self.subTest(span=(sequence.start_index, sequence.end_index)):
                left, right = sequence.words
                self.assertIs(left, sequence.phones[0].word)
                self.assertIs(right, sequence.phones[-1].word)
                self.assertIsNot(left, right)

    def test_exposes_parent_phrase_audio_indices_times_and_labels(self):
        phrase = make_phrase([['tʃ', 'aː'], ['n', 'd']])
        ps = cbl.PhraseSequences(phrase)
        sequence = find_sequence(ps, 1, 3)
        self.assertIs(sequence.parent, ps)
        self.assertIs(sequence.phrase, phrase)
        self.assertIs(sequence.audio, phrase.audio)
        self.assertEqual(sequence.phoneme_sequence, 'aː n')
        self.assertEqual(sequence.n_phones, 2)
        self.assertEqual((sequence.start, sequence.end), (1100, 1300))
        self.assertEqual((sequence.start_index, sequence.end_index), (1, 3))
        self.assertEqual(repr(sequence), "cbs('aː n', 200ms)")

    def test_equality_compares_phone_strings_across_occurrences(self):
        first = cbl.PhraseSequences(make_phrase([['x', 'a'], ['n', 'x']],
            start=0))
        second = cbl.PhraseSequences(make_phrase([['x', 'a'], ['n', 'x']],
            start=5000))
        different = cbl.PhraseSequences(make_phrase([['x', 'a'], ['m', 'x']]))
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
            make_phrase([['a', 'b', 'c'], ['d', 'e', 'f'],
                ['g', 'h', 'i']]))

    def test_selection_blocks_overlap_but_allows_touching_intervals(self):
        first = find_sequence(self.ps, 1, 4)
        overlapping = find_sequence(self.ps, 2, 5)
        adjacent = find_sequence(self.ps, 4, 7)
        self.assertIs(self.ps.select(first), first)
        self.assertFalse(self.ps.can_select(first))
        self.assertFalse(self.ps.can_select(overlapping))
        with self.assertRaises(ValueError):
            self.ps.select(overlapping)
        self.assertTrue(self.ps.can_select(adjacent))
        self.ps.select(adjacent)
        self.assertEqual([(s.start_index, s.end_index)
            for s in self.ps.selected_sequences], [(1, 4), (4, 7)])

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
        ps = cbl.PhraseSequences(make_phrase([['x', 'a'], ['n', 'x'],
            ['x', 'a'], ['n', 'x']]))
        first = find_sequence(ps, 1, 3)
        second = find_sequence(ps, 5, 7)
        self.assertEqual(first, second)
        ps.select(first)
        ps.select(second)
        self.assertEqual(len(ps.selected_sequences), 2)

    def test_available_sequences_and_reset_follow_selection(self):
        self.ps.select(find_sequence(self.ps, 1, 4))
        self.assertEqual([(s.start_index, s.end_index)
            for s in self.ps.available_sequences()],
            [(4, 7), (4, 8), (5, 7), (5, 8)])
        self.assertEqual(repr(self.ps), 'ps(sequences=8, selected=1)')
        self.ps.reset_selection()
        self.assertEqual(self.ps.selected_sequences, [])
        self.assertEqual(len(self.ps.available_sequences()), 8)
        self.assertTrue(all(self.ps.can_select(s) for s in self.ps.sequences))

    def test_phrase_container_equality_uses_phrase_object_identity(self):
        same = cbl.PhraseSequences(self.ps.phrase)
        equivalent = cbl.PhraseSequences(
            make_phrase([['a', 'b', 'c'], ['d', 'e', 'f'],
                ['g', 'h', 'i']]))
        self.assertIsNone(self.ps.parent)
        self.assertEqual(self.ps.phrase, equivalent.phrase)
        self.assertEqual(self.ps, same)
        self.assertNotEqual(self.ps, equivalent)
        self.assertIs(self.ps.__eq__(self.ps.phrase), NotImplemented)


class LexiconTests(unittest.TestCase):
    def test_aggregated_counts_track_selection_and_reset(self):
        phrases = [make_phrase([['x', 'a'], ['n', 'x'],
            ['x', 'a'], ['n', 'x']]),
            make_phrase([['x', 'a'], ['n', 'x']])]
        lexicon = cbl.CrossBoundaryLexicon(phrases)
        expected = Counter({'a n': 3, 'x x': 1})
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
            Counter({'a n': 1, 'x x': 1}))
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


class LengthBinTests(unittest.TestCase):
    def test_bounds_allow_ten_percent_or_two_types_and_never_go_negative(self):
        for target, expected in ((0, (0, 2)), (1, (0, 3)), (10, (8, 12)),
            (25, (23, 27)), (100, (90, 110))):
            with self.subTest(target=target):
                length_bin = cbl.LengthBin(4, target)
                self.assertEqual((length_bin.lower_bound,
                    length_bin.upper_bound), expected)

    def test_bin_state_tracks_selected_types(self):
        length_bin = cbl.LengthBin(3, 5)
        self.assertTrue(length_bin.below_minimum)
        self.assertFalse(length_bin.full)
        length_bin.selected_types.extend(['a', 'b', 'c'])
        self.assertEqual(length_bin.n_selected_types, 3)
        self.assertFalse(length_bin.below_minimum)
        length_bin.selected_types.extend(['d', 'e', 'f', 'g'])
        self.assertEqual(length_bin.n_selected_types, 7)
        self.assertTrue(length_bin.full)

    def test_one_phone_quota_is_added_to_two_phone_bin(self):
        targets = {1: 5, 2: 2, 3: 1, 4: 0}
        bins = cbl.make_length_bins(targets)
        self.assertEqual({b.n_phones: b.target_count for b in bins},
            {2: 7, 3: 1})
        self.assertEqual(sum(b.target_count for b in bins),
            sum(targets.values()))
        self.assertEqual(targets, {1: 5, 2: 2, 3: 1, 4: 0})

    def test_merging_is_independent_of_mapping_order(self):
        for targets in ({1: 1, 2: 1, 3: 1}, {3: 1, 1: 1, 2: 1}):
            with self.subTest(targets=targets):
                bins = cbl.make_length_bins(targets)
                self.assertEqual([(b.n_phones, b.target_count) for b in bins],
                    [(2, 2), (3, 1)])

    def test_empty_distribution_is_allowed(self):
        self.assertEqual(cbl.make_length_bins({}), [])
        self.assertEqual(cbl.make_length_bins({1: 0, 2: 0}), [])

    def test_one_phone_quota_creates_two_phone_bin(self):
        for targets in ({1: 3}, {1: 3, 2: 0}, {1: 3, 3: 2}):
            with self.subTest(targets=targets):
                bins = cbl.make_length_bins(targets)
                counts = {b.n_phones: b.target_count for b in bins}
                self.assertEqual(counts[2], 3)
                self.assertEqual(sum(counts.values()), sum(targets.values()))

    def test_invalid_counts_are_rejected(self):
        for targets in ({0: 1}, {-1: 1}, {2.5: 1}, {2: -1}, {2: 1.5}):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    cbl.make_length_bins(targets)


class SamplingTests(unittest.TestCase):
    def assert_unselected(self, lexicon):
        for ps in lexicon.phrase_sequences:
            self.assertEqual(ps.selected_sequences, [])
            self.assertFalse(any(ps._occupied))

    def test_cycles_bins_and_returns_exact_token_counts_without_overlap(self):
        lexicon = make_sampling_lexicon({2: 8, 3: 8})
        sampled = lexicon.sample_lexicon({2: 5, 3: 5}, tokens_per_type=2)
        counts = Counter(s.phoneme_sequence for s in sampled)
        self.assertEqual(len(counts), 10)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual([s.n_phones for s in sampled[::2]], [2, 3] * 5)
        occupied = set()
        for sequence in sampled:
            for index in range(sequence.start_index, sequence.end_index):
                position = (id(sequence.parent), index)
                self.assertNotIn(position, occupied)
                occupied.add(position)
            self.assertTrue(any(sequence is selected
                for selected in sequence.parent.selected_sequences))
        self.assertEqual(lexicon.selected_sequence_counts, counts)

    def test_fills_all_minimums_before_using_spare_capacity(self):
        lexicon = make_sampling_lexicon({2: 5, 3: 12})
        sampled = lexicon.sample_lexicon({2: 2, 3: 10}, tokens_per_type=2)
        type_lengths = [s.n_phones for s in sampled[::2]]
        self.assertEqual(type_lengths[:8], [3] * 8)
        self.assertEqual(len(type_lengths), 12)
        self.assertEqual(Counter(type_lengths), {2: 2, 3: 10})

    def test_exhausted_bin_can_shift_counts_within_bounds(self):
        lexicon = make_sampling_lexicon({2: 3, 3: 7})
        sampled = lexicon.sample_lexicon({2: 5, 3: 5}, tokens_per_type=2)
        self.assertEqual(Counter(s.n_phones for s in sampled[::2]),
            {2: 3, 3: 7})
        self.assertEqual(len(sampled), 20)

    def test_unreachable_minimum_does_not_commit_any_selection(self):
        lexicon = make_sampling_lexicon({2: 2, 3: 20})
        with self.assertRaisesRegex(ValueError, 'selection unchanged'):
            lexicon.sample_lexicon({2: 5, 3: 5}, tokens_per_type=2)
        self.assert_unselected(lexicon)

    def test_unreachable_total_rolls_back_successful_minimum_phase(self):
        lexicon = make_sampling_lexicon({2: 3, 3: 3})
        with self.assertRaisesRegex(ValueError, 'selection unchanged'):
            lexicon.sample_lexicon({2: 5, 3: 5}, tokens_per_type=2)
        self.assert_unselected(lexicon)

    def test_does_not_exceed_upper_bound_even_when_other_bin_is_empty(self):
        lexicon = make_sampling_lexicon({2: 10})
        with self.assertRaisesRegex(ValueError, r'2: 3 \[0, 3\]'):
            lexicon.sample_lexicon({2: 1, 3: 2, 4: 2}, tokens_per_type=2)
        self.assert_unselected(lexicon)

    def test_overlapping_types_cannot_both_use_the_same_phone_spans(self):
        phrases = [make_phrase([['edge', 'a', 'b'], ['c', 'edge']])
            for _ in range(2)]
        lexicon = cbl.CrossBoundaryLexicon(phrases)
        self.assertEqual(sorted(lexicon.sequence_counts.values()), [2, 2])
        with self.assertRaisesRegex(ValueError, 'selection unchanged'):
            lexicon.sample_lexicon({2: 1, 3: 1}, tokens_per_type=2)
        self.assert_unselected(lexicon)

    def test_merged_total_excludes_one_phone_candidates(self):
        lexicon = make_sampling_lexicon({2: 5, 3: 5})
        lexicon.phrase_sequences[0].sequences.extend([
            SimpleNamespace(n_phones=1, phoneme_sequence='one')
            for _ in range(4)])
        sampled = lexicon.sample_lexicon({1: 2, 2: 2, 3: 2}, tokens_per_type=2)
        self.assertEqual(len(set(s.phoneme_sequence for s in sampled)), 6)
        self.assertTrue(all(s.n_phones >= 2 for s in sampled))

    def test_candidate_collection_filters_rare_types(self):
        lexicon = make_sampling_lexicon({2: 3}, tokens_per_type=2)
        lexicon.phrase_sequences.pop()
        candidates = lexicon.collect_candidate_tokens(tokens_per_type=2)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(len(tokens) == 2 for tokens in candidates.values()))

    def test_seed_reproduces_selection_and_reset_allows_resampling(self):
        lexicon = make_sampling_lexicon({2: 8, 3: 8})
        parent_positions = {id(ps): i
            for i, ps in enumerate(lexicon.phrase_sequences)}
        signatures = []
        for _ in range(2):
            sampled = lexicon.sample_lexicon({2: 3, 3: 3},
                tokens_per_type=2, random_seed=42)
            signatures.append([(s.phoneme_sequence,
                parent_positions[id(s.parent)], s.start_index, s.end_index)
                for s in sampled])
            lexicon.reset_selection()
        self.assertEqual(signatures[0], signatures[1])

    def test_rejects_existing_selections_without_changing_them(self):
        lexicon = make_sampling_lexicon({2: 2})
        ps = lexicon.phrase_sequences[0]
        selected = ps.select(ps.sequences[0])
        original_mask = bytes(ps._occupied)
        with self.assertRaisesRegex(ValueError, 'reset selections'):
            lexicon.sample_lexicon({2: 1}, tokens_per_type=2)
        self.assertIs(ps.selected_sequences[0], selected)
        self.assertEqual(bytes(ps._occupied), original_mask)

    def test_empty_targets_and_invalid_token_counts(self):
        lexicon = cbl.CrossBoundaryLexicon([])
        self.assertEqual(lexicon.sample_lexicon({}, tokens_per_type=2), [])
        for n_tokens in (0, -1, 1.5):
            with self.subTest(n_tokens=n_tokens):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    lexicon.sample_lexicon({}, tokens_per_type=n_tokens)

    def test_failed_type_does_not_leak_temporary_reservations(self):
        lexicon = make_sampling_lexicon({2: 1}, tokens_per_type=1)
        sequence = lexicon.phrase_sequences[0].sequences[0]
        occupied = {id(sequence.parent):
            bytearray(len(sequence.parent._occupied))}
        occupied[id(sequence.parent)][0] = 1
        original = {key: bytes(mask) for key, mask in occupied.items()}
        chosen = cbl.sample_non_overlapping_tokens([sequence, sequence],
            2, random.Random(0), occupied)
        self.assertEqual(chosen, [])
        self.assertEqual({key: bytes(mask) for key, mask in occupied.items()},
            original)
        self.assert_unselected(lexicon)


class FilteringTests(unittest.TestCase):
    def test_component_is_read_from_audio_path(self):
        phrase = make_phrase([], component='h')
        self.assertEqual(cbl.phrase_to_component(phrase), 'h')

    def test_default_filters_match_historical_component_exclusions(self):
        components = 'abcdefghijklmno'
        phrases = [make_phrase([['a'], ['n']], component=c) for c in components]
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
        phrases = [make_phrase([['a'], ['n']], component=c) for c in 'abc']
        filtered = cbl.filter_phrases_for_cross_boundary_lexicon(
            phrases, remove_components=['b'])
        self.assertEqual([cbl.phrase_to_component(p) for p in filtered],
            ['a', 'c'])
        self.assertEqual(cbl.filter_phrases_by_component(
            phrases, remove_components=''), phrases)
        self.assertEqual(cbl.filter_phrases_for_cross_boundary_lexicon([]), [])

    def test_dutch_variant_filter_defaults_to_netherlands_dutch(self):
        dutch = make_phrase([['a'], ['n']])
        flemish = make_phrase([['a'], ['n']], variant='vl')
        phrases = [flemish, dutch]
        self.assertEqual(cbl.filter_phrases_by_dutch_variant(phrases), [dutch])
        self.assertEqual(cbl.filter_phrases_by_dutch_variant(
            phrases, dutch_variant='vl'), [flemish])
        self.assertEqual(phrases, [flemish, dutch])

    def test_word_count_filter_keeps_phrases_with_at_least_two_words(self):
        phrases = [make_phrase([]), make_phrase([['a']]),
            make_phrase([['a'], ['n']]), make_phrase([['a'], ['n'], ['d']])]
        filtered = cbl.filter_phrases_with_insufficient_words(phrases)
        self.assertEqual(filtered, phrases[2:])
        self.assertIs(filtered[0], phrases[2])
        self.assertEqual(cbl.filter_phrases_with_insufficient_words(
            phrases, min_words=3), phrases[3:])

    def test_combined_filter_applies_variant_and_word_count_without_overlap(self):
        kept = make_phrase([['a'], ['n']])
        kept.overlap = True
        excluded_component = make_phrase([['a'], ['n']], component='a')
        flemish = make_phrase([['a'], ['n']], variant='vl')
        single_word = make_phrase([['a']])
        phrases = [excluded_component, flemish, single_word, kept]
        self.assertEqual(cbl.filter_phrases_for_cross_boundary_lexicon(
            phrases), [kept])
        self.assertEqual(cbl.filter_phrases_for_cross_boundary_lexicon(
            phrases, dutch_variant='vl'), [flemish])
        self.assertEqual(cbl.filter_phrases_for_cross_boundary_lexicon(
            phrases, min_words=1), [single_word, kept])


if __name__ == '__main__':
    unittest.main()
