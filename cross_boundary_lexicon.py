from collections import Counter
from itertools import groupby, pairwise
from math import isfinite
from numbers import Real
from pathlib import Path
import pickle

from progressbar import progressbar

import cross_boundary_lexicon_sampling as sampling
from cross_boundary_lexicon_sampling import SamplingPlan, plan_sampling
from cross_boundary_lexicon_sampling import inspect_lexicon_phonemes


class CrossBoundaryLexicon:
    '''Hold phrase sequences and aggregate counts by phoneme string.'''

    def __init__(self, phrases, min_duration_ms=100, max_duration_ms=1500):
        for name, value in (('min_duration_ms', min_duration_ms),
            ('max_duration_ms', max_duration_ms)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                or not isfinite(value) or value < 0):
                raise ValueError(f'{name} must be a finite nonnegative number')
        if min_duration_ms > max_duration_ms:
            raise ValueError('min_duration_ms must not exceed max_duration_ms')
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.successful_seed = None
        self.sampling_result = None
        ps = PhraseSequences
        print('building cross-boundary phrase sequences')
        self.phrase_sequences = [ps(phrase, parent=self)
            for phrase in progressbar(phrases)]

    def __repr__(self):
        n_sequences = sum(len(ps.sequences) for ps in self.phrase_sequences)
        m = f'cbl(phrases={len(self.phrase_sequences)}, '
        m += f'sequences={n_sequences})'
        return m

    @property
    def sequence_counts(self):
        '''Return candidate occurrence counts keyed by phoneme string.'''
        counts = Counter()
        for ps in self.phrase_sequences:
            for sequence in ps.sequences:
                counts[sequence.phoneme_sequence] += 1
        return counts

    @property
    def selected_sequence_counts(self):
        '''Return selected occurrence counts keyed by phoneme string.'''
        counts = Counter()
        for ps in self.phrase_sequences:
            for sequence in ps.selected_sequences:
                counts[sequence.phoneme_sequence] += 1
        return counts

    def available_sequence_counts(self):
        '''Return currently selectable occurrence counts by phoneme string.'''
        counts = Counter()
        for ps in self.phrase_sequences:
            for sequence in ps.available_sequences():
                counts[sequence.phoneme_sequence] += 1
        return counts

    def reset_selection(self):
        self.successful_seed = None
        self.sampling_result = None
        for ps in self.phrase_sequences:
            ps.reset_selection()

    def sample_lexicon(self, plan, random_seed=0, tries_per_round=3,
        max_attempts=100, min_types=2200):
        '''Search adaptive quotas, committing only the best successful sample.
        plan:             SamplingPlan containing token count and tolerance
        random_seed:      first seed; incremented once per attempt
        tries_per_round:  attempts before adjusting quotas
        max_attempts:     search budget, including attempts after success
        min_types:        floor for totals explored in 5% reduction steps
        Rank by normalized distribution error, then prefer larger totals.
        Inspect sampling_result for final quotas, counts, and successful seed.
        '''
        if any(ps.selected_sequences for ps in self.phrase_sequences):
            raise ValueError('reset selections before sampling a lexicon')
        result = sampling.sample_lexicon(self, plan, random_seed,
            tries_per_round, max_attempts, min_types)
        print('committing sampled tokens')
        for sequence in progressbar(result.tokens):
            sequence.parent.select(sequence)
        self.sampling_result = result
        self.successful_seed = result.seed
        return result.tokens

    def make_markers(self, store, save=False,
        keys_path='data/cross_boundary_marker_keys.pkl'):
        '''Return markers for the retained sample, optionally saving in bulk.
        Use the phraser store containing the source phrases. Repeated calls
        create new marker identifiers; no manifest or deduplication is added.
        When saving, pickle the ordered marker keys to keys_path afterward.
        Set keys_path=None to skip the key file. Previews never write keys.
        '''
        if self.sampling_result is None:
            raise ValueError('sample the lexicon before making markers')
        from phraser import make_markers

        if save and keys_path is not None:
            keys_path = Path(keys_path)
            keys_path.parent.mkdir(parents=True, exist_ok=True)
        data = [dict(label=token.phoneme_sequence, start=token.start,
            end=token.end, audio=token.audio, phrase=token.phrase)
            for token in self.sampling_result.tokens]
        markers = make_markers(data, store=store, save=save)
        if save and keys_path is not None:
            with keys_path.open('wb') as f:
                pickle.dump([marker.key for marker in markers], f)
        return markers

    def collect_candidate_tokens(self, tokens_per_type):
        '''Group tokens by phoneme string, keeping sufficiently frequent types.
        Exclude one-phone sequences, which cannot cross a word boundary.
        Apply instance duration limits inclusively before counting frequency.
        Frequency is a preliminary filter; tokens may still overlap.
        '''
        tokens_by_type = {}
        print('grouping candidate tokens by phoneme string')
        for ps in progressbar(self.phrase_sequences):
            for sequence in ps.sequences:
                if sequence.n_phones < 2: continue
                duration = sequence.end - sequence.start
                if not self.min_duration_ms <= duration <= self.max_duration_ms:
                    continue
                label = sequence.phoneme_sequence
                tokens_by_type.setdefault(label, []).append(sequence)
        return {label: tokens for label, tokens in tokens_by_type.items()
            if len(tokens) >= tokens_per_type}


class PhraseSequences:
    '''Hold a phrase's candidates and select spans without shared phones.'''

    def __init__(self, phrase, parent=None):
        self.phrase = phrase
        self.parent = parent
        self.make_cross_word_boundary_phone_sequences()
        self.selected_sequences = []
        self._occupied = bytearray(len(phrase.phones))

    def __repr__(self):
        m = f'ps(sequences={len(self.sequences)}, '
        m += f'selected={len(self.selected_sequences)})'
        return m

    def __eq__(self, other):
        if not isinstance(other, PhraseSequences): return NotImplemented
        return self.phrase is other.phrase

    def make_cross_word_boundary_phone_sequences(self):
        '''Set sequences crossing one boundary, excluding full linked words.
        Omit the left word's first phone and the right word's last phone.
        '''
        phones = sorted(self.phrase.phones, key=lambda phone: phone.start)
        groups = groupby(phones, key=lambda phone: phone.word.key)
        word_phones = [tuple(group) for _, group in groups]
        self.sequences = []
        word_start = 0
        cbs = CrossBoundarySequence

        for left, right in pairwise(word_phones):
            for start in range(1, len(left)):
                for stop in range(1, len(right)):
                    phones = left[start:] + right[:stop]
                    start_index = word_start + start
                    sequence = cbs(phones, start_index, self)
                    self.sequences.append(sequence)
            word_start += len(left)

    def can_select(self, sequence):
        if sequence.parent is not self: return False
        return not any(self._occupied[sequence.start_index:sequence.end_index])

    def select(self, sequence):
        if not self.can_select(sequence):
            message = 'sequence overlaps selection or is from another container'
            raise ValueError(message)
        occupied = b'\x01' * sequence.n_phones
        self._occupied[sequence.start_index:sequence.end_index] = occupied
        self.selected_sequences.append(sequence)
        return sequence

    def reset_selection(self):
        self.selected_sequences.clear()
        self._occupied = bytearray(len(self._occupied))

    def available_sequences(self):
        return [sequence for sequence in self.sequences
            if self.can_select(sequence)]


class CrossBoundarySequence:
    '''One occurrence; indices address time-ordered phrase phones.
    start_index: inclusive phone index; end_index is exclusive
    start, end:  audio timestamps in milliseconds
    '''

    def __init__(self, phones, start_index, parent):
        self.parent = parent
        self.phones = tuple(phones)
        if not self.phones: raise ValueError('phones must not be empty')
        self.start_index = start_index
        self.end_index = start_index + len(self.phones)
        self.phoneme_sequence = ' '.join(phone.label for phone in self.phones)
        self.start = self.phones[0].start
        self.end = self.phones[-1].end
        self.n_phones = len(self.phones)

    def __repr__(self):
        return f'cbs({self.phoneme_sequence!r}, {self.end - self.start}ms)'

    @property
    def phrase(self):
        return self.parent.phrase

    @property
    def audio(self):
        return self.phrase.audio

    @property
    def words(self):
        return self.phones[0].word, self.phones[-1].word

    @property
    def has_vowel(self):
        return any(phone.type == 'vowel' for phone in self.phones)

    def max_phone_gap(self):
        '''Return the largest gap between consecutive phones in milliseconds.'''
        return max((right.start - left.end
            for left, right in pairwise(self.phones)), default=0)

    def __eq__(self, other):
        if not isinstance(other, CrossBoundarySequence): return NotImplemented
        return self.phoneme_sequence == other.phoneme_sequence


def phrase_to_component(phrase):
    '''Return the CGN component encoded in the phrase's audio path.'''
    return Path(phrase.audio.filename).parent.parent.name.split('-')[-1]


def filter_phrases_by_component(phrases, remove_components='acdhm'):
    '''Return phrases outside the excluded CGN components.'''
    return [phrase for phrase in phrases
        if phrase_to_component(phrase) not in remove_components]


def filter_phrases_by_dutch_variant(phrases, dutch_variant='nl'):
    '''Return phrases from the requested Dutch variant's audio directory.'''
    return [phrase for phrase in phrases
        if Path(phrase.audio.filename).parent.name == dutch_variant]


def filter_phrases_with_insufficient_words(phrases, min_words=2):
    '''Return phrases containing at least min_words words.'''
    return [phrase for phrase in phrases if len(phrase.words) >= min_words]


def filter_phrases_for_cross_boundary_lexicon(phrases, remove_components='acdhm',
    dutch_variant='nl', min_words=2):
    '''Apply phrase filters and return a filtered list of phrases.'''
    print('filtering phrases by component')
    phrases = filter_phrases_by_component(progressbar(phrases),
        remove_components)
    print('filtering phrases by Dutch variant')
    phrases = filter_phrases_by_dutch_variant(progressbar(phrases),
        dutch_variant)
    print('filtering phrases by word count')
    return filter_phrases_with_insufficient_words(progressbar(phrases),
        min_words)
