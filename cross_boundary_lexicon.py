from collections import Counter
from itertools import groupby, pairwise
from pathlib import Path
import random

from progressbar import progressbar


class CrossBoundaryLexicon:
    '''Hold phrase sequences and aggregate counts by phoneme string.'''

    def __init__(self, phrases):
        ps = PhraseSequences
        self.phrase_sequences = [ps(phrase, parent=self) for phrase in phrases]

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
        for ps in self.phrase_sequences:
            ps.reset_selection()

    def sample_lexicon(self, phone_count_counts, tokens_per_type=100,
        random_seed=0):
        '''Return and select a full lexicon, approximately matching lengths.
        phone_count_counts: reference phone count to number of word types
        tokens_per_type:    exact number of non-overlapping tokens per type
        random_seed:        seed for type, token, and target-length sampling
        Existing selections must be empty. Selection is unchanged on failure.
        '''
        if not isinstance(tokens_per_type, int) or tokens_per_type < 1:
            raise ValueError('tokens_per_type must be a positive integer')
        if any(ps.selected_sequences for ps in self.phrase_sequences):
            raise ValueError('reset selections before sampling a lexicon')
        rng = random.Random(random_seed)
        counts = self.sequence_counts
        candidates = {}
        print('grouping candidate tokens by phoneme string')
        for ps in progressbar(self.phrase_sequences):
            for sequence in ps.sequences:
                label = sequence.phoneme_sequence
                if counts[label] < tokens_per_type: continue
                candidates.setdefault(label, []).append(sequence)

        targets = []
        for n_phones, n_types in sorted(phone_count_counts.items()):
            targets.extend([n_phones] * n_types)
        if len(candidates) < len(targets):
            message = f'only {len(candidates)} candidate types for '
            message += f'{len(targets)} requested types'
            raise ValueError(message)
        rng.shuffle(targets)
        types_by_length = {}
        for label in sorted(candidates):
            n_phones = candidates[label][0].n_phones
            types_by_length.setdefault(n_phones, []).append(label)
        for labels in types_by_length.values():
            rng.shuffle(labels)

        occupied = {}
        for ps in self.phrase_sequences:
            occupied[id(ps)] = bytearray(len(ps._occupied))
        sampled = []
        print(f'sampling {len(targets)} types, {tokens_per_type} tokens each')
        for target in progressbar(targets):
            lengths = sorted(types_by_length, key=lambda n: (abs(n - target), n))
            chosen = []
            for n_phones in lengths:
                labels = types_by_length[n_phones]
                while labels:
                    label = labels.pop()
                    chosen = self._sample_type_tokens(candidates[label],
                        tokens_per_type, rng, occupied)
                    if chosen: break
                if chosen: break
            if not chosen:
                n_types = len(sampled) // tokens_per_type
                message = f'could select only {n_types} of {len(targets)} types '
                message += f'with {tokens_per_type} non-overlapping tokens each; '
                message += 'selection unchanged; try another seed or more phrases'
                raise ValueError(message)
            sampled.extend(chosen)

        for sequence in sampled:
            sequence.parent.select(sequence)
        return sampled

    def _sample_type_tokens(self, candidates, n_tokens, rng, occupied):
        '''Reserve a complete type in temporary masks, or return no tokens.'''
        candidates = list(candidates)
        rng.shuffle(candidates)
        trial = {}
        chosen = []
        for sequence in candidates:
            parent_id = id(sequence.parent)
            if parent_id not in trial:
                trial[parent_id] = bytearray(occupied[parent_id])
            mask = trial[parent_id]
            start, end = sequence.start_index, sequence.end_index
            if any(mask[start:end]): continue
            mask[start:end] = b'\x01' * sequence.n_phones
            chosen.append(sequence)
            if len(chosen) == n_tokens:
                occupied.update(trial)
                return chosen
        return []


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
    phrases = filter_phrases_by_component(phrases, remove_components)
    phrases = filter_phrases_by_dutch_variant(phrases, dutch_variant)
    return filter_phrases_with_insufficient_words(phrases, min_words)


def inspect_lexicon_phonemes():
    '''Return length and phone counters from one pronunciation per word type.
    Load layer 6 from lexicon100_2466_cgn. Each word type contributes its
    most frequent token pronunciation; ties use lexical phone-label order.
    Print token/type counts, token counts per type, and one-phone types.
    Return (phone_count_counts, phone_counts).
    '''
    from validate_lexicon import load_metadata_layer, load_store_w2v

    pronunciations = {}
    embedding_store = load_store_w2v()
    try:
        metadatas = load_metadata_layer(embedding_store, 6)
        if not metadatas: raise ValueError('no lexicon metadata for layer 6')
        print('counting pronunciations in CGN lexicon layer 6')
        for metadata in progressbar(metadatas):
            word = metadata.phraser_object
            phones = sorted(word.phones, key=lambda phone: phone.start)
            pronunciation = tuple(phone.label for phone in phones)
            if not pronunciation:
                raise ValueError(f'no phones found for word {word.label}')
            counts = pronunciations.setdefault(word.label, Counter())
            counts[pronunciation] += 1
    finally:
        embedding_store.close()

    phone_count_counts = Counter()
    phone_counts = Counter()
    tokens_per_type = Counter()
    for label, counts in pronunciations.items():
        tokens_per_type[sum(counts.values())] += 1
        pronunciation = min(counts, key=lambda p: (-counts[p], p))
        phone_count_counts[len(pronunciation)] += 1
        phone_counts.update(pronunciation)
        if len(pronunciation) == 1:
            print(f'one-phone type: {label!r} -> {pronunciation[0]!r}')
    print(f'{len(metadatas)} tokens, {len(pronunciations)} word types')
    print(f'tokens per type -> number of types: {dict(tokens_per_type)}')
    return phone_count_counts, phone_counts
