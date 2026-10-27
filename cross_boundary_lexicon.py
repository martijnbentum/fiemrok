from collections import Counter
from itertools import groupby, pairwise
from pathlib import Path
import random

from progressbar import ProgressBar, progressbar


class CrossBoundaryLexicon:
    '''Hold phrase sequences and aggregate counts by phoneme string.'''

    def __init__(self, phrases):
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
        for ps in self.phrase_sequences:
            ps.reset_selection()

    def sample_lexicon(self, sequence_types_by_phone_count, tokens_per_type=100,
        random_seed=0):
        '''Return and select tokens within each length bin's count bounds.
        sequence_types_by_phone_count: phone count to target number of types
        tokens_per_type:               non-overlapping tokens per type, exact
        random_seed:                   seed for type and token sampling
        Add one-phone targets to the two-phone bin.
        Allow deviations of 10% or at least two types, with a minimum of zero.
        Fill bin minimums first, then reach the exact total within maximums.
        Sampling is greedy; failure does not rule out a valid selection.
        Existing selections must be empty. Failure leaves selection unchanged.
        '''
        if not isinstance(tokens_per_type, int) or tokens_per_type < 1:
            raise ValueError('tokens_per_type must be a positive integer')
        if any(ps.selected_sequences for ps in self.phrase_sequences):
            raise ValueError('reset selections before sampling a lexicon')
        bins = make_length_bins(sequence_types_by_phone_count)
        if not bins: return []

        rng = random.Random(random_seed)
        tokens_by_type = self.collect_candidate_tokens(tokens_per_type)
        types_by_length = group_sequence_types_by_length(tokens_by_type, rng)
        for length_bin in bins:
            length_bin.remaining_types = types_by_length.get(
                length_bin.n_phones, [])

        reserved_spans = {id(ps): bytearray(len(ps._occupied))
            for ps in self.phrase_sequences}
        n_types = sum(length_bin.target_count for length_bin in bins)
        print(f'sampling {n_types} types, {tokens_per_type} tokens each')
        # Meet each bin's minimum.
        sampled_tokens = sample_types_from_bins(bins, tokens_by_type,
            tokens_per_type, rng, reserved_spans, minimums_only=True)
        # Fill the remaining total within bin maximums.
        sampled_tokens.extend(sample_types_from_bins(bins, tokens_by_type,
            tokens_per_type, rng, reserved_spans, minimums_only=False))

        # Commit phrase selections only after both sampling phases succeed.
        print('committing sampled tokens')
        for sequence in progressbar(sampled_tokens):
            sequence.parent.select(sequence)
        return sampled_tokens

    def collect_candidate_tokens(self, tokens_per_type):
        '''Group tokens by phoneme string, keeping sufficiently frequent types.
        Exclude one-phone sequences, which cannot cross a word boundary.
        Frequency is a preliminary filter; tokens may still overlap.
        '''
        tokens_by_type = {}
        print('grouping candidate tokens by phoneme string')
        for ps in progressbar(self.phrase_sequences):
            for sequence in ps.sequences:
                if sequence.n_phones < 2: continue
                label = sequence.phoneme_sequence
                tokens_by_type.setdefault(label, []).append(sequence)
        return {label: tokens for label, tokens in tokens_by_type.items()
            if len(tokens) >= tokens_per_type}


class LengthBin:
    '''Track candidate and selected sequence types for one phone count.
    Allow 10% of the target count, rounded down, or at least two types.
    Bounds are inclusive; the lower bound cannot be negative.
    Token overlap is handled by the sampling functions, outside this class.
    '''

    def __init__(self, n_phones, target_count):
        self.n_phones = n_phones
        self.target_count = target_count
        tolerance = max(2, target_count // 10)
        self.lower_bound = max(0, target_count - tolerance)
        self.upper_bound = target_count + tolerance
        self.remaining_types = []
        self.selected_types = []

    @property
    def n_selected_types(self):
        return len(self.selected_types)

    @property
    def below_minimum(self):
        return self.n_selected_types < self.lower_bound

    @property
    def full(self):
        return self.n_selected_types >= self.upper_bound


def make_length_bins(sequence_types_by_phone_count):
    '''Build length bins, adding one-phone targets to the two-phone bin.'''
    targets = Counter()
    for n_phones, n_types in sequence_types_by_phone_count.items():
        if not isinstance(n_phones, int) or n_phones < 1:
            raise ValueError('phone counts must be positive integers')
        if not isinstance(n_types, int) or n_types < 0:
            raise ValueError('type counts must be non-negative integers')
        if n_types:
            targets[max(2, n_phones)] += n_types
    return [LengthBin(n, count) for n, count in sorted(targets.items())]


def group_sequence_types_by_length(tokens_by_type, rng):
    '''Group type labels by number of phones and shuffle within each bin.'''
    types_by_length = {}
    for label in sorted(tokens_by_type):
        n_phones = tokens_by_type[label][0].n_phones
        types_by_length.setdefault(n_phones, []).append(label)
    for labels in types_by_length.values():
        rng.shuffle(labels)
    return types_by_length


def sample_types_from_bins(bins, tokens_by_type, tokens_per_type, rng,
    reserved_spans, minimums_only=False):
    '''Cycle bins, sampling at most one type per bin in each round.
    First meet all lower bounds; otherwise fill to the total target count.
    Modify bins and temporary occupancy only; return newly sampled tokens.
    Exhausted bins are skipped. Raise if a round cannot make progress.
    '''
    if minimums_only:
        target_total_types = sum(length_bin.lower_bound for length_bin in bins)
    else:
        target_total_types = sum(length_bin.target_count for length_bin in bins)
    n_selected_types = sum(length_bin.n_selected_types for length_bin in bins)
    sampled_tokens = []
    initial_selected_types = n_selected_types
    remaining_types = target_total_types - n_selected_types
    if not remaining_types: return sampled_tokens
    phase = 'bin minimums' if minimums_only else 'remaining total'
    print(f'sampling {remaining_types} types for {phase}')
    with ProgressBar(max_value=remaining_types) as bar:
        while n_selected_types < target_total_types:
            made_progress = False
            for length_bin in bins:
                if n_selected_types == target_total_types: break
                if length_bin.full: continue
                if minimums_only and not length_bin.below_minimum: continue
                chosen_tokens = sample_next_type_from_bin(length_bin,
                    tokens_by_type, tokens_per_type, rng, reserved_spans)
                if not chosen_tokens: continue
                sampled_tokens.extend(chosen_tokens)
                n_selected_types += 1
                made_progress = True
                bar.update(n_selected_types - initial_selected_types)
            if not made_progress:
                message = format_sampling_failure(bins, tokens_per_type)
                raise ValueError(message)
    return sampled_tokens


def format_sampling_failure(bins, tokens_per_type):
    '''Describe the failed sampling attempt and each bin's current count.'''
    counts = ', '.join(f'{b.n_phones}: {b.n_selected_types} '
        f'[{b.lower_bound}, {b.upper_bound}]' for b in bins)
    message = f'cannot meet type quotas with {tokens_per_type} '
    message += 'non-overlapping tokens per type; '
    message += f'bin counts [minimum, maximum]: {counts}; '
    message += 'selection unchanged; try another seed or more phrases'
    return message


def sample_next_type_from_bin(length_bin, tokens_by_type, n_tokens, rng,
    reserved_spans):
    '''Try unused types until one supplies n_tokens non-overlapping tokens.
    Record a successful type in the bin and reserve its temporary spans.
    Discard unsuccessful types; return [] if the bin runs out of types.
    '''
    while length_bin.remaining_types:
        label = length_bin.remaining_types.pop()
        chosen_tokens = sample_non_overlapping_tokens(tokens_by_type[label],
            n_tokens, rng, reserved_spans)
        if chosen_tokens:
            length_bin.selected_types.append(label)
            return chosen_tokens
    return []


def sample_non_overlapping_tokens(candidates, n_tokens, rng, reserved_spans):
    '''Return n_tokens occurrences and reserve their temporary phone spans.
    Try shuffled tokens greedily, without backtracking.
    Return [] without changing reserved_spans if this attempt falls short.
    Trial spans become shared reservations only when the full type succeeds.
    '''
    candidates = list(candidates)
    rng.shuffle(candidates)
    trial_spans = {}
    chosen_tokens = []
    for sequence in candidates:
        parent_id = id(sequence.parent)
        if parent_id not in trial_spans:
            trial_spans[parent_id] = bytearray(reserved_spans[parent_id])
        mask = trial_spans[parent_id]
        start, end = sequence.start_index, sequence.end_index
        if any(mask[start:end]): continue
        mask[start:end] = b'\x01' * sequence.n_phones
        chosen_tokens.append(sequence)
        if len(chosen_tokens) == n_tokens:
            reserved_spans.update(trial_spans)
            return chosen_tokens
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
    print('filtering phrases by component')
    phrases = filter_phrases_by_component(progressbar(phrases),
        remove_components)
    print('filtering phrases by Dutch variant')
    phrases = filter_phrases_by_dutch_variant(progressbar(phrases),
        dutch_variant)
    print('filtering phrases by word count')
    return filter_phrases_with_insufficient_words(progressbar(phrases),
        min_words)


def inspect_lexicon_phonemes():
    '''Return length and phone counters from one pronunciation per word type.
    Load layer 6 from lexicon100_2466_cgn. Each word type contributes its
    most frequent token pronunciation; ties use lexical phone-label order.
    Print token/type counts, token counts per type, and one-phone types.
    Return (n_types_by_phone_count, phone_counts).
    '''
    from stores import _attach_cgn
    from validate_lexicon import load_metadata_layer, load_store_w2v

    pronunciations = {}
    embedding_store = load_store_w2v()
    try:
        _attach_cgn(embedding_store)
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

    n_types_by_phone_count = Counter()
    phone_counts = Counter()
    tokens_per_type = Counter()
    for label, counts in pronunciations.items():
        tokens_per_type[sum(counts.values())] += 1
        pronunciation = min(counts, key=lambda p: (-counts[p], p))
        n_types_by_phone_count[len(pronunciation)] += 1
        phone_counts.update(pronunciation)
        if len(pronunciation) == 1:
            print(f'one-phone type: {label!r} -> {pronunciation[0]!r}')
    print(f'{len(metadatas)} tokens, {len(pronunciations)} word types')
    print(f'tokens per type -> number of types: {dict(tokens_per_type)}')
    return n_types_by_phone_count, phone_counts
