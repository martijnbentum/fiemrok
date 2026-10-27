from itertools import groupby, pairwise


def phrase_to_cross_word_boundary_phone_sequence(phrase):
    '''Return CrossBoundarySequence instances crossing one word boundary.'''
    phones = sorted(phrase.phones, key=lambda phone: phone.start)
    groups = groupby(phones, key=lambda phone: phone.word.key)
    word_phones = [tuple(group) for _, group in groups]
    sequences = []
    word_start = 0

    for left, right in pairwise(word_phones):
        for start in range(len(left)):
            for stop in range(1, len(right) + 1):
                sequence = CrossBoundarySequence(phrase,
                    left[start:] + right[:stop], word_start + start)
                sequences.append(sequence)
        word_start += len(left)

    return sequences


class PhraseSequences:
    '''Hold a phrase's candidates and select spans without shared phones.'''

    def __init__(self, phrase):
        self.phrase = phrase
        self.sequences = phrase_to_cross_word_boundary_phone_sequence(phrase)
        self.selected_sequences = []
        self._occupied = bytearray(len(phrase.phones))

    def can_select(self, sequence):
        if sequence.phrase is not self.phrase: return False
        return not any(self._occupied[sequence.start_index:sequence.end_index])

    def select(self, sequence):
        if not self.can_select(sequence):
            message = 'sequence overlaps selection or is from another phrase'
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

    def __init__(self, phrase, phones, start_index):
        self.phrase = phrase
        self.phones = tuple(phones)
        if not self.phones: raise ValueError('phones must not be empty')
        self.start_index = start_index
        self.end_index = start_index + len(self.phones)
        self.phoneme_sequence = ' '.join(phone.label for phone in self.phones)
        self.start = self.phones[0].start
        self.end = self.phones[-1].end
        self.n_phones = len(self.phones)

    @property
    def audio(self):
        return self.phrase.audio

    def __eq__(self, other):
        if not isinstance(other, CrossBoundarySequence): return NotImplemented
        return self.phoneme_sequence == other.phoneme_sequence
