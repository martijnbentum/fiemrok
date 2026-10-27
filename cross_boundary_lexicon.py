from itertools import groupby, pairwise


def phrase_to_cross_word_boundary_phone_sequence(phrase):
    '''Return phone sequences crossing exactly one word boundary.'''
    phones = sorted(phrase.phones, key=lambda phone: phone.start)
    groups = groupby(phones, key=lambda phone: phone.word.key)
    word_phones = [tuple(group) for _, group in groups]
    sequences = []

    for left, right in pairwise(word_phones):
        for start in range(len(left)):
            for stop in range(1, len(right) + 1):
                sequences.append(left[start:] + right[:stop])

    return sequences
