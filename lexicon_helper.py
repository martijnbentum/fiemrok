'''Shared, store-independent filters for lexicon selection.'''

from collections import Counter
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from statistics import quantiles


def filename_to_component(filename):
    return Path(filename).parent.parent.name.split('-')[-1]


def filename_to_dutch_variant(filename):
    return Path(filename).parent.name


def filter_components(items, remove_components='acdhm'):
    '''Filter words or phrases by their audio's CGN component.'''
    return [item for item in items
        if filename_to_component(item.audio.filename) not in remove_components]


def filter_dutch_variant(items, dutch_variant='nl', strict=False):
    '''Keep one variant; strict mode rejects paths outside nl and vl.'''
    selected = []
    for item in items:
        variant = filename_to_dutch_variant(item.audio.filename)
        if strict and variant not in ('nl', 'vl'):
            raise ValueError(f'unknown Dutch variant {variant}')
        if variant == dutch_variant: selected.append(item)
    return selected


def filter_phrases_with_insufficient_words(phrases, min_words=2):
    return [phrase for phrase in phrases if len(phrase.words) >= min_words]


def duration_is_eligible(duration, min_dur=100, max_dur=1500):
    '''Check inclusive duration bounds in milliseconds.'''
    return min_dur <= duration <= max_dur


def filter_duration(words, min_dur=100, max_dur=1500):
    return [word for word in words
        if duration_is_eligible(word.duration, min_dur, max_dur)]


def filter_outlier_tokens_from_type(tokens, iqr_multiplier=1.5, min_tokens=5):
    '''Return one type's tokens within inclusive duration IQR fences.
    tokens:          words with duration, or sequences with start/end
    iqr_multiplier:  fence distance below Q1 and above Q3 in IQR units
    min_tokens:      minimum group size for filtering; smaller groups pass

    Use inclusive, linearly interpolated quartiles. Preserve input order.
    With zero IQR, retain only durations equal to the shared quartile.
    Durations must be finite and positive, in consistent units. Apply before
    frequency eligibility and sampling; this can reduce a type below quota.
    '''
    if (isinstance(iqr_multiplier, bool)
        or not isinstance(iqr_multiplier, Real)
        or not isfinite(iqr_multiplier) or iqr_multiplier < 0):
        raise ValueError('iqr_multiplier must be finite and nonnegative')
    if (isinstance(min_tokens, bool) or not isinstance(min_tokens, Integral)
        or min_tokens < 4):
        raise ValueError('min_tokens must be an integer of at least 4')
    tokens = list(tokens)
    durations = []
    for token in tokens:
        duration = getattr(token, 'duration', None)
        if duration is None: duration = token.end - token.start
        if (isinstance(duration, bool) or not isinstance(duration, Real)
            or not isfinite(duration) or duration <= 0):
            raise ValueError('token durations must be finite and positive')
        durations.append(duration)
    if len(tokens) < min_tokens: return tokens
    q1, _, q3 = quantiles(durations, n=4, method='inclusive')
    margin = iqr_multiplier * (q3 - q1)
    return [token for token, duration in zip(tokens, durations)
        if q1 - margin <= duration <= q3 + margin]


def remove_overlap(words):
    '''Exclude annotated speech overlap, not shared selected phones.'''
    return [word for word in words if not word.overlap]


def filter_neighbour_overlap(audio, n_neighbours=5):
    '''Keep words whose same-speaker neighbourhood has no speech overlap.
    Check the word itself and up to n_neighbours preceding and following
    words from its speaker, ordered by time across phrase boundaries.
    Overlap means positive shared time with a word from another speaker;
    touching boundaries are allowed. Words within one speaker do not overlap.
    Return words in the original audio.words order without changing the audio.
    '''
    if (isinstance(n_neighbours, bool)
        or not isinstance(n_neighbours, Integral) or n_neighbours < 0):
        raise ValueError('n_neighbours must be a nonnegative integer')
    words = list(audio.words)
    if len(audio.speakers) <= 1: return words
    ordered = sorted(range(len(words)),
        key=lambda index: (words[index].start, words[index].end))
    overlapping = set()
    by_speaker = {}
    active = []
    for index in ordered:
        word = words[index]
        speaker = word.speaker_key
        by_speaker.setdefault(speaker, []).append(index)
        active = [other for other in active if words[other].end > word.start]
        for other in active:
            if (words[other].speaker_key != speaker
                and words[other].start < word.end):
                overlapping.update((index, other))
        active.append(index)
    excluded = set()
    for indices in by_speaker.values():
        counts = [0]
        for index in indices:
            counts.append(counts[-1] + (index in overlapping))
        for position, index in enumerate(indices):
            start = max(0, position - n_neighbours)
            end = min(len(indices), position + n_neighbours + 1)
            if counts[end] > counts[start]: excluded.add(index)
    return [word for index, word in enumerate(words) if index not in excluded]


def filter_on_n_syllables(words, n_syllables=1):
    if n_syllables is None: return words
    return [word for word in words if len(word.syllables) == n_syllables]


def make_frequency_dict(words, min_len_label=2):
    counts = Counter(word.label for word in words)
    return {label: count for label, count in counts.items()
        if len(label) >= min_len_label}


def filter_frequency(frequency_dict, words, min_freq=100, max_freq=None):
    '''Filter using counts from the already eligible word population.'''
    return [word for word in words if word.label in frequency_dict
        and (min_freq is None or frequency_dict[word.label] >= min_freq)
        and (max_freq is None or frequency_dict[word.label] <= max_freq)]
