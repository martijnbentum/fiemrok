'''Select the word dataset for lexicon validation.'''

from collections import defaultdict
from math import isfinite
from numbers import Real
from pathlib import Path
import pickle
import random

import lexicon_helper
from .cross_boundary_lexicon_sampling import validate_integer


def filter_word_tokens(words, n_tokens=100, iqr_multiplier=5,
    min_duration_ms=100, max_duration_ms=1500):
    '''Filter words from the already selected recording material.
    words:              local word objects; durations are in milliseconds
    n_tokens:           minimum eligible occurrences per orthographic type
    iqr_multiplier:     per-type fence multiplier, applied before frequency
    '''
    validate_integer(n_tokens, 'n_tokens', minimum=1)
    bounds = (min_duration_ms, max_duration_ms)
    if any(isinstance(value, bool) or not isinstance(value, Real)
        or not isfinite(value) or value < 0 for value in bounds):
        raise ValueError('duration bounds must be finite and nonnegative')
    if min_duration_ms > max_duration_ms:
        raise ValueError('duration bounds must be ordered and nonnegative')
    lexicon_helper.filter_outlier_tokens_from_type([], iqr_multiplier)
    words = [word for word in words if len(word.phones) >= 2]
    words = lexicon_helper.filter_duration(words, min_duration_ms,
        max_duration_ms)
    grouped = defaultdict(list)
    for word in words:
        grouped[word.label].append(word)
    filtered = []
    for label in sorted(grouped):
        tokens = lexicon_helper.filter_outlier_tokens_from_type(
            grouped[label], iqr_multiplier=iqr_multiplier)
        if len(tokens) >= n_tokens:
            filtered.extend(sorted(tokens, key=lambda word: word.key))
    return filtered


def select_words(filtered_words, n_tokens=100, random_seed=0):
    '''Sample exactly n_tokens per type in reproducible label/key order.'''
    validate_integer(n_tokens, 'n_tokens', minimum=1)
    validate_integer(random_seed, 'random_seed', minimum=None)
    grouped = defaultdict(list)
    for word in filtered_words:
        grouped[word.label].append(word)
    rng = random.Random(random_seed)
    selection = []
    for label in sorted(grouped):
        tokens = sorted(grouped[label], key=lambda word: word.key)
        if len(tokens) < n_tokens:
            raise ValueError(f'type {label!r} has fewer than {n_tokens} tokens')
        selection.extend(rng.sample(tokens, n_tokens))
    return selection


def save_selected_word_keys(keys, path):
    '''Save an ordered word-key selection.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(list(keys), handle)


def load_selected_word_keys(path):
    '''Load an ordered word-key selection.'''
    with Path(path).open('rb') as handle:
        return pickle.load(handle)
