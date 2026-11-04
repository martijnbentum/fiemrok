'''Select the word dataset for lexicon validation.'''

from collections import defaultdict
from math import isfinite
from numbers import Real
from pathlib import Path
import pickle
import random
from time import monotonic

from progressbar import progressbar

import lexicon_helper
from .cross_boundary_sampling import validate_integer


def _group_words(words, description):
    '''Group streamed words while showing loading progress.'''
    print(f'{description}: loading and grouping word tokens...', flush=True)
    started = monotonic()
    grouped = defaultdict(list)
    for word in progressbar(words):
        grouped[word.label].append(word)
    counts = {label: len(tokens) for label, tokens in grouped.items()}
    print(f'{description}: {len(counts):,} types; '
        f'{sum(counts.values()):,} tokens; '
        f'{monotonic() - started:.1f}s', flush=True)
    return grouped


def _report_filter(description, before, after, n_tokens, started):
    '''Show type/token losses separately from newly insufficient frequencies.'''
    lost_types = len(before) - len(after)
    before_tokens, after_tokens = sum(before.values()), sum(after.values())
    below_quota = sum(count >= n_tokens and after.get(label, 0) < n_tokens
        for label, count in before.items())
    print(f'{description}: types {len(before):,} -> {len(after):,} '
        f'(lost {lost_types:,}); tokens {before_tokens:,} -> '
        f'{after_tokens:,} (lost {before_tokens - after_tokens:,}); '
        f'types falling below {n_tokens} tokens: {below_quota:,}; '
        f'{monotonic() - started:.1f}s', flush=True)


def _filter_token_groups(grouped, predicate, description, n_tokens):
    '''Apply one token filter with progress over all input tokens.'''
    print(f'{description}: filtering word tokens...', flush=True)
    started = monotonic()
    before = {label: len(tokens) for label, tokens in grouped.items()}
    words = (word for tokens in grouped.values() for word in tokens)
    filtered = defaultdict(list)
    for word in progressbar(words, max_value=sum(before.values())):
        if predicate(word): filtered[word.label].append(word)
    after = {label: len(tokens) for label, tokens in filtered.items()}
    _report_filter(description, before, after, n_tokens, started)
    return filtered


def filter_word_tokens(words, n_tokens=100, iqr_multiplier=5,
    min_duration_ms=100, max_duration_ms=1500):
    '''Filter words from the already selected recording material.
    words:              local word objects; durations are in milliseconds
    n_tokens:           minimum eligible occurrences per orthographic type
    iqr_multiplier:     per-type fence multiplier, applied before frequency
    Print starting counts and type/token losses for each filtering stage.
    '''
    validate_integer(n_tokens, 'n_tokens', minimum=1)
    bounds = (min_duration_ms, max_duration_ms)
    if any(isinstance(value, bool) or not isinstance(value, Real)
        or not isfinite(value) or value < 0 for value in bounds):
        raise ValueError('duration bounds must be finite and nonnegative')
    if min_duration_ms > max_duration_ms:
        raise ValueError('duration bounds must be ordered and nonnegative')
    lexicon_helper.filter_outlier_tokens_from_type([], iqr_multiplier)
    grouped = _group_words(words, 'Starting word material')
    grouped = _filter_token_groups(grouped, lambda word: len(word.phones) >= 2,
        'At least 2 phones', n_tokens)
    grouped = _filter_token_groups(grouped,
        lambda word: lexicon_helper.duration_is_eligible(word.duration,
        min_duration_ms, max_duration_ms),
        f'Duration {min_duration_ms}-{max_duration_ms} ms', n_tokens)
    description = f'Per-type {iqr_multiplier} x IQR'
    print(f'{description}: filtering types...', flush=True)
    started = monotonic()
    before = {label: len(tokens) for label, tokens in grouped.items()}
    inliers = {}
    for label, tokens in progressbar(grouped.items()):
        tokens = lexicon_helper.filter_outlier_tokens_from_type(tokens,
            iqr_multiplier=iqr_multiplier)
        if tokens: inliers[label] = tokens
    after = {label: len(tokens) for label, tokens in inliers.items()}
    _report_filter(description, before, after, n_tokens, started)
    print(f'Frequency >= {n_tokens}: checking types...', flush=True)
    started = monotonic()
    filtered = []
    eligible_counts = {}
    for label in progressbar(sorted(inliers)):
        tokens = inliers[label]
        if len(tokens) >= n_tokens:
            filtered.extend(sorted(tokens, key=lambda word: word.key))
            eligible_counts[label] = len(tokens)
    _report_filter(f'Frequency >= {n_tokens}', after, eligible_counts,
        n_tokens, started)
    return filtered


def select_words(filtered_words, n_tokens=100, random_seed=0):
    '''Sample exactly n_tokens per type in reproducible label/key order.'''
    validate_integer(n_tokens, 'n_tokens', minimum=1)
    validate_integer(random_seed, 'random_seed', minimum=None)
    grouped = _group_words(filtered_words, 'Sampling input')
    rng = random.Random(random_seed)
    selection = []
    print(f'Sampling {n_tokens} tokens per type across '
        f'all {len(grouped):,} eligible types...', flush=True)
    started = monotonic()
    for label in progressbar(sorted(grouped)):
        tokens = sorted(grouped[label], key=lambda word: word.key)
        if len(tokens) < n_tokens:
            raise ValueError(f'type {label!r} has fewer than {n_tokens} tokens')
        selection.extend(rng.sample(tokens, n_tokens))
    before = {label: len(tokens) for label, tokens in grouped.items()}
    after = {label: n_tokens for label in grouped}
    _report_filter('Sampling', before, after, n_tokens, started)
    print(f'Sampled {len(grouped):,} types; {len(selection):,} tokens '
        f'({n_tokens} per type); {monotonic() - started:.1f}s', flush=True)
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
