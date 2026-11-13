'''Select the word dataset for lexicon validation.'''

from pathlib import Path
import pickle
from time import monotonic

import phraser

from .cross_boundary_sampling import validate_integer


def select_words(store, n_tokens=100, random_seed=0, min_duration_ms=100,
    max_duration_ms=1500):
    '''Return ordered keys of exactly n_tokens words per eligible type.
    store:            open local Phraser material store
    n_tokens:         minimum eligible occurrences and sample size per type
    random_seed:      local seed for the per-type draw
    min_duration_ms:  inclusive absolute duration minimum
    max_duration_ms:  inclusive absolute duration maximum

    Words need at least 2 phones, no recorded timing concern, and the
    store's within_range duration flag (per-type fences assessed over the
    whole source store). Keys follow label order, then key order. Print
    Phraser's report of token and type losses per filtering stage.
    '''
    validate_integer(n_tokens, 'n_tokens', minimum=1)
    print('Building the word lexicon from local materials...', flush=True)
    started = monotonic()
    lexicon = phraser.build_word_lexicon(store=store,
        min_tokens_per_type=n_tokens, max_retained_tokens_per_type=n_tokens,
        random_seed=random_seed, min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms, min_phones=2,
        exclude_timing_uncertain=True)
    print(lexicon.report, flush=True)
    keys = []
    for entry in lexicon.values(): keys.extend(entry.keys)
    print(f'Sampled {len(lexicon):,} types; {len(keys):,} tokens '
        f'({n_tokens} per type); {monotonic() - started:.1f}s', flush=True)
    return keys


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
