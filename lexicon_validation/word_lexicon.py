'''Select the word dataset for lexicon validation.'''

from pathlib import Path
import pickle
from time import monotonic

import phraser

from .cross_boundary_sampling import validate_integer


def select_words(store, n_tokens=100, min_duration_ms=100,
    max_duration_ms=1500, verbose=True, exclude_overlap=True):
    '''Build an eligible word inventory; sample it separately with .sample().
    store:            open local Phraser material store
    n_tokens:         minimum eligible occurrences per type
    min_duration_ms:  inclusive absolute duration minimum
    max_duration_ms:  inclusive absolute duration maximum
    verbose:          print loading, scan progress and the final report
    exclude_overlap:  exclude words overlapping another speaker's speech

    Words need at least 2 phones, no recorded timing concern, and the
    store's within_range duration flag (per-type fences assessed over the
    whole source store). Token keys follow label order, then key order. Print
    Phraser's report of token and type losses per filtering stage.
    '''
    validate_integer(n_tokens, 'n_tokens', minimum=1)
    if verbose:
        print('Building the word lexicon from local materials...', flush=True)
    started = monotonic()
    lexicon = phraser.build_word_lexicon(store=store,
        min_tokens_per_type=n_tokens, max_retained_tokens_per_type=None,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms, min_phones=2,
        exclude_duration_outliers=True, exclude_timing_uncertain=True,
        exclude_overlap=exclude_overlap, verbose=verbose)
    if verbose:
        print(lexicon.report, flush=True)
        print(f'Eligible inventory: {len(lexicon):,} types; '
            f'{lexicon.n_tokens:,} tokens; {monotonic() - started:.1f}s',
            flush=True)
    return lexicon


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
