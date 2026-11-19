'''Corpus frequencies from the local Phraser store as per-token columns.

Counts cover everything in the local store without eligibility filters:
words, syllables and phones by label, and cross-boundary phoneme sequences
as enumerated by the cross-boundary sampler. Columns use log10(count + 1).
'''

from pathlib import Path
import pickle
import tempfile

import numpy as np
import phraser
from phraser import Lexicon

from . import cross_boundary_lexicon as cbl
from . import lv_locations
from . import lv_stores


LEXICONS = {'word': phraser.build_word_lexicon,
    'syllable': phraser.build_syllable_lexicon,
    'phone': phraser.build_phone_lexicon}
UNFILTERED = dict(max_retained_tokens_per_type=0, min_duration_ms=None,
    max_duration_ms=None, exclude_duration_outliers=False,
    exclude_timing_uncertain=False, exclude_overlap=False)
NAMES = (*LEXICONS, 'cross_boundary_sequence')
SYLLABLE_SEPARATOR = ' | '


def build_frequencies(store=None, overwrite=False,
    workspace=lv_locations.workspace, verbose=True):
    '''Count units in the local store and save one file per unit.

    store:      local Phraser store; defaults to the shared lv_stores handle
    overwrite:  rebuild existing files; otherwise they are kept
    workspace:  validation workspace; files go to its frequencies/ folder
    Word, syllable and phone files are count-only Phraser Lexicons with
    their eligibility reports; the sequence file is a plain dict.
    Returns the paths keyed by unit name.
    '''
    if store is None: store = lv_stores.load_phraser_store(workspace=workspace)
    paths = {name: lv_locations.frequency_path(name, workspace)
        for name in NAMES}
    for name, build in LEXICONS.items():
        if paths[name].exists() and not overwrite: continue
        lexicon = build(store=store, verbose=verbose, **UNFILTERED)
        lexicon.save(paths[name])
    path = paths['cross_boundary_sequence']
    if overwrite or not path.exists():
        _save_pickle(path, count_cross_boundary_sequences(store))
    return paths


def count_cross_boundary_sequences(store):
    '''Count boundary-crossing phoneme sequences in all multi-word phrases.'''
    phrases = cbl.filter_phrases_for_cross_boundary_lexicon(
        store.phrases.all())
    phrases = sorted(phrases, key=lambda phrase: phrase.key)
    return dict(cbl.CrossBoundaryLexicon(phrases).sequence_counts)


def load_frequencies(workspace=lv_locations.workspace):
    '''Return {unit name: {label: count}} from the saved frequency files.'''
    frequencies = {}
    for name in NAMES:
        path = lv_locations.frequency_path(name, workspace)
        if not path.exists():
            raise FileNotFoundError(f'run build_frequencies first: {path}')
        if name in LEXICONS: frequencies[name] = Lexicon.load(path).counts
        else: frequencies[name] = _load_pickle(path)
    return frequencies


def frequency_columns(features, frequencies):
    '''Return per-token log10(count + 1) frequency arrays in row order.

    features:     table from token_features.load_token_features
    frequencies:  counts from load_frequencies
    Word rows fill word_log_freq; cross-boundary rows fill the left/right
    word and sequence columns; the others are NaN. Syllable and phone
    columns summarise the token's units.
    '''
    cross = features['boundary_index'] > 0
    words = frequencies['word']
    sequences = frequencies['cross_boundary_sequence']
    syllables = [row.split(SYLLABLE_SEPARATOR) for row in features['syllables']]
    syllable_freqs = [_log_counts(row, frequencies['syllable'])
        for row in syllables]
    phone_freqs = [_log_counts(row.split(), frequencies['phone'])
        for row in features['phones']]
    return dict(
        word_log_freq=_where(~cross, features['label'], words),
        left_word_log_freq=_where(cross, features['left_word'], words),
        right_word_log_freq=_where(cross, features['right_word'], words),
        sequence_log_freq=_where(cross, features['label'], sequences),
        syllable_log_freq_mean=np.array([row.mean() for row in syllable_freqs]),
        syllable_log_freq_min=np.array([row.min() for row in syllable_freqs]),
        phone_log_freq_mean=np.array([row.mean() for row in phone_freqs]))


def _log_counts(labels, counts):
    return np.log10(np.array([counts.get(label, 0) for label in labels]) + 1)


def _where(mask, labels, counts):
    '''Return log counts for rows in mask and NaN elsewhere.'''
    return np.where(mask, _log_counts(labels, counts), np.nan)


def _load_pickle(path):
    with Path(path).open('rb') as handle: return pickle.load(handle)


def _save_pickle(path, value):
    '''Write a pickle via a temporary file so failures leave no output.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
        delete=False) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    Path(handle.name).replace(path)
