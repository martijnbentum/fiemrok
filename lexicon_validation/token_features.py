'''Access selected validation tokens in vector-index order for analysis.

The feature table is a dict of NumPy arrays; row i describes token i.
'''

from pathlib import Path
import pickle
import re

import numpy as np
from progressbar import progressbar

from . import lv_locations
from . import lv_stores
from .neighbour_helper import fingerprint


_items = {}
COLUMNS = ('label', 'phraser_key', 'speaker', 'gender', 'age', 'recording',
    'component', 'start_ms', 'end_ms', 'duration_ms', 'phones', 'n_phones',
    'n_syllables', 'phrase_key', 'n_phrase_words', 'word_position',
    'phrase_initial', 'phrase_final', 'left_word', 'right_word',
    'boundary_index')


def load_selected_items(condition, store=None,
    workspace=lv_locations.workspace):
    '''Return saved Phraser keys and items in vector-index order.

    condition:  word or cross_boundary
    store:      local Phraser store; defaults to the shared lv_stores handle
    workspace:  validation workspace containing the selection PKLs
    Items are cached per selection file and reused while the same open store
    and saved keys apply. lv_stores.close_stores() clears the cache.
    '''
    path = lv_locations.selection_path(condition, workspace).resolve()
    with path.open('rb') as handle: keys = tuple(pickle.load(handle))
    if not keys: raise ValueError(f'empty {condition} selection')
    if store is None: store = lv_stores.load_phraser_store(workspace=workspace)
    entry = _items.get(path)
    if entry is None or not _entry_matches(entry, store, keys):
        items = tuple(store.load_many(list(keys)))
        if len(items) != len(keys) or any(item is None for item in items):
            raise ValueError(f'missing selected {condition} items')
        entry = dict(store=store, keys=keys, items=items)
        _items[path] = entry
    return entry['keys'], entry['items']


def clear_cache():
    '''Forget cached selection items; stores are left open.'''
    _items.clear()


def _entry_matches(entry, store, keys):
    '''Check that a cached entry uses this open store and selection.'''
    return entry['store'] is store and store.is_open() and entry['keys'] == keys


def load_token_features(condition, store=None, overwrite=False,
    workspace=lv_locations.workspace, selection_fingerprint=None):
    '''Load the saved feature table, building and saving it when missing.

    condition:              word or cross_boundary
    store:                  local Phraser store; defaults to lv_stores
    overwrite:              rebuild and replace an existing table
    workspace:              validation workspace with selections and tables
    selection_fingerprint:  optional expected fingerprint, e.g. from results
    Returns a dict of NumPy arrays keyed by COLUMNS, in vector-index order.
    Age 0 is CGN's unknown value; word rows use '' and -1 for the
    cross-boundary columns.
    '''
    path = lv_locations.token_features_path(condition, workspace)
    selection = lv_locations.selection_path(condition, workspace)
    with selection.open('rb') as handle: keys = pickle.load(handle)
    expected = fingerprint(keys)
    if selection_fingerprint not in (None, expected):
        raise ValueError('saved selection does not match results')
    if path.exists() and not overwrite:
        features = _load_npz(path)
        if features.pop('selection_fingerprint') != expected:
            message = f'{path} does not match the saved selection; rebuild it'
            raise ValueError(message)
        return features
    _, items = load_selected_items(condition, store, workspace)
    features = build_token_features(items)
    _save_npz(path, features, expected)
    return features


def build_token_features(items):
    '''Build the feature table for Phraser words or markers in row order.'''
    speakers = {}
    rows = [_item_features(item, speakers) for item in progressbar(items)]
    return {name: np.asarray([row[name] for row in rows])
        for name in COLUMNS}


def _item_features(item, speakers):
    '''Collect one token's features as a dict of plain values.'''
    phones, words = _target_phones_and_words(item)
    phrase_words = sorted(item.phrase.words, key=_time_order)
    phrase_keys = [word.key for word in phrase_words]
    first = phrase_keys.index(words[0].key)
    last = phrase_keys.index(words[-1].key)
    speaker, gender, age = _speaker(item, speakers)
    filename = item.audio.filename
    features = dict(label=item.label, phraser_key=item.key, speaker=speaker,
        gender=gender, age=age, recording=Path(filename).stem,
        component=_component(filename), start_ms=item.start,
        end_ms=item.end, duration_ms=item.duration,
        phones=' '.join(phone.label for phone in phones),
        n_phones=len(phones),
        n_syllables=len({phone.parent_key for phone in phones}),
        phrase_key=item.phrase.key, n_phrase_words=len(phrase_words),
        word_position=first, phrase_initial=first == 0,
        phrase_final=last == len(phrase_words) - 1, left_word='',
        right_word='', boundary_index=-1)
    if item.object_type == 'Marker':
        right = words[1]
        features.update(left_word=words[0].label, right_word=right.label,
            boundary_index=sum(phone.start < right.start for phone in phones))
    return features


def _target_phones_and_words(item):
    '''Return a word's phones, or a marker's phones and two words.'''
    if item.object_type == 'Word':
        return sorted(item.phones, key=_time_order), [item]
    if item.object_type != 'Marker':
        raise ValueError('expected a Word or cross-boundary Marker')
    words = sorted(item.overlap_word, key=_time_order)
    if len(words) != 2:
        raise ValueError('cross-boundary marker must span two words')
    phones = sorted(item.overlap_phone, key=_time_order)
    if any(p.start < item.start or p.end > item.end for p in phones):
        raise ValueError('marker boundaries cut through a phone')
    return phones, words


def _speaker(item, speakers):
    '''Return speaker name, gender and age, cached per speaker key.'''
    key = item.speaker_key
    if key not in speakers:
        speaker = item.speaker
        if speaker is None: raise ValueError('speaker missing')
        gender = speaker.gender
        if callable(gender): gender = gender()
        speakers[key] = (speaker.name, gender, speaker.age)
    return speakers[key]


def _component(filename):
    '''Return the CGN component letter from a comp-x audio path.'''
    match = re.search(r'comp-([a-o])', str(filename))
    if match is None:
        raise ValueError(f'no CGN component in audio path {filename}')
    return match.group(1)


def _time_order(item):
    return (item.start, item.end, item.key)


def _save_npz(path, features, selection_fingerprint):
    '''Write the table via a temporary file so failures leave no output.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez(temporary, selection_fingerprint=selection_fingerprint,
        **features)
    temporary.replace(path)


def _load_npz(path):
    '''Load all saved columns into memory without pickled objects.'''
    with np.load(path, allow_pickle=False) as saved:
        features = {name: saved[name] for name in COLUMNS}
        features['selection_fingerprint'] = str(saved['selection_fingerprint'])
    return features
