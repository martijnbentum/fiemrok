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
from .neighbour_metadata import context_labels


_items = {}
RATE_WINDOW_MS = 30_000
COLUMNS = ('label', 'phraser_key', 'speaker', 'gender', 'age', 'recording',
    'component', 'start_ms', 'end_ms', 'duration_ms', 'phones', 'n_phones',
    'syllables', 'n_syllables', 'phrase_key', 'n_phrase_words', 'word_position',
    'phrase_initial', 'phrase_final', 'left_word', 'right_word',
    'boundary_index', 'left_words', 'right_words', 'articulation_rate',
    'rate_speech_s')


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
    workspace=lv_locations.workspace, selection_fingerprint=None,
    n_context_words=3):
    '''Load the saved feature table, building and saving it when missing.

    condition:              word or cross_boundary
    store:                  local Phraser store; defaults to lv_stores
    overwrite:              rebuild and replace an existing table
    workspace:              validation workspace with selections and tables
    selection_fingerprint:  optional expected fingerprint, e.g. from results
    n_context_words:        words per side in left_words and right_words
    Returns a dict of NumPy arrays keyed by COLUMNS, in vector-index order.
    Age 0 is CGN's unknown value; word rows use '' and -1 for the
    cross-boundary columns. syllables joins the labels of the syllables
    linked to the token's phones with ' | ', including partly covered
    syllables of cross-boundary items. articulation_rate is phones per
    second over up to RATE_WINDOW_MS of the speaker's own phones on each
    side, excluding the token; rate_speech_s is the speech used, NaN rate
    when none.
    '''
    path = lv_locations.token_features_path(condition, workspace)
    selection = lv_locations.selection_path(condition, workspace)
    with selection.open('rb') as handle: keys = pickle.load(handle)
    expected = fingerprint(keys)
    if selection_fingerprint not in (None, expected):
        raise ValueError('saved selection does not match results')
    if path.exists() and not overwrite:
        features = _load_npz(path)
        saved = (features.pop('selection_fingerprint'),
            features.pop('n_context_words'))
        if saved != (expected, n_context_words):
            message = f'{path} does not match the selection or context size; '
            raise ValueError(message + 'rebuild it with overwrite=True')
        return features
    _, items = load_selected_items(condition, store, workspace)
    features = build_token_features(items, n_context_words)
    _save_npz(path, features, expected, n_context_words)
    return features


def build_token_features(items, n_context_words=3):
    '''Build the feature table for Phraser words or markers in row order.

    Context words come from the same speaker's words in the recording, cross
    phrase boundaries, and are padded with '*' at the recording edges.
    Items are processed per recording and speaker, so each word and phone
    stream is loaded once and only one is held in memory.
    '''
    if type(n_context_words) is not int or n_context_words < 0:
        raise ValueError('n_context_words must be a non-negative integer')
    order = sorted(range(len(items)), key=lambda i: _stream_key(items[i]))
    speakers, streams, rows = {}, {}, [None] * len(items)
    for index in progressbar(order):
        item = items[index]
        if _stream_key(item) not in streams: streams.clear()
        rows[index] = _item_features(item, speakers, streams, n_context_words)
    return {name: np.asarray([row[name] for row in rows])
        for name in COLUMNS}


def _item_features(item, speakers, streams, n_context_words):
    '''Collect one token's features as a dict of plain values.'''
    phones, words = _target_phones_and_words(item)
    phrase_words = sorted(item.phrase.words, key=_time_order)
    phrase_keys = [word.key for word in phrase_words]
    first = phrase_keys.index(words[0].key)
    last = phrase_keys.index(words[-1].key)
    syllables = {phone.parent_key: phone.parent for phone in phones}
    syllables = sorted(syllables.values(), key=_time_order)
    speaker, gender, age = _speaker(item, speakers)
    filename = item.audio.filename
    features = dict(label=item.label, phraser_key=item.key, speaker=speaker,
        gender=gender, age=age, recording=Path(filename).stem,
        component=_component(filename), start_ms=item.start,
        end_ms=item.end, duration_ms=item.duration,
        phones=' '.join(phone.label for phone in phones),
        n_phones=len(phones),
        syllables=' | '.join(syllable.label for syllable in syllables),
        n_syllables=len(syllables),
        phrase_key=item.phrase.key, n_phrase_words=len(phrase_words),
        word_position=first, phrase_initial=first == 0,
        phrase_final=last == len(phrase_words) - 1, left_word='',
        right_word='', boundary_index=-1)
    stream = _speaker_stream(item, streams)
    context = context_labels(stream['words'], words, n_context_words)
    rate, speech_ms = _articulation_rate(stream, item)
    features.update(left_words=' '.join(context['left_context']),
        right_words=' '.join(context['right_context']),
        articulation_rate=rate, rate_speech_s=speech_ms / 1000)
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


def _stream_key(item):
    return (item.audio_key, item.speaker_key)


def _speaker_stream(item, streams):
    '''Return one speaker's words and phone times in a recording, cached.'''
    key = _stream_key(item)
    if key not in streams:
        phrases = [phrase for phrase in item.audio.phrases
            if phrase.speaker_id == item.speaker_id]
        words = [word for phrase in phrases for word in phrase.words]
        times = sorted((phone.start, phone.end) for phrase in phrases
            for phone in phrase.phones)
        times = np.asarray(times, dtype=float).reshape(-1, 2)
        streams[key] = dict(words=sorted(words, key=_time_order),
            starts=times[:, 0], ends=times[:, 1])
    return streams[key]


def _articulation_rate(stream, item):
    '''Return phones per second and speech ms around, but excluding, item.

    Each side takes the nearest phones until RATE_WINDOW_MS is reached,
    including the phone that crosses it.
    '''
    starts, ends = stream['starts'], stream['ends']
    before = np.flatnonzero(ends <= item.start)[::-1]
    after = np.flatnonzero(starts >= item.end)
    used = [_window(side, ends - starts) for side in (before, after)]
    durations = (ends - starts)[np.concatenate(used)]
    speech_ms = float(durations.sum())
    if speech_ms == 0: return np.nan, 0.
    return len(durations) / speech_ms * 1000, speech_ms


def _window(indices, durations):
    '''Return the leading indices whose durations reach the window size.'''
    cumulative = np.cumsum(durations[indices])
    return indices[:np.searchsorted(cumulative, RATE_WINDOW_MS) + 1]


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


def _save_npz(path, features, selection_fingerprint, n_context_words):
    '''Write the table via a temporary file so failures leave no output.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez(temporary, selection_fingerprint=selection_fingerprint,
        n_context_words=n_context_words, **features)
    temporary.replace(path)


def _load_npz(path):
    '''Load all saved columns into memory without pickled objects.'''
    with np.load(path, allow_pickle=False) as saved:
        missing = [name for name in COLUMNS if name not in saved]
        if missing:
            message = f'{path} lacks columns {missing}; '
            raise ValueError(message + 'rebuild it with overwrite=True')
        features = {name: saved[name] for name in COLUMNS}
        features['selection_fingerprint'] = str(saved['selection_fingerprint'])
        features['n_context_words'] = int(saved['n_context_words'])
    return features
