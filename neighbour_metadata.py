'''Join neighbour source keys to annotations without changing retrieval rows.'''

from collections import Counter
import json
from pathlib import Path
import pickle


def load_or_build_metadata(dataset='word', overwrite=False,
    n_phone_context=10, n_word_context=3, directory='svs_data'):
    '''Load cached metadata or build it from the shared Phraser key file.

    dataset:    word or cross_boundary; shared across models and pooling methods
    overwrite:  rebuild the JSON when True
    directory:  directory containing the key files and output JSON
    '''
    directory = Path(directory)
    filename = directory / (f'{dataset}_metadata_'
        f'phones{n_phone_context}_words{n_word_context}.json')
    if filename.exists() and not overwrite:
        with filename.open() as handle: return json.load(handle)

    key_files = {'word': 'cgn_lexicon_phraser_keys.pkl',
        'cross_boundary': 'cross_boundary_marker_keys.pkl'}
    with (directory / key_files[dataset]).open('rb') as handle:
        phraser_keys = pickle.load(handle)

    import stores

    store = stores.load_phraser_cgn_store()
    result = build_metadata_dicts(phraser_keys, store,
        n_phone_context, n_word_context)
    result['contributions'] = metadata_contributions(result['key_dict'])
    result['contributions']['n_failed'] = len(result['errors'])
    with filename.open('w') as handle:
        json.dump(result, handle, ensure_ascii=False)
    return result


def build_metadata_dicts(phraser_keys, store, n_phone_context=10,
    n_word_context=3):
    '''Build dictionaries keyed by Phraser key and a separate failed-join report.

    phraser_keys: unique or repeated annotation keys from the Echoframe store
    store:        CGN Phraser store used to load the annotations
    Source gender/age values are preserved, including unknown-value conventions.
    '''
    _validate_context_sizes(n_phone_context, n_word_context)
    result = {name: {} for name in
        ('key_dict', 'speaker_dict', 'phone_dict', 'word_dict', 'errors')}
    cache = {}
    for key in sorted(set(_hex(key) for key in phraser_keys)):
        try:
            segment = store.load(bytes.fromhex(key))
            record = _segment_metadata(segment, n_phone_context,
                n_word_context, cache)
        except (ValueError, KeyError) as error:
            result['errors'][key] = str(error)
            continue
        keys = record['key']
        result['key_dict'][key] = keys
        result['speaker_dict'][keys['speaker_key']] = record['speaker']
        result['phone_dict'][key] = record['phone']
        result['word_dict'][key] = record['word']
    return result


def store_phraser_keys(store, model_name):
    '''Enumerate unique Phraser keys from layer 9 hidden-state index keys.'''
    from echoframe import key_helper

    model_id = store.model_registry.load_model_id(model_name)
    keys = set()
    for key in store.index.all_echoframe_keys:
        fields = key_helper.unpack_echoframe_key(key)
        if fields.get('output_type') != 'hidden_state': continue
        if fields.get('layer') != 9: continue
        if fields.get('model_id') != model_id: continue
        keys.add(fields['phraser_key'])
    if not keys: raise ValueError('no layer 9 tokens found for selected model')
    return sorted(keys)


def echoframe_key_to_phraser_key(echoframe_key):
    '''Decode a source key without opening a store; return a Phraser hex key.'''
    from echoframe import key_helper

    if isinstance(echoframe_key, str):
        echoframe_key = bytes.fromhex(echoframe_key)
    fields = key_helper.unpack_echoframe_key(echoframe_key)
    key = fields.get('phraser_key')
    if key is None: raise ValueError('Echoframe key has no Phraser key')
    return key.hex()


def echoframe_key_to_metadata(echoframe_key, store, n_phone_context=10,
    n_word_context=3, context_cache=None):
    '''Return key, speaker, phone, and word records for one source key.

    store:          Echoframe store with CGN Phraser attached; any model/layer
    context_cache:  optional shared recording/speaker annotation cache
    Context crosses phrases, stays with the speaker, and stops at recording end.
    boundary_index is the first right-word phone/word index, or None for words.
    Lookup failures raise ValueError; '*' is reserved for context padding.
    '''
    _validate_context_sizes(n_phone_context, n_word_context)
    metadata = store.load_metadata(echoframe_key)
    if metadata is None: raise ValueError('Echoframe metadata not found')
    segment = metadata.phraser_object
    if context_cache is None: context_cache = {}
    return _segment_metadata(segment, n_phone_context, n_word_context,
        context_cache)


def metadata_contributions(key_dict):
    '''Count speaker/recording contributions once per unique Phraser token.'''
    speakers = Counter(item['speaker_key'] for item in key_dict.values())
    recordings = Counter(item['recording_key'] for item in key_dict.values())
    return dict(n_tokens=len(key_dict), speaker_counts=dict(speakers),
        recording_counts=dict(recordings))


def query_distributions(filename, key_dict):
    '''Yield existing neighbour distributions, joining through Phraser keys.'''
    with open(filename) as handle:
        for line in handle:
            row = json.loads(line)
            query = _token_record(row, key_dict)
            neighbours = [_token_record(item, key_dict)
                for item in row['neighbours']]
            speakers = Counter(item['speaker_key'] for item in neighbours)
            recordings = Counter(item['recording_key'] for item in neighbours)
            yield dict(source_keys_hex=row['source_keys_hex'],
                phraser_key=echoframe_key_to_phraser_key(
                    row['source_keys_hex'][0]), label=row['label'], query=query,
                n_neighbours=len(neighbours), speaker_counts=dict(speakers),
                recording_counts=dict(recordings),
                same_speaker=speakers[query['speaker_key']],
                same_recording=recordings[query['recording_key']])


def _segment_metadata(segment, n_phone_context, n_word_context, context_cache):
    if segment is None: raise ValueError('Phraser object not found')
    if segment.audio is None or segment.speaker is None:
        raise ValueError('recording or speaker missing')
    if segment.object_type not in ('Word', 'Marker'):
        raise ValueError('expected a Word or cross-boundary Marker')
    if context_cache is None: context_cache = {}
    phones, words = _recording_items(segment, context_cache)
    if segment.object_type == 'Word':
        targets = sorted(segment.phones, key=lambda p: (p.start, p.end, p.key))
        target_words = [segment]
    else:
        targets = [p for p in phones
            if p.start < segment.end and p.end > segment.start]
        word_keys = dict.fromkeys(p.word.key for p in targets)
        target_words = [word for word in words if word.key in word_keys]
        if len(target_words) != 2:
            raise ValueError('cross-boundary marker must span two words')
        if any(p.start < segment.start or p.end > segment.end for p in targets):
            raise ValueError('marker boundaries cut through a phone')
    if not targets: raise ValueError('target has no phones')
    phone_record = _context(phones, targets, n_phone_context)
    word_record = _context(words, target_words, n_word_context)
    boundary = None
    if len(target_words) == 2:
        boundary = next(i for i, phone in enumerate(targets)
            if phone.word.key == target_words[1].key)
    phone_record['boundary_index'] = boundary
    word_record['boundary_index'] = 1 if boundary is not None else None
    speaker = segment.speaker
    key_record = dict(speaker_key=_hex(segment.speaker_key),
        recording_key=_hex(segment.audio_key))
    gender = speaker.gender
    if callable(gender): gender = gender()
    speaker_record = dict(gender=gender, age=speaker.age)
    return dict(key=key_record, speaker=speaker_record,
        phone=phone_record, word=word_record)


def _hex(key):
    return key.hex() if isinstance(key, bytes) else key


def _validate_context_sizes(*sizes):
    if any(type(size) is not int or size < 0 for size in sizes):
        raise ValueError('context sizes must be non-negative integers')


def _context(items, targets, size):
    '''Select context by occurrence key, keeping all labels in time order.'''
    positions = {item.key: i for i, item in enumerate(items)}
    indices = [positions[item.key] for item in targets]
    first, last = min(indices), max(indices)
    if indices != list(range(first, last + 1)):
        raise ValueError('target annotations are not contiguous in recording')
    left = [item.label for item in items[max(0, first - size):first]]
    right = [item.label for item in items[last + 1:last + 1 + size]]
    return dict(target=[item.label for item in targets],
        left_context=['*'] * (size - len(left)) + left,
        right_context=right + ['*'] * (size - len(right)))


def _recording_items(segment, cache):
    '''Cache one speaker's complete annotation stream within a recording.'''
    cache_key = (segment.audio_key, segment.speaker_key)
    if cache_key not in cache:
        phones, words = [], []
        for phrase in segment.audio.phrases:
            if phrase.speaker_id != segment.speaker_id: continue
            phones.extend(phrase.phones)
            words.extend(phrase.words)
        order = lambda item: (item.start, item.end, item.key)
        cache[cache_key] = (sorted(phones, key=order), sorted(words, key=order))
    return cache[cache_key]


def _token_record(row, key_dict):
    keys = row['source_keys_hex']
    if len(keys) != 1:
        raise ValueError('inspection requires one source key per vector')
    return key_dict[echoframe_key_to_phraser_key(keys[0])]
