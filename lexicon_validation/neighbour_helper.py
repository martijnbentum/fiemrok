'''Shared neighbour-result format, token identity and type mapping helpers.'''

from hashlib import sha256
from numbers import Integral

from echoframe.key_helper import unpack_echoframe_key
import numpy as np


INVALID_INDEX = np.iinfo('uint32').max
STATUSES = {None: 0, 'outside_recording': 1, 'outside_model_outputs': 2}


ARRAY_DTYPES = dict(neighbour_indices='uint32',
    neighbour_similarities='float32', same_type_ranks='uint32',
    same_type_similarities='float32')


def validate_positive_integer(value, name):
    '''Require a positive integer, excluding booleans.'''
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _fingerprint(values):
    digest = sha256()
    for value in values:
        if isinstance(value, str): value = value.encode('utf-8')
        else: value = bytes(value)
        digest.update(len(value).to_bytes(8, 'big'))
        digest.update(value)
    return digest.hexdigest()


def index_identity(metadatas, selected_keys=None):
    '''Fingerprint ordered source keys and labels; optionally verify selection.'''
    source_keys, phraser_keys, labels = [], [], []
    for metadata in metadatas:
        if len(metadata.source_echoframe_keys) != 1:
            raise ValueError('expected one Echoframe source key per vector')
        key = bytes(metadata.source_echoframe_keys[0])
        phraser_key = unpack_echoframe_key(key).get('phraser_key')
        if phraser_key is None: raise ValueError('source has no Phraser key')
        source_keys.append(key)
        phraser_keys.append(phraser_key)
        labels.append(metadata.label)
    if not phraser_keys: raise ValueError('empty vector index')
    if len(set(phraser_keys)) != len(phraser_keys):
        raise ValueError('duplicate Phraser token identities')
    if selected_keys is not None and list(selected_keys) != phraser_keys:
        raise ValueError('vector index does not match saved selection order')
    return dict(selection_fingerprint=_fingerprint(phraser_keys),
        source_fingerprint=_fingerprint(source_keys),
        label_fingerprint=_fingerprint(labels))


def type_mapping(metadatas):
    '''Group token indices by label, requiring the same quota for every type.'''
    groups = {}
    token_types = np.empty(len(metadatas), dtype='uint32')
    for token_index, metadata in enumerate(metadatas):
        if metadata.label not in groups:
            groups[metadata.label] = (len(groups), [])
        type_index, indices = groups[metadata.label]
        token_types[token_index] = type_index
        indices.append(token_index)
    counts = {len(indices) for _, indices in groups.values()}
    if len(counts) != 1:
        raise ValueError('all types must have the same token quota')
    type_indices = np.asarray([indices for _, indices in groups.values()],
        dtype='uint32')
    return dict(token_types=token_types, type_indices=type_indices)


def load_shared_mapping(filename, metadatas, identity):
    '''Check a saved mapping against token identity and return its grouping.'''
    expected = type_mapping(metadatas)
    fingerprints = ('selection_fingerprint', 'label_fingerprint')
    with np.load(filename, allow_pickle=False) as saved:
        if any(str(saved[key]) != identity[key] for key in fingerprints):
            message = 'shared mapping does not match index identity/order'
            raise ValueError(message)
        if any(not np.array_equal(saved[key], value)
            for key, value in expected.items()):
            message = 'shared mapping does not match index type grouping'
            raise ValueError(message)
    return expected
