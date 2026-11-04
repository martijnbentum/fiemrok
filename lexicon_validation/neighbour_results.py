'''Compact cosine retrieval results linked to ordered SVS vector metadata.

Row i always represents vector-index token i. Neighbours exclude only self.
Same-type columns follow the shared mapping and include self. Global ranks
are zero-based over all tokens; equal similarities rank by token index.
'''

from contextlib import contextmanager
from hashlib import sha256
from numbers import Integral
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import h5py
import numpy as np
from speech_vector_search.util_math import l2_normalize_rows

from . import lv_locations


ARRAY_DTYPES = dict(neighbour_indices='uint32',
    neighbour_similarities='float32', same_type_ranks='uint32',
    same_type_similarities='float32')


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _fingerprint(values):
    digest = sha256()
    for value in values:
        value = value.encode('utf-8') if isinstance(value, str) else bytes(value)
        digest.update(len(value).to_bytes(8, 'big'))
        digest.update(value)
    return digest.hexdigest()


def _identity(metadatas, selected_keys=None):
    from echoframe.key_helper import unpack_echoframe_key

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


def _type_mapping(metadatas):
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


@contextmanager
def _atomic_output(filename, overwrite=False):
    filename = Path(filename)
    if filename.exists() and not overwrite: raise FileExistsError(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=filename.parent, prefix=f'.{filename.name}.',
        suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        if overwrite:
            temporary.replace(filename)
        else:
            # Linking publishes a complete file and refuses a concurrent writer.
            os.link(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)


def _shared_mapping(filename, metadatas, identity, create=False):
    expected = _type_mapping(metadatas)
    fingerprints = ('selection_fingerprint', 'label_fingerprint')
    if not Path(filename).exists() and create:
        try:
            with _atomic_output(filename) as temporary:
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, **expected,
                        **{key: identity[key] for key in fingerprints})
        except FileExistsError:
            # Another condition may have published the same mapping first.
            pass
    with np.load(filename, allow_pickle=False) as saved:
        if any(str(saved[key]) != identity[key] for key in fingerprints):
            raise ValueError('shared mapping does not match index identity/order')
        if any(not np.array_equal(saved[key], value)
            for key, value in expected.items()):
            raise ValueError('shared mapping does not match index type grouping')
    return expected


def _cosine_batches(index, batch_size):
    '''Match SVS normalization and backend precision, with bounded query rows.

    FAISS uses float32 inner products; brute_force uses float64. BLAS and
    FAISS may differ in the last floating-point bits. This exhaustive cosine
    implementation defines one consistent ordering for ranks and neighbours.
    Memory is O(batch_size * n_tokens), never O(n_tokens ** 2).
    '''
    backend = index.backend_name
    if backend not in ('faiss', 'brute_force'):
        raise ValueError(f'unsupported SVS backend: {backend}')
    dtype = 'float32' if backend == 'faiss' else 'float64'
    vectors = np.asarray(index.vectors, dtype=dtype)
    if not np.isfinite(vectors).all():
        raise ValueError('vectors must contain finite values')
    for start in range(0, len(vectors), batch_size):
        queries = l2_normalize_rows(index.vectors[start:start + batch_size])
        yield start, np.asarray(queries, dtype=dtype) @ vectors.T


def _write_batches(output, index, mapping, top_k, batch_size):
    started, reported = monotonic(), 0
    n_tokens = len(index.metadatas)
    positions = np.arange(n_tokens, dtype='uint32')
    for start, scores in _cosine_batches(index, batch_size):
        batch = {name: np.empty((len(scores), output[name].shape[1]),
            dtype=dtype) for name, dtype in ARRAY_DTYPES.items()}
        for offset, similarities in enumerate(scores):
            token_index = start + offset
            # Stable sorting retains ascending token index for equal scores.
            order = np.argsort(-similarities, kind='stable')
            neighbours = order[order != token_index][:top_k]
            ranks = np.empty(n_tokens, dtype='uint32')
            ranks[order] = positions
            type_index = mapping['token_types'][token_index]
            same_type = mapping['type_indices'][type_index]
            batch['neighbour_indices'][offset] = neighbours
            batch['neighbour_similarities'][offset] = similarities[
                neighbours]
            batch['same_type_ranks'][offset] = ranks[same_type]
            batch['same_type_similarities'][offset] = similarities[
                same_type]
        done = start + len(scores)
        for name, values in batch.items():
            output[name][start:done] = values
        elapsed = monotonic() - started
        if start == 0 or elapsed - reported >= 10 or done == n_tokens:
            print(f'Neighbours: {done:,}/{n_tokens:,} ({done / n_tokens:.1%}); '
                f'{elapsed:.1f}s elapsed', flush=True)
            reported = elapsed


def write_neighbours(index, filename, mapping_path, selected_keys,
    top_k=30, batch_size=64, overwrite=False, compression='lzf',
    model='hubert', condition='word', workspace=lv_locations.workspace):
    '''Write exact cosine neighbours and all same-type global ranks.

    index:         searched SVS index, in saved selection order
    selected_keys: exact ordered Phraser dataset definition
    mapping_path:  shared type mapping for this dataset
    batch_size:    bounds temporary similarities to this many query rows
    compression:   lossless HDF5 filter, or None
    Returns the published HDF5 path; refuses existing results by default.
    '''
    _positive_integer(top_k, 'top_k')
    _positive_integer(batch_size, 'batch_size')
    filename, mapping_path = Path(filename), Path(mapping_path)
    if filename.exists() and not overwrite: raise FileExistsError(filename)
    lv_locations.store_path(model, condition, 1, workspace)
    n_tokens = len(index.metadatas)
    if n_tokens <= top_k: raise ValueError('index needs more entries than top_k')
    if n_tokens > np.iinfo('uint32').max:
        raise ValueError('token count exceeds uint32 capacity')
    if len(index.vectors) != n_tokens:
        raise ValueError('vector count does not match metadata')
    identity = _identity(index.metadatas, selected_keys)
    mapping = _shared_mapping(mapping_path, index.metadatas, identity,
        create=True)
    n_same_type = mapping['type_indices'].shape[1]
    with _atomic_output(filename, overwrite) as temporary:
        with h5py.File(temporary, 'w') as output:
            output.attrs.update(identity)
            output.attrs.update(format_version=1, model=model,
                condition=condition, index_name=index.metadatas.name,
                index_directory=os.path.relpath(
                    Path(index.metadatas.directory).resolve(),
                    filename.parent.resolve()),
                mapping_path=os.path.relpath(mapping_path.resolve(),
                    filename.parent.resolve()),
                workspace=os.path.relpath(Path(workspace).resolve(),
                    filename.parent.resolve()))
            for name, dtype in ARRAY_DTYPES.items():
                width = top_k if name.startswith('neighbour_') else n_same_type
                output.create_dataset(name, (n_tokens, width), dtype=dtype,
                    chunks=(min(batch_size, n_tokens), width),
                    compression=compression, shuffle=compression is not None)
            _write_batches(output, index, mapping, top_k, batch_size)
    return filename


class NeighbourResults:
    '''Disk-backed query results; row i belongs to vector-index token i.

    index: optional existing SVS index; otherwise load only its metadata
    store: optional caller-owned Echoframe store with local Phraser attached
    Numeric reads need no annotation store. Metadata access opens local stores
    lazily when no store was passed. Close this object or use a with block.
    '''

    def __init__(self, filename, index=None, mapping_path=None, store=None):
        self.filename = Path(filename)
        self._store = store
        self._owns_store = False
        self._file = h5py.File(self.filename, 'r')
        try:
            attrs = self._file.attrs
            if attrs['format_version'] != 1:
                raise ValueError('unsupported neighbour results format')
            if index is None:
                from speech_vector_search.metadata import VectorMetadatas

                path = (self.filename.parent / attrs['index_directory']).resolve()
                self.metadatas = VectorMetadatas.load_json(
                    path / f'{attrs["index_name"]}.json')
            else:
                self.metadatas = index.metadatas
                if self.metadatas.name != attrs['index_name']:
                    raise ValueError('results belong to a different vector index')
            identity = _identity(self.metadatas)
            if any(attrs[key] != value for key, value in identity.items()):
                raise ValueError('results do not match vector index identity/order')
            if mapping_path is None:
                mapping_path = self.filename.parent / attrs['mapping_path']
            self._mapping = _shared_mapping(mapping_path, self.metadatas,
                identity)
            self._validate_arrays()
        except BaseException:
            self._file.close()
            raise

    def _validate_arrays(self):
        n_tokens = len(self.metadatas)
        n_same_type = self._mapping['type_indices'].shape[1]
        neighbour_shape = self._file['neighbour_indices'].shape
        if (len(neighbour_shape) != 2 or neighbour_shape[0] != n_tokens
            or not 0 < neighbour_shape[1] < n_tokens):
            raise ValueError('invalid neighbour array shape')
        for name, dtype in ARRAY_DTYPES.items():
            expected = (neighbour_shape if name.startswith('neighbour_')
                else (n_tokens, n_same_type))
            array = self._file[name]
            if array.shape != expected or array.dtype != np.dtype(dtype):
                raise ValueError(f'invalid {name} shape or dtype')

    def __len__(self):
        return len(self.metadatas)

    def _token_index(self, token_index):
        if (isinstance(token_index, bool) or not isinstance(token_index, Integral)
            or not 0 <= token_index < len(self)):
            raise IndexError('token_index outside vector index')
        if not self._file.id.valid: raise ValueError('results are closed')
        return int(token_index)

    def __getitem__(self, token_index):
        return QueryResult(self, self._token_index(token_index))

    def read(self, start, stop):
        '''Read a contiguous query batch without opening annotation stores.'''
        self._token_index(start)
        if (isinstance(stop, bool) or not isinstance(stop, Integral)
            or not start <= stop <= len(self)):
            raise IndexError('invalid query batch bounds')
        return {name: self._file[name][start:stop] for name in ARRAY_DTYPES}

    def _annotation_store(self):
        if not self._file.id.valid: raise ValueError('results are closed')
        if self._store is None:
            from . import lv_stores

            attrs = self._file.attrs
            self._store = lv_stores.load_echoframe_store(attrs['model'],
                attrs['condition'], workspace=self.filename.parent /
                attrs['workspace'])
            self._owns_store = True
        return self._store

    def source_key(self, token_index):
        token_index = self._token_index(token_index)
        return self.metadatas[token_index].source_echoframe_keys[0]

    def metadata(self, token_index):
        '''Load one Echoframe metadata object, including Phraser access.'''
        key = self.source_key(token_index)
        metadata = self._annotation_store().load_metadata(key)
        if metadata is None: raise ValueError('Echoframe metadata not found')
        return metadata

    def context(self, token_index, n_phone_context=10, n_word_context=3):
        '''Build phone/word context on demand using the existing helpers.'''
        from .neighbour_metadata import echoframe_key_to_metadata

        return echoframe_key_to_metadata(self.source_key(token_index),
            self._annotation_store(), n_phone_context, n_word_context)

    def close(self):
        try:
            if self._owns_store:
                self._owns_store = False
                try:
                    self._store.close()
                finally:
                    self._store.close_phraser_stores()
                    self._store = None
        finally:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class QueryResult:
    '''One query view; same-type columns follow same_type_indices, not rank.'''

    def __init__(self, results, token_index):
        self.results = results
        self.token_index = token_index

    @property
    def label(self):
        return self.results.metadatas[self.token_index].label

    def _array(self, name):
        self.results._token_index(self.token_index)
        return self.results._file[name][self.token_index]

    @property
    def neighbour_indices(self):
        return self._array('neighbour_indices')

    @property
    def neighbour_similarities(self):
        return self._array('neighbour_similarities')

    @property
    def same_type_indices(self):
        self.results._token_index(self.token_index)
        mapping = self.results._mapping
        return mapping['type_indices'][mapping['token_types'][
            self.token_index]].copy()

    @property
    def same_type_ranks(self):
        return self._array('same_type_ranks')

    @property
    def same_type_similarities(self):
        return self._array('same_type_similarities')

    @property
    def metadata(self):
        return self.results.metadata(self.token_index)

    def precision_at_k(self, top_k=10):
        _positive_integer(top_k, 'top_k')
        neighbours = self.neighbour_indices
        if top_k > len(neighbours):
            raise ValueError('top_k exceeds saved neighbour count')
        return sum(self.results.metadatas[int(i)].label == self.label
            for i in neighbours[:top_k]) / top_k
