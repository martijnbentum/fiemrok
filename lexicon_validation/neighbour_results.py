'''Compact cosine retrieval results linked to ordered SVS vector metadata.

Row i always represents vector-index token i. Neighbours exclude only self.
Same-type columns follow the shared mapping and include self. Global ranks
are zero-based over all tokens; equal similarities rank by token index.
'''

from contextlib import contextmanager
from hashlib import sha256
from itertools import islice
from numbers import Integral
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import h5py
import numpy as np
from progressbar import progressbar
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
        if isinstance(value, str): value = value.encode('utf-8')
        else: value = bytes(value)
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
            message = 'shared mapping does not match index identity/order'
            raise ValueError(message)
        if any(not np.array_equal(saved[key], value)
            for key, value in expected.items()):
            message = 'shared mapping does not match index type grouping'
            raise ValueError(message)
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
    if not index.metadatas.name.endswith(('_center', '_mean')):
        message = 'index name must end in _center or _mean'
        raise ValueError(message)
    filename, mapping_path = Path(filename), Path(mapping_path)
    if filename.exists() and not overwrite: raise FileExistsError(filename)
    lv_locations.store_path(model, condition, 1, workspace)
    n_tokens = len(index.metadatas)
    if n_tokens <= top_k:
        raise ValueError('index needs more entries than top_k')
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


class Result:
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

                path = self.filename.parent / attrs['index_directory']
                path = path.resolve()
                self.metadatas = VectorMetadatas.load_json(
                    path / f'{attrs["index_name"]}.json')
            else:
                self.metadatas = index.metadatas
                if self.metadatas.name != attrs['index_name']:
                    message = 'results belong to a different vector index'
                    raise ValueError(message)
            identity = _identity(self.metadatas)
            if any(attrs[key] != value for key, value in identity.items()):
                message = 'results do not match vector index identity/order'
                raise ValueError(message)
            if mapping_path is None:
                mapping_path = self.filename.parent / attrs['mapping_path']
            self._mapping = _shared_mapping(mapping_path, self.metadatas,
                identity)
            self._validate_arrays()
            self._tokens = {}
            self._types = {}
            self._queries = {}
            self._type_labels = {self.metadatas[int(indices[0])].label: i
                for i, indices in enumerate(self._mapping['type_indices'])}
            self._description = (f'model={self.model!r} '
                f'condition={self.condition!r} layer={self.layer} '
                f'pooling={self.pooling!r} tokens={len(self)}')
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

    def __repr__(self):
        if not self._file.id.valid: return '<Result closed>'
        return f'<Result {self._description}>'

    def _check_open(self):
        '''Raise when the result file has been closed.'''
        if not self._file.id.valid: raise ValueError('results are closed')

    def _token_index(self, token_index):
        valid_integer = (isinstance(token_index, Integral)
            and not isinstance(token_index, bool))
        if not valid_integer or not 0 <= token_index < len(self):
            raise IndexError('token_index outside vector index')
        self._check_open()
        return int(token_index)

    def __getitem__(self, token_index):
        token_index = self._token_index(token_index)
        if token_index not in self._queries:
            self._queries[token_index] = Query(self, token_index)
        return self._queries[token_index]

    def __iter__(self):
        for token_index in range(len(self)):
            yield self[token_index]

    def collect_queries(self):
        '''Cache all queries and matches with a progress bar; returns None.

        Reads numeric results only; annotation stores remain unopened.
        '''
        self._check_open()
        for token_index in progressbar(range(len(self))):
            self[token_index].neighbours

    @property
    def attributes(self):
        '''General file metadata, without annotation or vector loading.'''
        self._check_open()
        return dict(self._file.attrs)

    @property
    def model(self):
        return self.attributes['model']

    @property
    def condition(self):
        return self.attributes['condition']

    @property
    def layer(self):
        from echoframe.key_helper import unpack_echoframe_key

        return unpack_echoframe_key(self.source_key(0))['layer']

    @property
    def pooling(self):
        return self.attributes['index_name'].rsplit('_', 1)[1]

    @property
    def types(self):
        '''Types in first-token order; objects are shared within this result.'''
        self._check_open()
        return tuple(self.type(label) for label in self._type_labels)

    def type(self, label):
        '''Look up a type by its orthographic or phoneme-sequence label.'''
        self._check_open()
        if label not in self._type_labels:
            expected = ', '.join(repr(x) for x in islice(self._type_labels, 3))
            if len(self._type_labels) > 3: expected += ', ...'
            message = f'label not in lexicon: {label!r}; expected: {expected}'
            raise ValueError(message)
        type_index = self._type_labels[label]
        if type_index not in self._types:
            self._types[type_index] = Type(self, type_index)
        return self._types[type_index]

    def token(self, token_index):
        '''Return the shared token object for one vector-index position.'''
        token_index = self._token_index(token_index)
        if token_index not in self._tokens:
            self._tokens[token_index] = Token(self, token_index)
        return self._tokens[token_index]

    def read(self, start, stop):
        '''Read a contiguous query batch without opening annotation stores.'''
        self._token_index(start)
        if (isinstance(stop, bool) or not isinstance(stop, Integral)
            or not start <= stop <= len(self)):
            raise IndexError('invalid query batch bounds')
        return {name: self._file[name][start:stop] for name in ARRAY_DTYPES}

    def _annotation_store(self):
        self._check_open()
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
            for query in self._queries.values():
                query._same = None
                query._neighbours = None
            self._queries.clear()
            self._tokens.clear()
            self._types.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class Query:
    '''One query with neighbours and same-type matches in global rank order.

    The raw same_type_* arrays retain their on-disk token order. Match ranks
    are zero-based over all dataset tokens, including the query itself.
    '''

    def __init__(self, result, token_index):
        self.result = result
        self.token_index = token_index
        self._same = None
        self._neighbours = None

    def __repr__(self):
        if not self.result._file.id.valid:
            return f'<Query index={self.token_index} closed>'
        metadatas = self.result.metadatas
        label = metadatas[self.token_index].label
        description = f'index={self.token_index} label={label!r}'
        if self._neighbours is not None and len(self._neighbours) >= 10:
            correct = sum(metadatas[match._token_index].label == label
                for match in self._neighbours[:10])
            description += f' precision@10={correct / 10:.1f}'
        return f'<Query {description}>'

    @property
    def token(self):
        return self.result.token(self.token_index)

    @property
    def same(self):
        self.result._check_open()
        if self._same is None:
            indices = self.same_type_indices
            ranks = self.same_type_ranks
            similarities = self.same_type_similarities
            order = np.argsort(ranks, kind='stable')
            self._same = tuple(Match(self, indices[i], similarities[i],
                ranks[i]) for i in order)
        return self._same

    @property
    def neighbours(self):
        self.result._check_open()
        if self._neighbours is None:
            same = {match._token_index: match for match in self.same}
            self_rank = same[self.token_index].rank
            indices = self.neighbour_indices
            similarities = self.neighbour_similarities
            # Removing self shifts only positions at or beyond its global rank.
            self._neighbours = tuple(same[int(index)] if int(index) in same
                else Match(self, index, similarities[position],
                    position + int(position >= self_rank))
                for position, index in enumerate(indices))
        return self._neighbours

    def _array(self, name):
        self.result._check_open()
        return self.result._file[name][self.token_index]

    @property
    def neighbour_indices(self):
        return self._array('neighbour_indices')

    @property
    def neighbour_similarities(self):
        return self._array('neighbour_similarities')

    @property
    def same_type_indices(self):
        self.result._check_open()
        mapping = self.result._mapping
        return mapping['type_indices'][mapping['token_types'][
            self.token_index]].copy()

    @property
    def same_type_ranks(self):
        return self._array('same_type_ranks')

    @property
    def same_type_similarities(self):
        return self._array('same_type_similarities')

    def precision(self, n=10):
        '''Return the fraction of the first n neighbours matching this type.'''
        _positive_integer(n, 'n')
        neighbours = self.neighbours
        if n > len(neighbours):
            raise ValueError('n exceeds saved neighbour count')
        metadatas = self.result.metadatas
        label = metadatas[self.token_index].label
        return sum(metadatas[match._token_index].label == label
            for match in neighbours[:n]) / n


class Token:
    '''A token identity shared by queries, matches and its parent Type.'''

    def __init__(self, result, token_index):
        self.result = result
        self.index = token_index

    def __repr__(self):
        if not self.result._file.id.valid:
            return f'<Token index={self.index} closed>'
        return f'<Token index={self.index} label={self.label!r}>'

    @property
    def label(self):
        self.result._check_open()
        return self.result.metadatas[self.index].label

    @property
    def type(self):
        return self.result.type(self.label)

    @property
    def source_key(self):
        '''Echoframe key; no annotation store is opened.'''
        return self.result.source_key(self.index)

    @property
    def phraser_key(self):
        '''Phraser key decoded from the source key, without opening stores.'''
        from echoframe.key_helper import unpack_echoframe_key

        return unpack_echoframe_key(self.source_key)['phraser_key']

    @property
    def metadata(self):
        '''Look up Echoframe metadata lazily, with Phraser attached.'''
        return self.result.metadata(self.index)

    def context(self, n_phone_context=10, n_word_context=3):
        return self.result.context(self.index, n_phone_context, n_word_context)


class Type:
    '''A label and its tokens in stable vector-index order, without ranks.'''

    def __init__(self, result, type_index):
        self.result = result
        self.index = type_index
        self._tokens = None

    def __repr__(self):
        if not self.result._file.id.valid:
            return f'<Type index={self.index} closed>'
        return f'<Type label={self.label!r} tokens={len(self)}>'

    @property
    def label(self):
        self.result._check_open()
        index = int(self.result._mapping['type_indices'][self.index, 0])
        return self.result.metadatas[index].label

    def __len__(self):
        self.result._check_open()
        return self.result._mapping['type_indices'].shape[1]

    @property
    def tokens(self):
        self.result._check_open()
        if self._tokens is None:
            self._tokens = tuple(self.result.token(int(index))
                for index in self.result._mapping['type_indices'][self.index])
        return self._tokens


class Match:
    '''A query-relative token with a global, zero-based rank including self.'''

    def __init__(self, query, token_index, similarity, rank):
        self.query = query
        self._token_index = int(token_index)
        self._similarity = float(similarity)
        self._rank = int(rank)

    def __repr__(self):
        if not self.query.result._file.id.valid:
            return f'<Match index={self._token_index} closed>'
        label = self.query.result.metadatas[self._token_index].label
        return (f'<Match index={self._token_index} label={label!r} '
            f'rank={self._rank} similarity={self._similarity:.4f}>')

    @property
    def token(self):
        return self.query.result.token(self._token_index)

    @property
    def similarity(self):
        self.query.result._check_open()
        return self._similarity

    @property
    def rank(self):
        self.query.result._check_open()
        return self._rank
