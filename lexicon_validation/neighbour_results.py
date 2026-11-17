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


INVALID_INDEX = np.iinfo('uint32').max
STATUSES = {None: 0, 'outside_recording': 1, 'outside_model_outputs': 2}


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


def _ranked_row(similarities, token_index, mapping, top_k):
    '''Rank one query; ties use candidate index, and only self is excluded.'''
    order = np.argsort(-similarities, kind='stable')
    neighbours = order[order != token_index][:top_k]
    ranks = np.empty(len(order), dtype='uint32')
    ranks[order] = np.arange(len(order), dtype='uint32')
    same_type = mapping['type_indices'][mapping['token_types'][token_index]]
    return dict(neighbour_indices=neighbours,
        neighbour_similarities=similarities[neighbours],
        same_type_ranks=ranks[same_type],
        same_type_similarities=similarities[same_type])


def _result_attributes(output, index, identity, filename, mapping_path,
    workspace, model, condition, result_kind):
    output.attrs.update(identity)
    output.attrs.update(format_version=1, result_kind=result_kind,
        model=model, condition=condition, index_name=index.metadatas.name,
        index_directory=os.path.relpath(
            Path(index.metadatas.directory).resolve(), filename.parent.resolve()),
        mapping_path=os.path.relpath(mapping_path.resolve(), filename.parent.resolve()),
        workspace=os.path.relpath(Path(workspace).resolve(), filename.parent.resolve()))


def _write_batches(output, index, mapping, top_k, batch_size):
    started, reported = monotonic(), 0
    n_tokens = len(index.metadatas)
    for start, scores in _cosine_batches(index, batch_size):
        batch = {name: np.empty((len(scores), output[name].shape[1]),
            dtype=dtype) for name, dtype in ARRAY_DTYPES.items()}
        for offset, similarities in enumerate(scores):
            token_index = start + offset
            row = _ranked_row(similarities, token_index, mapping, top_k)
            for name, values in row.items():
                batch[name][offset] = values
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
            _result_attributes(output, index, identity, filename, mapping_path,
                workspace, model, condition, 'neighbours')
            for name, dtype in ARRAY_DTYPES.items():
                width = top_k if name.startswith('neighbour_') else n_same_type
                output.create_dataset(name, (n_tokens, width), dtype=dtype,
                    chunks=(min(batch_size, n_tokens), width),
                    compression=compression, shuffle=compression is not None)
            _write_batches(output, index, mapping, top_k, batch_size)
    return filename


def write_frame_offsets(index, filename, mapping_path, selected_keys, queries,
    max_offset=100, top_k=30, batch_size=32, overwrite=False, compression='lzf',
    model='hubert', condition='word', workspace=lv_locations.workspace,
    stride=0.02, field=0.025, base_collar_ms=2000):
    '''Stream one token's contextual frame queries at a time into HDF5.

    queries yields dictionaries in saved token order: key, records,
    window_start/end, reference_start/end and reference_kind. Each record has
    offset, center_index, query_time, invalid_reason and vector. All same-type
    ranks include self; neighbours exclude only the original query token.
    Temporary scores are bounded by batch_size * number of candidate tokens.
    '''
    if isinstance(max_offset, bool) or not isinstance(max_offset, Integral) or max_offset < 0:
        raise ValueError('max_offset must be a non-negative integer')
    _positive_integer(top_k, 'top_k')
    _positive_integer(batch_size, 'batch_size')
    if stride <= 0 or field <= 0 or base_collar_ms < 0:
        raise ValueError('invalid frame timing or collar')
    filename, mapping_path = Path(filename), Path(mapping_path)
    if filename.exists() and not overwrite: raise FileExistsError(filename)
    lv_locations.store_path(model, condition, 1, workspace)
    n_tokens = len(index.metadatas)
    if not top_k < n_tokens < INVALID_INDEX:
        raise ValueError('index size must exceed top_k and fit uint32')
    if len(index.vectors) != n_tokens:
        raise ValueError('vector count does not match metadata')
    selected_keys = list(selected_keys)
    identity = _identity(index.metadatas, selected_keys)
    mapping = _shared_mapping(mapping_path, index.metadatas, identity, create=True)
    if index.backend_name not in ('faiss', 'brute_force'):
        raise ValueError(f'unsupported SVS backend: {index.backend_name}')
    dtype = 'float32' if index.backend_name == 'faiss' else 'float64'
    candidates = np.asarray(index.vectors, dtype=dtype)
    if not np.isfinite(candidates).all():
        raise ValueError('vectors must contain finite values')
    offsets = np.arange(-max_offset, max_offset + 1, dtype='int32')
    n_offsets = len(offsets)
    n_same = mapping['type_indices'].shape[1]
    with _atomic_output(filename, overwrite) as temporary:
        with h5py.File(temporary, 'w') as output:
            _result_attributes(output, index, identity, filename, mapping_path,
                workspace, model, condition, 'frame_offsets')
            output.attrs.update(stride=stride, field=field,
                base_collar_ms=base_collar_ms, max_offset=max_offset)
            output.create_dataset('offsets', data=offsets)
            for name, array_dtype in ARRAY_DTYPES.items():
                width = top_k if name.startswith('neighbour_') else n_same
                fill = np.nan if 'similarities' in name else INVALID_INDEX
                output.create_dataset(name, (n_tokens, n_offsets, width),
                    dtype=array_dtype, chunks=(1, min(batch_size, n_offsets), width),
                    compression=compression, shuffle=compression is not None,
                    fillvalue=fill)
            for name, array_dtype, fill in (('status', 'uint8', 255),
                ('query_times', 'float64', np.nan)):
                output.create_dataset(name, (n_tokens, n_offsets), dtype=array_dtype,
                    chunks=(1, n_offsets), compression=compression, fillvalue=fill)
            for name, array_dtype in (('center_indices', 'uint32'),
                ('window_starts', 'float64'), ('window_ends', 'float64'),
                ('reference_starts', 'float64'), ('reference_ends', 'float64'),
                ('reference_kinds', 'S7')):
                output.create_dataset(name, (n_tokens,), dtype=array_dtype)
            iterator = iter(queries)
            started = reported = monotonic()
            for token_index, key in enumerate(selected_keys):
                try: query = next(iterator)
                except StopIteration:
                    raise ValueError('missing query tokens') from None
                if query['key'] != key:
                    raise ValueError('queries do not follow saved selection order')
                records = query['records']
                if [r['offset'] for r in records] != offsets.tolist():
                    raise ValueError('query offsets do not match requested offsets')
                center = records[0]['center_index']
                if any(r['center_index'] != center for r in records):
                    raise ValueError('inconsistent center indices')
                output['center_indices'][token_index] = center
                for name in ('window_start', 'window_end', 'reference_start',
                    'reference_end', 'reference_kind'):
                    output[f'{name}s'][token_index] = query[name]
                statuses = np.asarray([STATUSES[r['invalid_reason']] for r in records],
                    dtype='uint8')
                output['status'][token_index] = statuses
                output['query_times'][token_index] = [r['query_time']
                    if status == 0 else np.nan for r, status in zip(records, statuses)]
                for start in range(0, n_offsets, batch_size):
                    stop = min(start + batch_size, n_offsets)
                    valid = [i for i in range(start, stop) if statuses[i] == 0]
                    if not valid: continue
                    vectors = np.asarray([records[i]['vector'] for i in valid])
                    if vectors.ndim != 2 or vectors.shape[1] != candidates.shape[1]:
                        raise ValueError('query vector dimensions do not match index')
                    if not np.isfinite(vectors).all():
                        raise ValueError('query vectors must contain finite values')
                    scores = np.asarray(l2_normalize_rows(vectors), dtype=dtype) @ candidates.T
                    batch = {name: np.full((stop - start, output[name].shape[-1]),
                        np.nan if 'similarities' in name else INVALID_INDEX,
                        dtype=array_dtype) for name, array_dtype in ARRAY_DTYPES.items()}
                    for position, similarities in zip(valid, scores):
                        row = _ranked_row(similarities, token_index, mapping, top_k)
                        for name, values in row.items(): batch[name][position - start] = values
                    for name, values in batch.items():
                        output[name][token_index, start:stop] = values
                done = token_index + 1
                elapsed = monotonic() - started
                if token_index == 0 or elapsed - reported >= 10 or done == n_tokens:
                    print(f'Frame offsets: {done:,}/{n_tokens:,} tokens; '
                        f'{elapsed:.1f}s elapsed', flush=True)
                    reported = elapsed
            if next(iterator, None) is not None:
                raise ValueError('too many query tokens')
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
        kind = self.result_kind
        if kind not in ('neighbours', 'frame_offsets'):
            raise ValueError('unsupported result kind')
        dimensions = 3 if kind == 'frame_offsets' else 2
        if (len(neighbour_shape) != dimensions or neighbour_shape[0] != n_tokens
            or not 0 < neighbour_shape[-1] < n_tokens):
            raise ValueError('invalid neighbour array shape')
        for name, dtype in ARRAY_DTYPES.items():
            expected = (neighbour_shape if name.startswith('neighbour_')
                else (*neighbour_shape[:-1], n_same_type))
            array = self._file[name]
            if array.shape != expected or array.dtype != np.dtype(dtype):
                raise ValueError(f'invalid {name} shape or dtype')
        if kind == 'neighbours': return
        offsets = self._file['offsets'][:]
        maximum = self._file.attrs['max_offset']
        if (offsets.dtype != np.dtype('int32') or offsets.ndim != 1
            or not np.array_equal(offsets, np.arange(-maximum, maximum + 1))
            or neighbour_shape[1] != len(offsets)):
            raise ValueError('invalid offsets')
        for name, dtype in (('status', 'uint8'), ('query_times', 'float64'),
            ('center_indices', 'uint32'), ('window_starts', 'float64'),
            ('window_ends', 'float64'), ('reference_starts', 'float64'),
            ('reference_ends', 'float64'), ('reference_kinds', 'S7')):
            expected = neighbour_shape[:2] if name in ('status', 'query_times') else (n_tokens,)
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
            cls = TokenQueries if self.result_kind == 'frame_offsets' else Query
            self._queries[token_index] = cls(self, token_index)
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
            query = self[token_index]
            if self.result_kind == 'frame_offsets':
                for offset in self.offsets: query.offset(offset).neighbours
            else:
                query.neighbours

    @property
    def attributes(self):
        '''General file metadata, without annotation or vector loading.'''
        self._check_open()
        return dict(self._file.attrs)

    @property
    def result_kind(self):
        # Ordinary files written before the explicit discriminator have this shape.
        return self.attributes.get('result_kind', 'neighbours')

    @property
    def offsets(self):
        self._check_open()
        if self.result_kind != 'frame_offsets':
            raise ValueError('ordinary results have no offset axis')
        return self._file['offsets'][:]

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
        names = list(ARRAY_DTYPES)
        if self.result_kind == 'frame_offsets':
            names += ['status', 'query_times', 'center_indices', 'window_starts',
                'window_ends', 'reference_starts', 'reference_ends', 'reference_kinds']
        return {name: self._file[name][start:stop] for name in names}

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
                query._clear()
            self._queries.clear()
            self._tokens.clear()
            self._types.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class TokenQueries:
    '''All signed frame offsets for one original token.'''

    def __init__(self, result, token_index):
        self.result = result
        self.token_index = token_index
        self._queries = {}

    @property
    def token(self):
        return self.result.token(self.token_index)

    def offset(self, offset):
        self.result._check_open()
        offsets = self.result.offsets
        if (isinstance(offset, bool) or not isinstance(offset, Integral)
            or offset < offsets[0] or offset > offsets[-1]):
            raise ValueError('offset outside saved frame range')
        offset = int(offset)
        if offset not in self._queries:
            self._queries[offset] = Query(self.result, self.token_index,
                offset_position=offset - int(offsets[0]))
        return self._queries[offset]

    def _clear(self):
        for query in self._queries.values(): query._clear()
        self._queries.clear()


class Query:
    '''One query with neighbours and same-type matches in global rank order.

    The raw same_type_* arrays retain their on-disk token order. Match ranks
    are zero-based over all dataset tokens, including the query itself.
    '''

    def __init__(self, result, token_index, offset_position=None):
        self.result = result
        self.token_index = token_index
        self.offset_position = offset_position
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
    def valid(self):
        self.result._check_open()
        return self.offset_position is None or int(self._array('status')) == 0

    @property
    def invalid_reason(self):
        if self.offset_position is None:
            self.result._check_open()
            return None
        status = int(self._array('status'))
        reasons = {value: reason for reason, value in STATUSES.items()}
        if status not in reasons: raise ValueError('invalid query status')
        return reasons[status]

    @property
    def query_time(self):
        if self.offset_position is None: return None
        return float(self._array('query_times'))

    @property
    def offset(self):
        if self.offset_position is None: return None
        return int(self.result.offsets[self.offset_position])

    def _clear(self):
        self._same = None
        self._neighbours = None

    @property
    def same(self):
        self.result._check_open()
        if not self.valid: return ()
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
        if not self.valid: return ()
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
        position = (self.token_index if self.offset_position is None
            else (self.token_index, self.offset_position))
        return self.result._file[name][position]

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
        self.result._check_open()
        if n > self.result._file['neighbour_indices'].shape[-1]:
            raise ValueError('n exceeds saved neighbour count')
        if not self.valid: return float('nan')
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
