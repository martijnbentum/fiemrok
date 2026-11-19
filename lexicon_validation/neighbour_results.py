'''Read and explore cosine retrieval results linked to ordered SVS metadata.

Row i always represents vector-index token i. Neighbours exclude only self.
Same-type columns follow the shared mapping and include self. Global ranks
are zero-based over all tokens; equal similarities rank by token index.
'''

from itertools import islice
from numbers import Integral
from pathlib import Path

from echoframe.key_helper import unpack_echoframe_key
import h5py
import numpy as np
from progressbar import progressbar
from speech_vector_search.metadata import VectorMetadatas
from speech_vector_search.search import make_vector_path

from . import lv_stores
from . import token_features
from .neighbour_helper import (ARRAY_DTYPES, INVALID_INDEX, STATUSES,
    fingerprint, index_identity, load_shared_mapping,
    validate_positive_integer)


class Result:
    '''Disk-backed query results; row i belongs to vector-index token i.

    index: optional existing SVS index; otherwise load only its metadata
    store: optional caller-owned Echoframe store with local Phraser attached
    Numeric reads need no annotation store. Metadata access uses module-owned
    shared local stores when none was passed. Phraser items are loaded in
    bulk from the saved selection and shared by results of one dataset. Closing this result closes only
    its HDF5 file; call lv_stores.close_stores() when all store users are done.
    '''

    def __init__(self, filename, index=None, mapping_path=None, store=None):
        '''Open the saved result and load its metadata and query views.'''
        self.filename = Path(filename)
        self._store = store
        load_result(self, index=index, mapping_path=mapping_path)

    def __len__(self):
        '''Return the number of indexed tokens.'''
        return len(self.metadatas)

    def __repr__(self):
        '''Describe the result condition or report that it is closed.'''
        if not self._file.id.valid: return '<Result closed>'
        return f'<Result {self._description}>'

    def _check_open(self):
        '''Raise when the result file has been closed.'''
        if not self._file.id.valid: raise ValueError('results are closed')

    def _token_index(self, token_index):
        '''Validate a token index and return it as a Python integer.'''
        token_index = _validate_integer_index(token_index, len(self),
            'token_index outside vector index')
        self._check_open()
        return token_index

    def __getitem__(self, token_index):
        '''Return cached queries for one token in vector-index order.'''
        token_index = self._token_index(token_index)
        if token_index not in self._queries:
            cls = FrameOffsetQueries if self.result_kind == 'frame_offsets' else Query
            self._queries[token_index] = cls(self, token_index)
        return self._queries[token_index]

    def __iter__(self):
        '''Yield token queries in vector-index order.'''
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
        '''Return the cached result kind while the file is open.'''
        self._check_open()
        return self._result_kind

    @property
    def offsets(self):
        '''Return cached, read-only signed offsets for frame-offset results.'''
        self._check_open()
        if self.result_kind != 'frame_offsets':
            raise ValueError('ordinary results have no offset axis')
        return self._offsets

    @property
    def model(self):
        '''Return the embedding model name.'''
        return self.attributes['model']

    @property
    def condition(self):
        '''Return the word or cross-boundary dataset condition.'''
        return self.attributes['condition']

    @property
    def layer(self):
        '''Return the model layer decoded from the first source key.'''
        return unpack_echoframe_key(self.echoframe_key(0))['layer']

    @property
    def pooling(self):
        '''Return the pooling method recorded in the index name.'''
        return self.attributes['index_name'].rsplit('_', 1)[1]

    @property
    def types(self):
        '''Types in first-token order; objects are shared within this result.'''
        self._check_open()
        return tuple(self.type(label) for label in self._type_labels)

    def type(self, label):
        '''Look up a type by its orthographic or phoneme-sequence label.

        Example: results.type('huis').tokens returns all selected tokens
        of the word type 'huis' in vector-index order.
        '''
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

    @property
    def echoframe_store(self):
        '''Return the shared Echoframe store, opening it lazily if needed.

        The store links embeddings to the attached Phraser store containing
        speech annotations.
        '''
        self._check_open()
        if self._store is None:
            f = lv_stores.load_echoframe_store
            self._store = f(self.model, self.condition,
                workspace=self._workspace)
        return self._store

    @property
    def _workspace(self):
        '''Return the validation workspace recorded in the result file.'''
        return self.filename.parent / self._file.attrs['workspace']

    @property
    def phraser_store(self):
        '''Return the local speech-annotation store already attached to Echoframe.'''
        source_id = lv_stores.CGN_SOURCE_ID
        return self.echoframe_store.load_phraser_store(source_id)

    def echoframe_key(self, token_index):
        '''Return a token’s Echoframe source key without loading annotations.'''
        token_index = self._token_index(token_index)
        return self.metadatas[token_index].source_echoframe_keys[0]

    def echoframe_metadata(self, token_index):
        '''Load one Echoframe metadata object, including Phraser access.'''
        key = self.echoframe_key(token_index)
        metadata = self.echoframe_store.load_metadata(key)
        if metadata is None: raise ValueError('Echoframe metadata not found')
        return metadata

    @property
    def items(self):
        '''Phraser words or markers in row order, loaded once per dataset.

        Uses the Phraser store attached to a caller-owned Echoframe store,
        otherwise the shared local store; the Echoframe store stays unopened.
        '''
        self._check_open()
        if self._items is None:
            store = None if self._store is None else self.phraser_store
            keys, items = token_features.load_selected_items(self.condition,
                store, workspace=self._workspace)
            if fingerprint(keys) != self.attributes['selection_fingerprint']:
                raise ValueError('saved selection does not match results')
            self._items = items
        return self._items

    @property
    def vectors(self):
        '''Pooled vectors in row order, L2-normalized as saved by SVS.

        Uses the supplied index, otherwise memory-maps the saved .npy file
        so indexing reads only the requested rows.
        '''
        self._check_open()
        if self._vectors is None:
            if self._index is not None: vectors = self._index.vectors
            else: vectors = np.load(self._vector_path, mmap_mode='r')
            if vectors.ndim != 2 or len(vectors) != len(self):
                raise ValueError('vectors do not match the result rows')
            self._vectors = vectors
        return self._vectors

    @property
    def _vector_path(self):
        '''Return the saved SVS vector file named in the result file.'''
        attrs = self._file.attrs
        directory = (self.filename.parent / attrs['index_directory']).resolve()
        return make_vector_path(directory, attrs['index_name'])

    @property
    def features(self):
        '''Per-token feature arrays in row order; see token_features.

        Loads the saved table, building it first when missing.
        '''
        self._check_open()
        if self._features is None:
            store = None if self._store is None else self.phraser_store
            f = token_features.load_token_features
            self._features = f(self.condition, store,
                workspace=self._workspace,
                selection_fingerprint=self.attributes['selection_fingerprint'])
        return self._features

    def phraser_metadata(self, token_index):
        '''Return the Phraser annotation object, such as a word or marker.'''
        token_index = self._token_index(token_index)
        return self.items[token_index]

    def context(self, token_index, n_phone_context=10, n_word_context=3):
        '''Build phone/word context on demand using the existing helpers.'''
        from .neighbour_metadata import echoframe_key_to_metadata

        return echoframe_key_to_metadata(self.echoframe_key(token_index),
            self.echoframe_store, n_phone_context, n_word_context)

    def close(self):
        '''Close the result file and clear views without closing shared stores.'''
        close_result(self)

    def __enter__(self):
        '''Return this result for use in a with block.'''
        return self

    def __exit__(self, *exception):
        '''Close the result when leaving a with block.'''
        self.close()


class FrameOffsetQueries:
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
        '''Reference one token query, optionally at a saved frame-offset position.'''
        self.result = result
        self.token_index = token_index
        self.offset_position = offset_position
        self._same = None
        self._neighbours = None
        self._status = None

    def __repr__(self):
        '''Describe the query, showing precision@10 when neighbours are cached.'''
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
        '''Return the original token shared with this result.'''
        return self.result.token(self.token_index)

    @property
    def valid(self):
        '''Return whether this query has usable retrieval results.'''
        return self._get_status() == 0

    @property
    def invalid_reason(self):
        '''Return None, 'outside_recording', or 'outside_model_outputs'.'''
        status = self._get_status()
        reasons = {value: reason for reason, value in STATUSES.items()}
        if status not in reasons: raise ValueError('invalid query status')
        return reasons[status]

    def _get_status(self):
        '''Return cached query status, reading it once while the result is open.'''
        self.result._check_open()
        if self._status is None:
            self._status = (0 if self.offset_position is None
                else int(self._read_query_array('status')))
        return self._status

    @property
    def query_time(self):
        '''Return frame-query time in seconds, NaN if invalid, or None for ordinary queries.'''
        if self.offset_position is None: return None
        return float(self._read_query_array('query_times'))

    @property
    def offset(self):
        '''Return the signed frame offset, or None for ordinary queries.'''
        if self.offset_position is None: return None
        return int(self.result.offsets[self.offset_position])

    def _clear(self):
        '''Clear cached status, same-type matches and neighbours.'''
        self._status = None
        self._same = None
        self._neighbours = None

    @property
    def ranked_same_type_matches(self):
        '''Return same-type matches including self in rank order; empty if invalid.'''
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
        '''Return nearest matches excluding self in rank order; empty if invalid.'''
        self.result._check_open()
        if not self.valid: return ()
        if self._neighbours is None:
            same = {match._token_index: match for match in self.ranked_same_type_matches}
            self_rank = same[self.token_index].rank
            indices = self.neighbour_indices
            similarities = self.neighbour_similarities
            # Removing self shifts only positions at or beyond its global rank.
            self._neighbours = tuple(same[int(index)] if int(index) in same
                else Match(self, index, similarities[position],
                    position + int(position >= self_rank))
                for position, index in enumerate(indices))
        return self._neighbours

    def _read_query_array(self, name):
        """Read this query's values from a named HDF5 dataset."""
        self.result._check_open()
        position = (self.token_index if self.offset_position is None
            else (self.token_index, self.offset_position))
        return self.result._file[name][position]

    @property
    def neighbour_indices(self):
        '''Return the saved nearest-neighbour token indices.'''
        return self._read_query_array('neighbour_indices')

    @property
    def neighbour_similarities(self):
        '''Return cosine similarities in saved neighbour order.'''
        return self._read_query_array('neighbour_similarities')

    @property
    def same_type_indices(self):
        '''Return a copy of same-type token indices in shared mapping order.'''
        self.result._check_open()
        mapping = self.result._mapping
        return mapping['type_indices'][mapping['token_types'][
            self.token_index]].copy()

    @property
    def same_type_ranks(self):
        '''Return zero-based global ranks, including self, in mapping order.'''
        return self._read_query_array('same_type_ranks')

    @property
    def same_type_similarities(self):
        '''Return same-type cosine similarities in mapping order.'''
        return self._read_query_array('same_type_similarities')

    def precision(self, n=10):
        '''Return the fraction of the first n neighbours matching this type.'''
        validate_positive_integer(n, 'n')
        self.result._check_open()
        if n > self.result._n_neighbours:
            raise ValueError('n exceeds saved neighbour count')
        if not self.valid: return float('nan')
        neighbours = self.neighbours
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
    def echoframe_key(self):
        '''Echoframe key; no annotation store is opened.'''
        return self.result.echoframe_key(self.index)

    @property
    def phraser_key(self):
        '''Phraser key decoded from the source key, without opening stores.'''
        return unpack_echoframe_key(self.echoframe_key)['phraser_key']

    @property
    def echoframe_metadata(self):
        '''Look up Echoframe metadata lazily, with Phraser attached.'''
        return self.result.echoframe_metadata(self.index)

    @property
    def phraser_metadata(self):
        '''Return the token annotation from the bulk-loaded Phraser items.'''
        return self.result.phraser_metadata(self.index)

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


def close_result(result):
    '''Close the HDF5 file and clear cached views without closing stores.'''
    result._file.close()
    result._items = None
    result._features = None
    result._vectors = None
    for query in result._queries.values():
        query._clear()
    result._queries.clear()
    result._tokens.clear()
    result._types.clear()


def load_result(result, index=None, mapping_path=None):
    '''Open the result file and load metadata, validated arrays and views.

    Use the supplied index and mapping path, or load those saved in the file.
    Close the file if loading fails; otherwise Result.close owns cleanup.
    '''
    result._file = h5py.File(result.filename, 'r')
    result._index = index
    try:
        result.metadatas = _load_metadatas(result, index)
        result._mapping = _load_mapping(result, mapping_path)
        _initialize_result_views(result)
        validate_result(result)
    except BaseException:
        result._file.close()
        raise


def _load_metadatas(result, index):
    '''Load index metadata or check the supplied index name.'''
    attrs = result._file.attrs
    if attrs['format_version'] != 1:
        raise ValueError('unsupported neighbour results format')
    if index is None:
        path = (result.filename.parent / attrs['index_directory']).resolve()
        return VectorMetadatas.load_json(path / f'{attrs["index_name"]}.json')
    if index.metadatas.name != attrs['index_name']:
        raise ValueError('results belong to a different vector index')
    return index.metadatas


def _load_mapping(result, mapping_path):
    '''Load the type mapping after checking index identity and order.'''
    attrs = result._file.attrs
    identity = index_identity(result.metadatas)
    if any(attrs[key] != value for key, value in identity.items()):
        raise ValueError('results do not match vector index identity/order')
    if mapping_path is None:
        mapping_path = result.filename.parent / attrs['mapping_path']
    return load_shared_mapping(mapping_path, result.metadatas, identity)


def _initialize_result_views(result):
    '''Cache result axes, object views, type labels and the description.'''
    result._result_kind = result._file.attrs.get('result_kind', 'neighbours')
    result._offsets = None
    if result._result_kind == 'frame_offsets':
        result._offsets = result._file['offsets'][:]
        result._offsets.setflags(write=False)
    result._tokens = {}
    result._types = {}
    result._queries = {}
    result._items = None
    result._features = None
    result._vectors = None
    result._type_labels = {result.metadatas[int(indices[0])].label: i
        for i, indices in enumerate(result._mapping['type_indices'])}
    result._description = (f'model={result.model!r} '
        f'condition={result.condition!r} layer={result.layer} '
        f'pooling={result.pooling!r} tokens={len(result)}')


def validate_result(result):
    '''Validate stored array shapes and dtypes for an open Result.

    Checks the four retrieval arrays and, for frame-offset results, their
    offsets and timing arrays. Does not inspect similarity or rank values.
    Raises ValueError for invalid shapes, dtypes or result kinds, and
    KeyError when required datasets or attributes are missing.
    '''
    result._check_open()
    neighbour_shape = _validate_neighbour_indices(result)
    _validate_neighbour_similarities(result, neighbour_shape)
    _validate_same_type_ranks(result, neighbour_shape)
    _validate_same_type_similarities(result, neighbour_shape)
    if result.result_kind == 'frame_offsets':
        _validate_frame_arrays(result, neighbour_shape)
    result._n_neighbours = neighbour_shape[-1]


def _validate_neighbour_indices(result):
    '''Check token/offset/neighbour axes and return the validated shape.'''
    kind = result.result_kind
    if kind not in ('neighbours', 'frame_offsets'):
        raise ValueError('unsupported result kind')
    shape = result._file['neighbour_indices'].shape
    dimensions = 3 if kind == 'frame_offsets' else 2
    if (len(shape) != dimensions or shape[0] != len(result)
        or not 0 < shape[-1] < len(result)):
        raise ValueError('invalid neighbour array shape')
    _validate_array(result._file, 'neighbour_indices', shape,
        ARRAY_DTYPES['neighbour_indices'])
    return shape


def _validate_neighbour_similarities(result, neighbour_shape):
    _validate_array(result._file, 'neighbour_similarities', neighbour_shape,
        ARRAY_DTYPES['neighbour_similarities'])


def _validate_same_type_ranks(result, neighbour_shape):
    n_same_type = result._mapping['type_indices'].shape[1]
    shape = (*neighbour_shape[:-1], n_same_type)
    _validate_array(result._file, 'same_type_ranks', shape,
        ARRAY_DTYPES['same_type_ranks'])


def _validate_same_type_similarities(result, neighbour_shape):
    n_same_type = result._mapping['type_indices'].shape[1]
    shape = (*neighbour_shape[:-1], n_same_type)
    _validate_array(result._file, 'same_type_similarities', shape,
        ARRAY_DTYPES['same_type_similarities'])


def _validate_offsets(result, n_offsets):
    offsets = result.offsets
    maximum = result._file.attrs['max_offset']
    if (offsets.dtype != np.dtype('int32') or offsets.ndim != 1
        or not np.array_equal(offsets, np.arange(-maximum, maximum + 1))
        or n_offsets != len(offsets)):
        raise ValueError('invalid offsets')

def _validate_frame_arrays(result, neighbour_shape):
    _validate_offsets(result, neighbour_shape[1])
    for name, dtype in (('status', 'uint8'), ('query_times', 'float64')):
        _validate_array(result._file, name, neighbour_shape[:2], dtype)
    for name, dtype in (('center_indices', 'uint32'),
        ('window_starts', 'float64'), ('window_ends', 'float64'),
        ('reference_starts', 'float64'), ('reference_ends', 'float64'),
        ('reference_kinds', 'S7')):
        _validate_array(result._file, name, (len(result),), dtype)

def _validate_array(file, name, shape, dtype):
    array = file[name]
    if array.shape != shape or array.dtype != np.dtype(dtype):
        raise ValueError(f'invalid {name} shape or dtype')


def _validate_integer_index(value, size, message='index outside range'):
    '''Return an integer index in [0, size), rejecting booleans and other types.'''
    valid_integer = isinstance(value, Integral) and not isinstance(value, bool)
    if not valid_integer or not 0 <= value < size:
        raise IndexError(message)
    return int(value)
