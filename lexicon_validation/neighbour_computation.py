'''Compute cosine neighbours and same-type ranks, streaming results to HDF5.

Ordinary token queries and contextual frame queries share the saved type
mapping and result format read by neighbour_results.Result.
'''

from contextlib import contextmanager
from numbers import Integral
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import h5py
import numpy as np
from echoframe.key_helper import unpack_echoframe_key
from speech_vector_search import gpu_search, ranking

from . import lv_locations
from .neighbour_helper import (ARRAY_DTYPES, INVALID_INDEX, STATUSES,
    fingerprint, index_identity, load_shared_mapping, type_mapping,
    validate_positive_integer)

CROSS_SLOTS = ('a', 'b', 'prev', 'next')


def write_neighbours(index, filename, mapping_path, selected_keys,
    top_k=30, batch_size=64, overwrite=False, compression='lzf',
    model='hubert', condition='word', workspace=lv_locations.workspace,
    gpu=False):
    '''Write exact cosine neighbours and all same-type global ranks.

    index:         searched SVS index, in saved selection order
    selected_keys: exact ordered Phraser dataset definition
    mapping_path:  shared type mapping for this dataset
    batch_size:    bounds temporary similarities to this many query rows
    compression:   lossless HDF5 filter, or None
    gpu:           score and rank on a CUDA GPU; raises without CUDA
    Returns the published HDF5 path; refuses existing results by default.
    '''
    validate_positive_integer(top_k, 'top_k')
    validate_positive_integer(batch_size, 'batch_size')
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
    identity = index_identity(index.metadatas, selected_keys)
    mapping = _create_mapping(mapping_path, index.metadatas, identity)
    n_same_type = mapping['type_indices'].shape[1]
    scorer = gpu_search.GpuScorer(index.vectors) if gpu else None
    with _atomic_output(filename, overwrite) as temporary:
        with h5py.File(temporary, 'w') as output:
            _result_attributes(output, index, identity, filename, mapping_path,
                workspace, model, condition, 'neighbours')
            for name, dtype in ARRAY_DTYPES.items():
                width = top_k if name.startswith('neighbour_') else n_same_type
                output.create_dataset(name, (n_tokens, width), dtype=dtype,
                    chunks=(min(batch_size, n_tokens), width),
                    compression=compression, shuffle=compression is not None)
            _write_batches(output, index, scorer, mapping, top_k, batch_size)
    return filename


def write_frame_offsets(index, filename, mapping_path, selected_keys, queries,
    max_offset=100, top_k=30, batch_size=32, overwrite=False, compression='lzf',
    model='hubert', condition='word', workspace=lv_locations.workspace,
    stride=0.02, field=0.025, base_collar_ms=2000, gpu=False,
    query_pooling='center'):
    '''Stream one token's contextual frame queries at a time into HDF5.

    queries yields dictionaries in saved token order: key, records,
    window_start/end, reference_start/end and reference_kind. Each record has
    offset, center_index, query_time, invalid_reason and vector. All same-type
    ranks include self; neighbours exclude only the original query token.
    Temporary scores are bounded by batch_size * number of candidate tokens.
    gpu scores and ranks on a CUDA GPU and raises without CUDA.
    query_pooling records whether queries are single frames (center) or
    shifted frame means (mean).
    '''
    offsets = _frame_offsets(max_offset, top_k, batch_size, stride, field,
        base_collar_ms, query_pooling)
    filename, mapping_path = Path(filename), Path(mapping_path)
    if filename.exists() and not overwrite: raise FileExistsError(filename)
    lv_locations.store_path(model, condition, 1, workspace)
    n_tokens = _check_index(index, top_k)
    selected_keys = list(selected_keys)
    identity = index_identity(index.metadatas, selected_keys)
    mapping = _create_mapping(mapping_path, index.metadatas, identity)
    n_same = mapping['type_indices'].shape[1]
    scorer = gpu_search.GpuScorer(index.vectors) if gpu else None

    def score(token_index, vectors):
        same_type = mapping['type_indices'][mapping['token_types'][token_index]]
        excluded = np.full(len(vectors), token_index)
        same_types = np.tile(same_type, (len(vectors), 1))
        return _score_batches(index, scorer, vectors, excluded, same_types,
            top_k, batch_size)

    with _atomic_output(filename, overwrite) as temporary:
        with h5py.File(temporary, 'w') as output:
            _result_attributes(output, index, identity, filename, mapping_path,
                workspace, model, condition, 'frame_offsets')
            output.attrs.update(stride=stride, field=field,
                base_collar_ms=base_collar_ms, max_offset=max_offset,
                query_pooling=query_pooling)
            widths = {name: (dtype, top_k if name.startswith('neighbour_')
                else n_same) for name, dtype in ARRAY_DTYPES.items()}
            _create_frame_datasets(output, n_tokens, offsets, widths,
                batch_size, compression)
            _write_frame_queries(output, selected_keys, queries, offsets,
                widths, score)
    return filename


def write_cross_frame_offsets(index, filename, query_keys, queries,
    context_keys, target_labels, max_offset=100, top_k=30, batch_size=32,
    overwrite=False, compression='lzf', model='hubert',
    query_condition='cross_boundary', workspace=lv_locations.workspace,
    stride=0.02, field=0.025, base_collar_ms=2000, gpu=False,
    query_pooling='center'):
    '''Stream frame queries of one selection against another dataset's index.

    index:          searched SVS index of the candidate dataset
    query_keys:     ordered Phraser keys of the query selection
    queries:        frame queries in query_keys order, as write_frame_offsets
    context_keys:   per query the Phraser keys of the a, b, prev and next
                    occurrences; empty keys mean the word is missing
    target_labels:  per query the a and b labels whose index tokens are ranked
    Context occurrences present in the index are excluded from neighbours
    and rank targets, and ranks are counted as if they were absent. Labels
    without index tokens get INVALID_INDEX ranks and NaN similarities.
    '''
    offsets = _frame_offsets(max_offset, top_k, batch_size, stride, field,
        base_collar_ms, query_pooling)
    filename = Path(filename)
    if filename.exists() and not overwrite: raise FileExistsError(filename)
    lv_locations.store_path(model, query_condition, 1, workspace)
    query_keys = list(query_keys)
    n_queries = len(query_keys)
    if len(context_keys) != n_queries or len(target_labels) != n_queries:
        raise ValueError('context keys and target labels must follow the queries')
    _check_index(index, top_k + len(CROSS_SLOTS))
    identity = index_identity(index.metadatas)
    excluded = _excluded_rows(index.metadatas, context_keys)
    targets = _target_rows(index.metadatas, target_labels, excluded)
    scorer = gpu_search.GpuScorer(index.vectors) if gpu else None

    def score(token_index, vectors):
        return _score_cross_batches(index, scorer, vectors,
            excluded[token_index], targets[token_index], top_k, batch_size)

    with _atomic_output(filename, overwrite) as temporary:
        with h5py.File(temporary, 'w') as output:
            settings = lv_locations.parse_index_name(index.metadatas.name)
            output.attrs.update({f'index_{k}': v for k, v in identity.items()})
            output.attrs.update(format_version=1,
                result_kind='cross_frame_offsets', model=model,
                condition=query_condition,
                index_condition=settings['condition'],
                selection_fingerprint=fingerprint(query_keys),
                index_name=index.metadatas.name,
                index_directory=os.path.relpath(
                    Path(index.metadatas.directory).resolve(),
                    filename.parent.resolve()),
                workspace=os.path.relpath(Path(workspace).resolve(),
                    filename.parent.resolve()),
                stride=stride, field=field, base_collar_ms=base_collar_ms,
                max_offset=max_offset, query_pooling=query_pooling,
                context_slots=CROSS_SLOTS)
            output.create_dataset('excluded_indices', data=excluded)
            n_targets = targets.shape[-1]
            for i, slot in enumerate(('a', 'b')):
                output.create_dataset(f'{slot}_type_indices',
                    data=targets[:, i])
            widths = dict(neighbour_indices=('uint32', top_k),
                neighbour_similarities=('float32', top_k),
                a_type_ranks=('uint32', n_targets),
                a_type_similarities=('float32', n_targets),
                b_type_ranks=('uint32', n_targets),
                b_type_similarities=('float32', n_targets))
            _create_frame_datasets(output, n_queries, offsets, widths,
                batch_size, compression)
            _write_frame_queries(output, query_keys, queries, offsets,
                widths, score)
    return filename


def _frame_offsets(max_offset, top_k, batch_size, stride, field,
    base_collar_ms, query_pooling):
    '''Validate frame-offset settings and return the signed offsets.'''
    if query_pooling not in ('center', 'mean'):
        raise ValueError(f'unknown query pooling {query_pooling}')
    if (isinstance(max_offset, bool) or not isinstance(max_offset, Integral)
        or max_offset < 0):
        raise ValueError('max_offset must be a non-negative integer')
    validate_positive_integer(top_k, 'top_k')
    validate_positive_integer(batch_size, 'batch_size')
    if stride <= 0 or field <= 0 or base_collar_ms < 0:
        raise ValueError('invalid frame timing or collar')
    return np.arange(-max_offset, max_offset + 1, dtype='int32')


def _check_index(index, top_k):
    '''Return the index size after checking it fits the requested results.'''
    n_tokens = len(index.metadatas)
    if not top_k < n_tokens < INVALID_INDEX:
        raise ValueError('index size must exceed top_k and fit uint32')
    if len(index.vectors) != n_tokens:
        raise ValueError('vector count does not match metadata')
    return n_tokens


def _create_frame_datasets(output, n_tokens, offsets, widths, batch_size,
    compression):
    '''Create per-offset result arrays and per-token query metadata.'''
    n_offsets = len(offsets)
    output.create_dataset('offsets', data=offsets)
    for name, (dtype, width) in widths.items():
        fill = np.nan if 'similarities' in name else INVALID_INDEX
        output.create_dataset(name, (n_tokens, n_offsets, width), dtype=dtype,
            chunks=(1, min(batch_size, n_offsets), width),
            compression=compression, shuffle=compression is not None,
            fillvalue=fill)
    for name, dtype, fill in (('status', 'uint8', 255),
        ('query_times', 'float64', np.nan)):
        output.create_dataset(name, (n_tokens, n_offsets), dtype=dtype,
            chunks=(1, n_offsets), compression=compression, fillvalue=fill)
    for name, dtype in (('center_indices', 'uint32'),
        ('window_starts', 'float64'), ('window_ends', 'float64'),
        ('reference_starts', 'float64'), ('reference_ends', 'float64'),
        ('reference_kinds', 'S7')):
        output.create_dataset(name, (n_tokens,), dtype=dtype)


def _write_frame_queries(output, keys, queries, offsets, names, score):
    '''Write query metadata in key order and score valid offsets per token.

    score(token_index, vectors) yields (start, arrays) over rows of vectors.
    '''
    n_tokens, n_offsets = len(keys), len(offsets)
    iterator = iter(queries)
    started = reported = monotonic()
    for token_index, key in enumerate(keys):
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
        valid = np.flatnonzero(statuses == 0)
        if len(valid):
            vectors = np.asarray([records[i]['vector'] for i in valid])
            batch = {name: np.full((n_offsets, output[name].shape[-1]),
                output[name].fillvalue, dtype=output[name].dtype)
                for name in names}
            for start, values in score(token_index, vectors):
                stop = start + len(values['neighbour_indices'])
                for name, array in values.items():
                    batch[name][valid[start:stop]] = array
            for name, values in batch.items():
                output[name][token_index] = values
        done = token_index + 1
        elapsed = monotonic() - started
        if token_index == 0 or elapsed - reported >= 10 or done == n_tokens:
            print(f'Frame offsets: {done:,}/{n_tokens:,} tokens; '
                f'{elapsed:.1f}s elapsed', flush=True)
            reported = elapsed
    if next(iterator, None) is not None:
        raise ValueError('too many query tokens')


def _excluded_rows(metadatas, context_keys):
    '''Return per query the index rows of its context occurrences.'''
    rows = {unpack_echoframe_key(bytes(m.source_echoframe_keys[0]))
        ['phraser_key']: row for row, m in enumerate(metadatas)}
    excluded = np.full((len(context_keys), len(CROSS_SLOTS)), INVALID_INDEX,
        dtype='uint32')
    for query, keys in enumerate(context_keys):
        if len(keys) != len(CROSS_SLOTS):
            raise ValueError(f'expected keys for {CROSS_SLOTS}')
        for slot, key in enumerate(keys):
            row = rows.get(bytes(key)) if len(key) else None
            if row is not None: excluded[query, slot] = row
    return excluded


def _target_rows(metadatas, target_labels, excluded):
    '''Return per query the a and b type rows, without excluded rows.'''
    mapping = type_mapping(metadatas)
    type_rows = {metadatas[int(rows[0])].label: rows
        for rows in mapping['type_indices']}
    width = mapping['type_indices'].shape[1]
    targets = np.full((len(target_labels), 2, width), INVALID_INDEX,
        dtype='uint32')
    for query, labels in enumerate(target_labels):
        if len(labels) != 2: raise ValueError('expected a and b labels')
        for slot, label in enumerate(labels):
            rows = type_rows.get(str(label))
            if rows is None: continue
            rows = rows[~np.isin(rows, excluded[query])]
            targets[query, slot, :len(rows)] = rows
    return targets


def _score_cross_batches(index, scorer, vectors, excluded, targets, top_k,
    batch_size):
    '''Yield (start, result arrays) for one query token's valid offsets.

    excluded rows are removed from neighbours; each target rank drops by
    the number of excluded rows ranked above it.
    '''
    excluded = excluded[excluded != INVALID_INDEX].astype(np.intp)
    targets = targets.reshape(-1)
    present = targets != INVALID_INDEX
    columns = np.concatenate([np.where(present, targets, 0),
        excluded]).astype(np.intp)
    n_targets, width = len(targets), len(targets) // 2
    count = top_k + len(excluded)
    batches = (index.cosine_batches(vectors, batch_size) if scorer is None
        else scorer.cosine_batches(vectors, batch_size))
    for start, scores in batches:
        top, top_scores, ranks, similarities = _rows(scorer, scores, count,
            columns)
        keep = ~np.isin(top, excluded)
        neighbours = np.stack([row[k][:top_k] for row, k in zip(top, keep)])
        neighbour_scores = np.stack([row[k][:top_k]
            for row, k in zip(top_scores, keep)])
        excluded_ranks = ranks[:, n_targets:]
        ranks = ranks[:, :n_targets]
        ranks = ranks - (excluded_ranks[:, None, :] < ranks[:, :, None]).sum(2)
        ranks = np.where(present, ranks, INVALID_INDEX).astype('uint32')
        similarities = np.where(present, similarities[:, :n_targets], np.nan)
        yield start, dict(neighbour_indices=neighbours.astype('uint32'),
            neighbour_similarities=neighbour_scores.astype('float32'),
            a_type_ranks=ranks[:, :width], b_type_ranks=ranks[:, width:],
            a_type_similarities=similarities[:, :width].astype('float32'),
            b_type_similarities=similarities[:, width:].astype('float32'))


def _rows(scorer, scores, count, columns):
    '''Return each row's top count candidates and the ranks of columns.'''
    if scorer is not None:
        top, top_scores = gpu_search.k_nearest_neighbours_rows(scores, count)
        ranks, similarities = gpu_search.global_ranks_rows(scores,
            np.tile(columns, (len(scores), 1)))
        return top, top_scores, ranks, similarities
    top = np.array([ranking.k_nearest_neighbours(row, count) for row in scores])
    ranks = np.array([ranking.global_ranks(row, columns) for row in scores])
    return top, np.take_along_axis(scores, top, 1), ranks, scores[:, columns]


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


def _create_mapping(filename, metadatas, identity):
    expected = type_mapping(metadatas)
    fingerprints = ('selection_fingerprint', 'label_fingerprint')
    if not Path(filename).exists():
        try:
            with _atomic_output(filename) as temporary:
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, **expected,
                        **{key: identity[key] for key in fingerprints})
        except FileExistsError:
            # Another condition may have published the same mapping first.
            pass
    return load_shared_mapping(filename, metadatas, identity)


def _result_attributes(output, index, identity, filename, mapping_path,
    workspace, model, condition, result_kind):
    output.attrs.update(identity)
    output.attrs.update(format_version=1, result_kind=result_kind,
        model=model, condition=condition, index_name=index.metadatas.name,
        index_directory=os.path.relpath(
            Path(index.metadatas.directory).resolve(), filename.parent.resolve()),
        mapping_path=os.path.relpath(mapping_path.resolve(), filename.parent.resolve()),
        workspace=os.path.relpath(Path(workspace).resolve(), filename.parent.resolve()))


def _write_batches(output, index, scorer, mapping, top_k, batch_size):
    started, reported = monotonic(), 0
    n_tokens = len(index.metadatas)
    excluded = np.arange(n_tokens)
    same_types = mapping['type_indices'][mapping['token_types']]
    for start, batch in _score_batches(index, scorer, index.vectors, excluded,
        same_types, top_k, batch_size):
        done = start + len(batch['same_type_ranks'])
        for name, values in batch.items():
            output[name][start:done] = values
        elapsed = monotonic() - started
        if start == 0 or elapsed - reported >= 10 or done == n_tokens:
            print(f'Neighbours: {done:,}/{n_tokens:,} ({done / n_tokens:.1%}); '
                f'{elapsed:.1f}s elapsed', flush=True)
            reported = elapsed


def _score_batches(index, scorer, vectors, excluded, same_types, top_k,
    batch_size):
    '''Yield (start, result arrays) per batch, on the GPU when scorer is set.

    excluded holds each query's own token; same_types holds one row of
    same-type candidates per query.
    '''
    if scorer is not None:
        for start, scores in scorer.cosine_batches(vectors, batch_size):
            stop = start + len(scores)
            neighbours, neighbour_scores = gpu_search.k_nearest_neighbours_rows(
                scores, top_k, excluded_indices=excluded[start:stop])
            ranks, same_scores = gpu_search.global_ranks_rows(scores,
                same_types[start:stop])
            yield start, _result_arrays(neighbours, neighbour_scores, ranks,
                same_scores)
        return
    for start, scores in index.cosine_batches(vectors, batch_size):
        batch = {name: np.empty((len(scores), top_k
            if name.startswith('neighbour_') else same_types.shape[1]),
            dtype=dtype) for name, dtype in ARRAY_DTYPES.items()}
        for offset, similarities in enumerate(scores):
            query = start + offset
            neighbours = ranking.k_nearest_neighbours(similarities, top_k,
                excluded_index=excluded[query])
            same_type = same_types[query]
            batch['neighbour_indices'][offset] = neighbours
            batch['neighbour_similarities'][offset] = similarities[neighbours]
            batch['same_type_ranks'][offset] = ranking.global_ranks(
                similarities, same_type)
            batch['same_type_similarities'][offset] = similarities[same_type]
        yield start, batch


def _result_arrays(neighbours, neighbour_scores, ranks, same_scores):
    '''Cast row results to the saved array dtypes.'''
    values = dict(neighbour_indices=neighbours,
        neighbour_similarities=neighbour_scores, same_type_ranks=ranks,
        same_type_similarities=same_scores)
    return {name: values[name].astype(dtype)
        for name, dtype in ARRAY_DTYPES.items()}
