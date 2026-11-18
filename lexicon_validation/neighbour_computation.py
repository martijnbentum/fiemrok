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
from speech_vector_search import ranking

from . import lv_locations
from .neighbour_helper import (ARRAY_DTYPES, INVALID_INDEX, STATUSES,
    index_identity, load_shared_mapping, type_mapping, validate_positive_integer)


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
    validate_positive_integer(top_k, 'top_k')
    validate_positive_integer(batch_size, 'batch_size')
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
    identity = index_identity(index.metadatas, selected_keys)
    mapping = _create_mapping(mapping_path, index.metadatas, identity)
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
                valid = np.flatnonzero(statuses == 0)
                if len(valid):
                    vectors = np.asarray([records[i]['vector'] for i in valid])
                    batch = {name: np.full((n_offsets, output[name].shape[-1]),
                        np.nan if 'similarities' in name else INVALID_INDEX,
                        dtype=array_dtype) for name, array_dtype in ARRAY_DTYPES.items()}
                    same_type = mapping['type_indices'][mapping['token_types'][token_index]]
                    for start, scores in index.cosine_batches(vectors, batch_size):
                        positions = valid[start:start + len(scores)]
                        for position, similarities in zip(positions, scores):
                            neighbours = ranking.k_nearest_neighbours(
                                similarities, top_k, excluded_index=token_index)
                            batch['neighbour_indices'][position] = neighbours
                            batch['neighbour_similarities'][position] = similarities[neighbours]
                            batch['same_type_ranks'][position] = ranking.global_ranks(
                                similarities, same_type)
                            batch['same_type_similarities'][position] = similarities[same_type]
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
    return filename


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


def _write_batches(output, index, mapping, top_k, batch_size):
    started, reported = monotonic(), 0
    n_tokens = len(index.metadatas)
    for start, scores in index.cosine_batches(index.vectors, batch_size):
        batch = {name: np.empty((len(scores), output[name].shape[1]),
            dtype=dtype) for name, dtype in ARRAY_DTYPES.items()}
        for offset, similarities in enumerate(scores):
            token_index = start + offset
            neighbours = ranking.k_nearest_neighbours(
                similarities, top_k, excluded_index=token_index)
            same_type = mapping['type_indices'][mapping['token_types'][token_index]]
            batch['neighbour_indices'][offset] = neighbours
            batch['neighbour_similarities'][offset] = similarities[neighbours]
            batch['same_type_ranks'][offset] = ranking.global_ranks(similarities, same_type)
            batch['same_type_similarities'][offset] = similarities[same_type]
        done = start + len(scores)
        for name, values in batch.items():
            output[name][start:done] = values
        elapsed = monotonic() - started
        if start == 0 or elapsed - reported >= 10 or done == n_tokens:
            print(f'Neighbours: {done:,}/{n_tokens:,} ({done / n_tokens:.1%}); '
                f'{elapsed:.1f}s elapsed', flush=True)
            reported = elapsed
