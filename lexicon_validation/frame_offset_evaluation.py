'''Evaluate lexicon retrieval with query frames around a word center.'''

import pickle
from math import floor
from pathlib import Path

import frame
import numpy as np
from echoframe.key_helper import unpack_echoframe_key
import speech_vector_search as svs
import to_vector

from . import lv_locations, lv_stores
import model_paths


def collared_window(word, max_offset, base_collar_ms=2000, stride=0.02):
    '''Return a symmetric audio window aligned to the reference frame grid.'''
    if type(max_offset) is not int or max_offset < 0:
        raise ValueError('max_offset must be a non-negative integer')
    if base_collar_ms < 0 or stride <= 0:
        raise ValueError('collar must be non-negative and stride positive')
    start = float(word.start_seconds)
    end = float(word.end_seconds)
    audio_end = float(word.audio.duration) / 1000
    extra = max_offset * stride
    collar = base_collar_ms / 1000 + extra
    reference_start = max(0.0, start - base_collar_ms / 1000)
    requested_start = start - collar
    steps = floor((requested_start - reference_start) / stride + 1e-9)
    window_start = reference_start + steps * stride
    window_end = min(audio_end, end + collar)
    return window_start, window_end


def compute_contextual_embedding(word, window_start, window_end, model,
    sample_rate=16_000):
    '''Compute an embedding, padding silence before time zero when required.'''
    audio_start = max(0.0, window_start)
    audio = to_vector.load_audio(word.audio.filename, start=audio_start,
        end=window_end)
    if window_start < 0:
        n_padding = round(-window_start * sample_rate)
        audio = np.pad(audio, (n_padding, 0))
    return to_vector.audio_to_vector(audio, model=model,
        numpify_output=True)


def select_query_frames(outputs, word, nucleus, window_start, max_offset,
    layer, stride=0.02, field=0.025, pooling='center'):
    '''Return queries at signed frame offsets around the word reference.

    center pooling queries single frames around the reference center frame.
    mean pooling queries the mean of the reference's fully contained frames,
    shifted together by each offset, matching the index's mean pooling.
    Every query frame must be fully within the recording; padding is
    context only.
    '''
    if pooling not in ('center', 'mean'):
        raise ValueError(f'unknown pooling method {pooling}')
    frames = frame.make_frames_from_outputs(outputs, stride=stride,
        field=field, start_time=window_start)
    word_frames = frames.select_frames(word.start_seconds, word.end_seconds,
        percentage_overlap=100)
    reference = word if nucleus is None else nucleus
    reference_frames = frames.select_frames(reference.start_seconds,
        reference.end_seconds, percentage_overlap=100)
    if not word_frames or not reference_frames:
        raise ValueError('word or reference has no fully contained frames')
    center = reference_frames[len(reference_frames) // 2]
    hidden = outputs.hidden_states[layer]
    if hidden.ndim == 3: hidden = hidden[0]
    if len(hidden) != len(frames):
        raise ValueError('hidden-state frame count does not match alignment')
    first, last = center.index, center.index
    if pooling == 'mean':
        first, last = reference_frames[0].index, reference_frames[-1].index
    records = []
    audio_end = float(word.audio.duration) / 1000
    for offset in range(-max_offset, max_offset + 1):
        index = center.index + offset
        start, stop = first + offset, last + offset
        query_start = window_start + start * stride
        query_end = window_start + stop * stride + field
        in_recording = (query_start >= -1e-9
            and query_end <= audio_end + 1e-9)
        in_outputs = 0 <= start and stop < len(frames)
        valid = in_recording and in_outputs
        reason = (None if valid else 'outside_recording' if not in_recording
            else 'outside_model_outputs')
        vector = None
        if valid and pooling == 'center': vector = hidden[index]
        elif valid: vector = hidden[start:stop + 1].mean(axis=0)
        records.append(dict(offset=offset, center_index=center.index,
            invalid_reason=reason,
            word_left_index=word_frames[0].index,
            word_right_index=word_frames[-1].index,
            query_index=index if valid else None,
            query_time=(query_start + query_end) / 2 if valid else None,
            vector=vector, valid=valid))
    return records


def evaluate_word(word, model, nucleus=None, layer=6, max_offset=100,
    base_collar_ms=2000, stride=0.02, field=0.025, pooling='center'):
    '''Compute one token's queries; retrieval and storage are shared.

    The reference is the word, unless a nucleus is supplied. center pooling
    queries single frames around its middle fully contained frame; mean
    pooling queries its shifted frame mean. Negative recording-start padding
    is context only.
    '''
    window_start, window_end = collared_window(word, max_offset,
        base_collar_ms=base_collar_ms, stride=stride)
    outputs = compute_contextual_embedding(word, window_start, window_end, model)
    records = select_query_frames(outputs, word, nucleus, window_start,
        max_offset, layer, stride=stride, field=field, pooling=pooling)
    reference = word if nucleus is None else nucleus
    return dict(key=word.key, records=records, window_start=window_start,
        window_end=window_end, reference_start=reference.start_seconds,
        reference_end=reference.end_seconds,
        reference_kind='word' if nucleus is None else 'nucleus')


def evaluate_words(words, vector_index, model, output_path, nuclei=None,
    max_offset=100, base_collar_ms=2000, stride=0.02, field=0.025,
    top_k=30, *, workspace=lv_locations.workspace,
    batch_size=32, overwrite=False, compression='lzf', gpu=False):
    '''Stream contextual queries to the shared HDF5 result format.

    model is the caller-supplied speech model matching vector_index. Query
    layer comes from the source keys and must match the parsed index name.
    Saved model name, dataset condition and pooling come from that name.
    Queries follow the index pooling: single frames for center, shifted
    reference-frame means for mean.
    Stores nearest non-self neighbours and every same-type rank/similarity.
    Invalid offsets remain present with NA results. Metrics are computed by
    Result on access. Callers own the model and vector index. gpu scores and
    ranks on a CUDA GPU and raises without CUDA.
    '''
    from .neighbour_computation import write_frame_offsets

    words = list(words)
    nuclei = [None] * len(words) if nuclei is None else list(nuclei)
    if len(words) != len(nuclei):
        raise ValueError('words and supplied nuclei must have equal lengths')
    settings = lv_locations.parse_index_name(vector_index.metadatas.name)
    layers = {unpack_echoframe_key(key)['layer']
        for metadata in vector_index.metadatas
        for key in metadata.source_echoframe_keys}
    if len(layers) != 1:
        raise ValueError(f'expected one source layer; found {sorted(layers)}')
    layer = layers.pop()
    if layer != settings['layer']:
        raise ValueError(f'index name specifies layer {settings["layer"]}, '
            f'but source keys specify layer {layer}')
    model_name, condition = settings['model'], settings['condition']
    pooling = settings['pooling']
    queries = (evaluate_word(word, model, nucleus=nucleus, layer=layer,
        max_offset=max_offset, base_collar_ms=base_collar_ms, stride=stride,
        field=field, pooling=pooling) for word, nucleus in zip(words, nuclei))
    mapping_path = lv_locations.neighbour_mapping_path(condition, workspace)
    return write_frame_offsets(vector_index, output_path, mapping_path,
        [word.key for word in words], queries, max_offset=max_offset,
        top_k=top_k, batch_size=batch_size, overwrite=overwrite,
        compression=compression, model=model_name, condition=condition,
        workspace=workspace, stride=stride, field=field,
        base_collar_ms=base_collar_ms, gpu=gpu, query_pooling=pooling)



def load_model(model_name=model_paths.wav2vec2_nl1_name, gpu=False):
    '''Load the configured model for on-the-fly vectorization.'''
    model_map = {model_paths.wav2vec2_nl1_name: ('wav2vec2', 200_000),
        model_paths.hubert_nl1_name: ('hubert', 200_000)}
    if model_name not in model_map:
        raise ValueError(f'unknown model {model_name}')
    architecture, checkpoint = model_map[model_name]
    path = model_paths.get_pretrained_model(architecture,
        checkpoint=checkpoint, version=1)
    if path is None:
        raise ValueError(f'model checkpoint not found for {model_name}')
    return to_vector.load.load_model(path, gpu=gpu)


def load_index(model, layer, pooling, condition, workspace=lv_locations.workspace,
    directory=None):
    '''Load the current validation index for an explicit condition.'''
    if directory is None: directory = Path(workspace) / 'svs_data'
    path = lv_locations.find_index_path(model, condition, layer, pooling,
        directory)
    return svs.VectorIndex.load(str(path.parent), path.stem)


def evaluate_selection(model, layer, pooling, condition, output_path,
    workspace=lv_locations.workspace, gpu=False, **kwargs):
    '''Evaluate the frozen selection in order, using local materials only.

    model: w2v or hubert; pooling selects the candidate index and the query
    pooling (single frames for center, shifted frame means for mean). Optional nuclei must follow the
    saved selection order. Callers may pass evaluate_words options in kwargs.
    gpu runs both the model and retrieval on a CUDA GPU.
    '''
    from .embeddings import MODELS

    index = load_index(model, layer, pooling, condition, workspace)
    with lv_locations.selection_path(condition, workspace).open('rb') as handle:
        keys = pickle.load(handle)
    index_keys = []
    for metadata in index.metadatas:
        if len(metadata.source_echoframe_keys) != 1:
            raise ValueError('expected one source key per indexed token')
        source = metadata.source_echoframe_keys[0]
        index_keys.append(unpack_echoframe_key(source)['phraser_key'])
    if not keys or index_keys != keys:
        raise ValueError('index does not match saved selection order')
    materials = lv_stores.load_phraser_store(workspace=workspace)
    items = list(materials.load_many(keys))
    if (len(items) != len(keys) or any(item is None for item in items)
        or [item.key for item in items] != keys):
        raise ValueError('missing or reordered selected materials')
    loaded_model = load_model(MODELS[model][0], gpu=gpu)
    return evaluate_words(items, index, loaded_model, output_path,
        workspace=workspace, gpu=gpu, **kwargs)
