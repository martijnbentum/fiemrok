'''Evaluate lexicon retrieval with query frames around a nucleus center.'''

import csv
import json
from math import floor
from pathlib import Path

import frame
import numpy as np
import speech_vector_search as svs
import to_vector
from progressbar import progressbar

import locations
import model_paths


def collared_window(word, max_offset, base_collar_ms=500, stride=0.02):
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
    layer, stride=0.02, field=0.025):
    '''Return frame metadata and vectors for valid offsets around the nucleus.'''
    frames = frame.make_frames_from_outputs(outputs, stride=stride,
        field=field, start_time=window_start)
    word_frames = frames.select_frames(word.start_seconds, word.end_seconds,
        percentage_overlap=100)
    nucleus_frames = frames.select_frames(nucleus.start_seconds,
        nucleus.end_seconds, percentage_overlap=100)
    if not word_frames or not nucleus_frames:
        raise ValueError('word or nucleus has no fully contained frames')
    center = nucleus_frames[len(nucleus_frames) // 2]
    hidden = outputs.hidden_states[layer]
    if hidden.ndim == 3: hidden = hidden[0]
    if len(hidden) != len(frames):
        raise ValueError('hidden-state frame count does not match alignment')
    records = []
    for offset in range(-max_offset, max_offset + 1):
        index = center.index + offset
        valid = 0 <= index < len(frames)
        records.append(dict(offset=offset, center_index=center.index,
            word_left_index=word_frames[0].index,
            word_right_index=word_frames[-1].index,
            query_index=index if valid else None,
            query_time=(frames[index].start_time + field / 2)
                if valid else None,
            vector=hidden[index] if valid else None,
            valid=valid))
    return records


def evaluate_word(word, nucleus, vector_index, model, layer=6,
    max_offset=50, base_collar_ms=500, stride=0.02, field=0.025,
    top_k=10):
    '''Compute, query, and score all valid offset frames for one word.
    word:             Phraser word with audio and start/end times
    nucleus:          Phraser segment with nucleus start/end times
    vector_index:     fixed center-frame lexicon index
    model:            loaded model instance or to_vector model name/path
    max_offset:       largest absolute query offset, in frames
    base_collar_ms:   unused context outside the word, before offset context
    '''
    window_start, window_end = collared_window(word, max_offset,
        base_collar_ms=base_collar_ms, stride=stride)
    if top_k < 1 or len(vector_index.metadatas) < top_k:
        raise ValueError('index must contain at least top_k neighbours')
    outputs = compute_contextual_embedding(word, window_start, window_end,
        model)
    records = select_query_frames(outputs, word, nucleus, window_start,
        max_offset, layer, stride=stride, field=field)
    result_rows = []
    for record in records:
        vector = record.pop('vector')
        if vector is None:
            record.update(precision_at_k=None, neighbours=[])
        else:
            result = vector_index.query(vector, top_k)
            neighbours = [dict(label=md.label, score=float(score))
                for md, score in zip(result['metadatas'], result['scores'])]
            matches = sum(x['label'] == word.label for x in neighbours)
            record.update(precision_at_k=matches / top_k,
                neighbours=neighbours)
        record.update(label=word.label, window_start=window_start,
            window_end=window_end, layer=layer, top_k=top_k,
            index_name=vector_index.metadatas.name,
            max_offset=max_offset, base_collar_ms=base_collar_ms)
        result_rows.append(record)
    return result_rows


def evaluate_words(words, nuclei, vector_index, model, output_path,
    layer=6, max_offset=50, base_collar_ms=500, stride=0.02, field=0.025,
    top_k=10):
    '''Write per-word/per-offset CSV with neighbour details as JSON.'''
    fields = ['label', 'offset', 'center_index', 'word_left_index',
        'word_right_index', 'query_index', 'query_time', 'valid',
        'precision_at_k', 'neighbours', 'window_start', 'window_end',
        'layer', 'top_k', 'index_name', 'max_offset', 'base_collar_ms']
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for word, nucleus in progressbar(list(zip(words, nuclei))):
            rows = evaluate_word(word, nucleus, vector_index, model,
                layer=layer, max_offset=max_offset,
                base_collar_ms=base_collar_ms, stride=stride,
                field=field, top_k=top_k)
            for row in rows:
                row['neighbours'] = json.dumps(row['neighbours'],
                    ensure_ascii=False)
                writer.writerow(row)
    return output_path


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


def load_index(layer=6, model_name=model_paths.wav2vec2_nl1_name,
    name_prefix='cgn_center_nucleus'):
    '''Load the existing center-frame nucleus lexicon index.'''
    name = f'{name_prefix}_layer-{layer}_model-{model_name}'
    return svs.VectorIndex.load(locations.svs_experiment_data, name)
