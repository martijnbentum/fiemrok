import json
import random

import echoframe
import frame
import numpy
import phraser
from progressbar import progressbar
import to_vector

import locations
import model_paths


def load_store():
    store = echoframe.Store(locations.phone_embedding_store_path)
    return store

def load_models_final_checkpoint():
    wav2vec2 = load_model('wav2vec2', checkpoint = 200_000)
    hubert = load_model('hubert', checkpoint = 200_000)
    spidr = load_model('spidr', checkpoint = 400_000)
    return [wav2vec2, hubert, spidr]

def store_all_phones(store, models = None, layers = [3,6,12], collar = 500,
    min_count = 20_000, n_store = 10_000):
    with open('../phone_counts.json') as f:
        phone_counts = json.load(f)
    if models is None: models = load_models_final_checkpoint()
    phone_labels = [k for k,v in phone_counts.items() if v > min_count]
    for model in models:
        print(f'Storing phones for model {model.model_name[0]}')
        store_phones(store, phone_labels, model, 
            tags = [f'min_count_{min_count}', 'cgn', 'nl'],
            layers = layers, collar = collar, n_store = n_store)
    

def store_phones(store, phone_labels, model, tags, layers = [3,6,12], 
    collar = 500, n_store = 10_000):
    random.seed(42)
    for phone_label in progressbar(phone_labels):
        print(f'Processing phone {phone_label}, with model {model.model_name[0]}')
        phraser_phone_tokens = load_phraser_tokens(phone_label, 
            token_type = 'Phone')
        phraser_phone_tokens = random.sample(phraser_phone_tokens, n_store)
        print(f'Loaded {len(phraser_phone_tokens)} tokens for phone {phone_label}')
        for phraser_phone_token in progressbar(phraser_phone_tokens):
            for layer in layers:
                handle_phraser_token(phraser_phone_token, model, store,
                    layers = layers, collar = collar, tags = tags)

def handle_phraser_token(phraser_token, model, store, layers = [6],
    collar = 500, tags = None):
    try:audio = load_audio(phraser_token, collar = collar)
    except ValueError as e:
        print(f'Error loading audio for {phraser_token.key.hex()}: {e}')
        return
    outputs = to_vector.to_embeddings.audio_to_vector(audio, model)
    data = select_transformer_frames(phraser_token, outputs, 
        collar = collar)
    phraser_key = phraser_token.key.hex()
    print(f'Putting {phraser_key} in store with tags {tags}')
    for layer in layers:
        store.put(phraser_key, collar, model.model_name[-1], 'hidden_state',
            layer = layer, data = data, tags = tags)
    

def load_phraser_tokens(token_label, token_type):
    return phraser.models.cache.label_to_instances(token_label, token_type)

def load_audio(phraser_token, collar = 500, enforce_collar = True):
    filename = phraser_token.audio.filename
    audio_duration = phraser_token.audio.duration
    if phraser_token.start - collar < 0 and enforce_collar:
        m = f'Collar of {collar}ms is too large for token starting at '
        m += f'{phraser_token.start}ms'
        raise ValueError(m)
    if phraser_token.end + collar > audio_duration and enforce_collar:
        m = f'Collar of {collar}ms is too large for token ending at '
        m += f'{phraser_token.end}ms in audio of duration {audio_duration}ms'
        raise ValueError(m)
    start = phraser_token.start - collar
    end = phraser_token.end + collar
    audio = to_vector.audio.load_audio_milliseconds(filename, start, end)
    return audio

def load_model(model_name = 'wav2vec2', checkpoint = None, version = None, 
    finetuned = False):
    if finetuned: 
        raise not NotImplementedError('Finetuned models not yet implemented')
    model_path = model_paths.get_pretrained_model(model_name, checkpoint, version)
    print(f'Loading model from {model_path}')
    model = to_vector.load.load_model(model_path)
    model.model_name = [model_name,str(model_path)]
    return model

def select_transformer_frames(phraser_token, outputs, layer = 6, collar = 500):
    collar_seconds = collar / 1000
    start = phraser_token.start_seconds - collar_seconds
    n_frames = frame.frames.determine_n_frames_from_outputs(outputs)
    f = frame.Frames(n_frames, start_time = start, outputs = outputs)
    M = f.transformer(layer, phraser_token.start_seconds,
        phraser_token.end_seconds, average = False, percentage_overlap = 100)
    return M

def to_phraser_token_vector(phraser_token, outputs, layer = 6, collar = 500):
    collar_seconds = collar / 1000
    start = phraser_token.start_seconds - collar_seconds
    n_frames = frame.frames.determine_n_frames_from_outputs(outputs)
    f = frame.Frames(n_frames, start_time = start, outputs = outputs)
    embedding = f.transformer(layer, phraser_token.start_seconds,
        phraser_token.end_seconds, average = True, percentage_overlap = 100)
    return embedding
