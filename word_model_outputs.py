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
    store = echoframe.Store(locations.embedding_store_path)
    return store

def load_models_final_checkpoint():
    wav2vec2 = load_model('wav2vec2', checkpoint = 200_000)
    hubert = load_model('hubert', checkpoint = 200_000)
    spidr = load_model('spidr', checkpoint = 400_000)
    return [wav2vec2, hubert, spidr]

def store_all_words(store, models = None, layers = [3,6,12], collar = 500):
    if models is None: models = load_models_final_checkpoint()
    store_fillers(store, models, layers = layers, collar = collar)
    store_targets(store, models, layers = layers, collar = collar)


def store_fillers(store, models, layers = [3,6,12], collar = 500):
    words = load_filler_words()
    tags = ['filler']
    for model in models:
        print(f'Storing fillers for model {model.model_name[0]}')
        store_words(store, words, model, tags, layers = layers, 
            collar = collar)

def store_targets(store, models, layers = [3,6,12], collar = 500):
    words = load_target_words()
    tags = ['target']
    for model in models:
        print(f'Storing targets for model {model.model_name[0]}')
        store_words(store, words, model, tags, layers = layers,
            collar = collar)

def store_words(store, words, model, tags, layers = [3,6,12], collar = 500):
    for word in progressbar(words):
        print(f'Processing word {word}, with model {model.model_name[0]}')
        phraser_word_tokens = load_phraser_word_token(word)
        for phraser_word_token in progressbar(phraser_word_tokens):
            for layer in layers:
                handle_phraser_word_token(phraser_word_token, model, store,
                    layers = layers, collar = collar, tags = tags)

def handle_phraser_word_token(phraser_word_token, model, store, layers = [6],
    collar = 500, tags = None):
    try:audio = load_audio(phraser_word_token, collar = collar)
    except ValueError as e:
        print(f'Error loading audio for {phraser_word_token.key.hex()}: {e}')
        return
    outputs = to_vector.to_embeddings.audio_to_vector(audio, model)
    data = select_transformer_frames(phraser_word_token, outputs, 
        collar = collar)
    phraser_key = phraser_word_token.key.hex()
    print(f'Putting {phraser_key} in store with tags {tags}')
    for layer in layers:
        store.put(phraser_key, collar, model.model_name[-1], 'hidden_state',
            layer = layer, data = data, tags = tags)
    

def load_phraser_word_token(word_label):
    return phraser.models.cache.label_to_instances(word_label, 'Word')

def load_audio(phraser_word, collar = 500, enforce_collar = True):
    filename = phraser_word.audio.filename
    audio_duration = phraser_word.audio.duration
    if phraser_word.start - collar < 0 and enforce_collar:
        m = f'Collar of {collar}ms is too large for word starting at '
        m += f'{phraser_word.start}ms'
        raise ValueError(m)
    if phraser_word.end + collar > audio_duration and enforce_collar:
        m = f'Collar of {collar}ms is too large for word ending at '
        m += f'{phraser_word.end}ms in audio of duration {audio_duration}ms'
        raise ValueError(m)
    start = phraser_word.start - collar
    end = phraser_word.end + collar
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

def select_transformer_frames(phraser_word, outputs, layer = 6, collar = 500):
    collar_seconds = collar / 1000
    start = phraser_word.start_seconds - collar_seconds
    n_frames = frame.frames.determine_n_frames_from_outputs(outputs)
    f = frame.Frames(n_frames, start_time = start, outputs = outputs)
    M = f.transformer(layer, phraser_word.start_seconds,
        phraser_word.end_seconds, average = False, percentage_overlap = 100)
    return M

def to_word_token_vector(phraser_word, outputs, layer = 6, collar = 500):
    collar_seconds = collar / 1000
    start = phraser_word.start_seconds - collar_seconds
    n_frames = frame.frames.determine_n_frames_from_outputs(outputs)
    f = frame.Frames(n_frames, start_time = start, outputs = outputs)
    embedding = f.transformer(layer, phraser_word.start_seconds,
        phraser_word.end_seconds, average = True, percentage_overlap = 100)
    return embedding

def load_filler_words():
    with open('data/filler_words') as f:
        t = f.read().split('\n')
    return [word.strip() for word in t]

def load_target_words():
    with open('data/target_words') as f:
        t = f.read().split('\n')
    return [word.strip() for word in t]
