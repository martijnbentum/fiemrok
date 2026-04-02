import random

import frame
import numpy
import phraser
from progressbar import progressbar
import to_vector

import model_paths


def load_words():
    with open('data/filler_words') as f:
        t = f.read().split('\n')
    return [word.strip() for word in t]

def load(phraser_words = None, sample = 100, model = None):
    random.seed(42)
    if phraser_words is None: phraser_words = load_phraser_words()
    if model is None: model = load_model()
    d = {}
    failed = []
    for word_label, pw in progressbar(phraser_words.items()):
        if len(pw) < sample:
            failed.append({word_label:None})
            continue
        items = []
        random.shuffle(pw)
        print(f'\n{word_label}\n')
        for phraser_word in progressbar(pw):
            item = {}
            try:audio = load_audio(phraser_word)
            except ValueError as e:
                if word_label not in failed: failed.append({word_label:None})
                continue
            hs = to_vector.to_embeddings.audio_to_vector(audio, model)
            item['phraser_word'] = phraser_word
            item['audio'] = audio
            item['hidden_states'] = hs
            items.append(item)
            if len(items) >= sample: break
        if len(items) < sample and word_label not in failed:
            print(f'FAILED: Only {len(items)} items found for {word_label}')
            failed.append({word_label: items})
        print(f'Finished {word_label} with {len(items)} items')
        d[word_label] = items
    return d, failed

def load_phraser_words(words = None):
    if words is None: words = load_words()
    d = {}
    for word in progressbar(words):
        pw = phraser.models.cache.label_to_instances(word, 'Word')
        d[word] = pw
    return d



def split_list(l, n):
    """Splits list l into n approximately equal parts."""
    k, m = divmod(len(l), n)
    return [l[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def to_word_token_vector(phraser_word, outputs, layer = 6, collar = 500):
    collar_seconds = collar / 1000
    start = phraser_word.start_seconds - collar_seconds
    n_frames = frame.frames.determine_n_frames_from_outputs(outputs)
    f = frame.Frames(n_frames, start_time = start, outputs = outputs)
    embedding = f.transformer(layer, phraser_word.start_seconds,
        phraser_word.end_seconds, average = True, percentage_overlap = 100)
    return embedding
    
