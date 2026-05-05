import pickle
from progressbar import progressbar
from phraser import models
import echoframe
import to_vector
from echoframe import store
from echoframe import segment_features as sf
from pathlib import Path
import random
from collections import Counter


def load_filtered_word_keys():
    print('loading filtered word keys')
    with open('data/filtered_word_keys.pkl', 'rb') as f:
        filtered_word_keys = pickle.load(f)
    return filtered_word_keys

def load_filtered_words():
    filtered_word_keys = load_filtered_word_keys()
    print(f'found {len(filtered_word_keys)} filtered word keys, loading words')
    words = models.cache.load_many(filtered_word_keys)
    return words

def filter_word_tokens_for_lexicon(words = None, min_freq = 100,max_freq = None,
    min_dur = 100, max_dur = 1500):
    if words is None: words = load_words()
    words = filter_components(words)
    words = filter_duration(words, min_dur, max_dur)
    frequency_dict = make_frequency_dict(words)
    words = filter_frequency(frequency_dict, words, min_freq, max_freq)
    return words

def select_words_for_lexicon(filtered_words, n_tokens = 100, random_seed = 0):
    random.seed(random_seed)
    selection = []
    labels = list(set([x.label for x in filtered_words]))
    print(f'found {len(labels)} unique labels after filtering')
    label_to_token = {label:[] for label in labels}
    print('grouping tokens by label')
    for word in progressbar(filtered_words):
        label_to_token[word.label].append(word)
    print('selecting tokens for lexicon')
    for label in progressbar(labels):
        candidates = label_to_token[label]
        selection += random.sample(candidates, n_tokens)
    print(f'Selected {len(selection)} tokens for lexicon')
    print(f'Unique labels: {len(set([x.label for x in selection]))}')
    return selection

def load_words():
    words = list(models.Word.objects.all())
    return words

def load_phones():
    phones = list(models.Phones.objects.all())
    return phones

def load_syllables():
    syllables = list(models.Syllable.objects.all())
    return syllables

def filter_components(words = None, remove_components = 'acdhm'):
    if words is None:words = load_words()
    filtered = []
    print(f'filtering components {remove_components} nwords: {len(words)}')
    for word in progressbar(words):
        f = word.audio.filename
        comp = _filename_to_comp(f)
        if comp not in remove_components: filtered.append(word)
    print(f'filtered {len(words) - len(filtered)} words') 
    print(f'remaining words: {len(filtered)}')
    return filtered
            
def filter_duration(words = None, min_dur = 100, max_dur = 1500):
    if words is None:words = load_words()
    filtered = []
    print(f'filtering dur min: {min_dur} max: {max_dur} nwords: {len(words)}')
    for word in progressbar(words):
        dur = word.duration
        if dur >= min_dur and dur <= max_dur: filtered.append(word)
    print(f'filtered {len(words) - len(filtered)} words') 
    print(f'remaining words: {len(filtered)}')
    return filtered
        
def filter_frequency(frequency_dict, words = None, min_freq = 100, 
    max_freq = None):
    print(f'filtering freq min: {min_freq} max: {max_freq} nwords: {len(words)}')
    if words is None:words = load_words()
    filtered = []
    for word in progressbar(words):
        if word.label not in frequency_dict: continue
        freq = frequency_dict[word.label]
        if min_freq is None and max_freq is None: filtered.append(word)
        elif min_freq is None and freq <= max_freq: filtered.append(word)
        elif max_freq is None and freq >= min_freq: filtered.append(word)
        elif freq >= min_freq and freq <= max_freq: filtered.append(word)
    print(f'filtered {len(words) - len(filtered)} words') 
    return filtered

def make_frequency_dict(words = None):
    if words is None:words = load_words()
    c = Counter([x.label for x in words])
    frequency_dict = {}
    print('making frequency dict')
    for word in progressbar(words):
        if len(word.label) == 1: continue
        frequency_dict[word.label] = c[word.label]
    print(f'found {len(frequency_dict)} unique labels')
    return frequency_dict

        
def _filename_to_comp(filename):
    return Path(filename).parent.parent.name.split('-')[-1]
    
    
