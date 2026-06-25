import pickle
from progressbar import progressbar
from stores import cgn
import echoframe
import to_vector
from echoframe import store
from echoframe import segment_features as sf
from pathlib import Path
import random
from collections import Counter

def save_filtered_word_keys(filtered_word_keys):
    print('saving filtered word keys')
    with open('data/filtered_word_keys.pkl', 'wb') as f:
        pickle.dump(filtered_word_keys, f)

def load_filtered_word_keys():
    print('loading filtered word keys')
    with open('data/filtered_word_keys.pkl', 'rb') as f:
        filtered_word_keys = pickle.load(f)
    return filtered_word_keys

def load_filtered_words():
    filtered_word_keys = load_filtered_word_keys()
    print(f'found {len(filtered_word_keys)} filtered word keys, loading words')
    words = cgn.load_many(filtered_word_keys)
    return words

def filter_word_tokens_for_lexicon(words = None, min_freq = 100,
    max_freq = None, min_dur = 100, max_dur = 1500, filter_out_overlap = True, 
    filter_out_flemish = True, n_syllables = 1):
    if words is None: words = load_words()
    if filter_out_overlap: words = remove_overlap(words)
    words = filter_duration(words, min_dur, max_dur)
    words = filter_components(words)
    if filter_out_flemish: words = remove_flemish(words)
    words = filter_on_n_syllables(words, n_syllables)
    frequency_dict = make_frequency_dict(words, min_len_label = 2)
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
    print(f'loading all words from cgn database {cgn}')
    words = list(cgn.words.all())
    return words

def load_phones():
    phones = list(cgn.phones.all())
    return phones

def load_syllables():
    syllables = list(cgn.syllables.all())
    return syllables

def filter_components(words = None, remove_components = 'cdm'):
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
    print(f'remaining words: {len(filtered)}')
    return filtered

def remove_overlap(words = None):
    if words is None: words = load_words()
    filtered = []
    print(f'filtering overlapping words nwords: {len(words)}')
    for word in progressbar(words):
        if word.overlap: continue
        filtered.append(word)
    print(f'filtered {len(words) - len(filtered)} words') 
    print(f'remaining words: {len(filtered)}')
    return filtered

def remove_flemish(words = None):
    if words is None: words = load_words()
    filtered = []
    print(f'filtering flemish words nwords: {len(words)}')
    for word in progressbar(words):
        dutch_variant = _filename_to_dutch_variant(word.audio.filename)
        if dutch_variant == 'nl': filtered.append(word)
        elif dutch_variant != 'vl': 
            m = f'unknown dutch variant {dutch_variant} for word {word.label}'
            raise ValueError(m)
    print(f'filtered {len(words) - len(filtered)} words')
    print(f'remaining words: {len(filtered)}')
    return filtered
        

def filter_on_n_syllables(words = None, n_syllables = 1):
    if n_syllables is None: return words
    if words is None: words = load_words()
    filtered = []
    print(f'filtering words on n_syllables: {n_syllables} nwords: {len(words)}')
    for word in progressbar(words):
        if len(word.syllables) == n_syllables: filtered.append(word)
    print(f'filtered {len(words) - len(filtered)} words')
    print(f'remaining words: {len(filtered)}')
    return filtered
    

def make_frequency_dict(words = None, min_len_label = 2):
    if words is None:words = load_words()
    c = Counter([x.label for x in words])
    frequency_dict = {}
    print('making frequency dict')
    for word in progressbar(words):
        if len(word.label) < min_len_label: continue
        frequency_dict[word.label] = c[word.label]
    print(f'found {len(frequency_dict)} unique labels')
    return frequency_dict

        
def _filename_to_comp(filename):
    return Path(filename).parent.parent.name.split('-')[-1]

def _filename_to_dutch_variant(filename):
    return Path(filename).parent.name
    
    
