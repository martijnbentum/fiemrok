import pickle
from progressbar import progressbar
from stores import cgn, echoframe_stimuli, echoframe_cgn
import echoframe
import to_vector
from echoframe import store
from echoframe import segment_features 
from pathlib import Path
import random
from collections import Counter

def make_echoframe_cgn_store(lexicon = None, layers = [3,6,9,12], tags = ['cgn'], 
    model_name = 'wav2vec2_nl1_checkpoint-200000', gpu = False):
    if lexicon is None: lexicon = make_lexicon()
    for word in progressbar(lexicon):
        segment_features.compute_embeddings(word, store = echoframe_cgn, 
            layers = layers, tags = tags, model_name = model_name, gpu = gpu)


def make_lexicon(experiment = None, selected_words = None):
    print('making lexicon')
    if experiment is None: 
        import fiemrok
        e = fiemrok.Experiment() 
    if selected_words is None: selected_words = load_selected_words()
    lexicon_addition = find_targets_to_add(experiment, selected_words)
    print(f'found {len(lexicon_addition)} tokens to add to lexicon')
    lexicon = selected_words + lexicon_addition
    return lexicon

def find_targets_to_add(experiment, selected_words):
    labels = list(set([x.label for x in selected_words]))
    target_word_labels = experiment.final_target_words
    lexicon_addition = []
    for word in progressbar(target_word_labels):
        if word in labels:
            print(f'word {word} already in lexicon, skipping')
            continue
        tokens = cgn.label_to_instances(word, 'Word')
        print(f'found {len(tokens)} tokens for word {word}')
        tokens = filter_word_tokens_for_lexicon(tokens, min_freq = None)
        print(f'found {len(tokens)} tokens for word {word} after filtering')
        lexicon_addition += tokens
    labels = list(set([x.label for x in lexicon_addition]))
    print(f'found {len(labels)} unique labels for lexicon addition')
    return lexicon_addition

def make_filtered_word_keys(save = False):
    filtered_words = filter_word_tokens_for_lexicon()
    filtered_word_keys = [x.key for x in filtered_words]
    if save: save_filtered_word_keys(filtered_word_keys)
    return filtered_word_keys


def save_filtered_word_keys(filtered_word_keys):
    print('saving filtered word keys')
    with open('data/filtered_word_keys.pkl', 'wb') as f:
        pickle.dump(filtered_word_keys, f)

def save_selected_word_keys(selected_word_keys):
    print('saving selected word keys')
    with open('data/selected_word_keys.pkl', 'wb') as f:
        pickle.dump(selected_word_keys, f)

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

def load_selected_word_keys():
    print('loading selected word keys')
    with open('data/selected_word_keys.pkl', 'rb') as f:
        selected_word_keys = pickle.load(f)
    return selected_word_keys

def load_selected_words():
    selected_word_keys = load_selected_word_keys()
    print(f'found {len(selected_word_keys)} selected word keys, loading words')
    words = cgn.load_many(selected_word_keys)
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
    
    
