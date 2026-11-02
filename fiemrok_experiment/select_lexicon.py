'''Select CGN word tokens for the reference lexicon.'''

import pickle
import random

from progressbar import progressbar

import lexicon_helper
import locations
import stores


def make_lexicon(experiment=None, selected_words=None):
    '''Return the selected CGN words plus tokens of missing target words.
    experiment:         fiemrok.Experiment, loaded when None
    selected_words:     phraser words, loaded from the saved keys when None
    '''
    print('making lexicon')
    if experiment is None:
        from . import fiemrok
        experiment = fiemrok.Experiment()
    if selected_words is None: selected_words = load_selected_words()
    lexicon_addition = find_targets_to_add(experiment, selected_words)
    print(f'found {len(lexicon_addition)} tokens to add to lexicon')
    return selected_words + lexicon_addition


def find_targets_to_add(experiment, selected_words, cgn=None):
    '''Return CGN tokens for final target words missing from the lexicon.'''
    if cgn is None: cgn = stores.load_phraser_cgn_store()
    labels = set([x.label for x in selected_words])
    lexicon_addition = []
    for word in progressbar(experiment.final_target_words):
        if word in labels:
            print(f'word {word} already in lexicon, skipping')
            continue
        tokens = cgn.label_to_instances(word, 'Word')
        print(f'found {len(tokens)} tokens for word {word}')
        tokens = filter_word_tokens_for_lexicon(tokens, min_freq=None)
        print(f'found {len(tokens)} tokens for word {word} after filtering')
        lexicon_addition += tokens
    labels = set([x.label for x in lexicon_addition])
    print(f'found {len(labels)} unique labels for lexicon addition')
    return lexicon_addition


def make_filtered_word_keys(save=False):
    filtered_words = filter_word_tokens_for_lexicon()
    filtered_word_keys = [x.key for x in filtered_words]
    if save: save_filtered_word_keys(filtered_word_keys)
    return filtered_word_keys


def save_filtered_word_keys(filtered_word_keys):
    print('saving filtered word keys')
    with open(locations.filtered_word_keys, 'wb') as f:
        pickle.dump(filtered_word_keys, f)


def save_selected_word_keys(selected_word_keys):
    print('saving selected word keys')
    with open(locations.selected_word_keys, 'wb') as f:
        pickle.dump(selected_word_keys, f)


def load_filtered_word_keys():
    print('loading filtered word keys')
    with open(locations.filtered_word_keys, 'rb') as f:
        return pickle.load(f)


def load_filtered_words():
    filtered_word_keys = load_filtered_word_keys()
    print(f'found {len(filtered_word_keys)} filtered word keys, loading words')
    return stores.load_phraser_cgn_store().load_many(filtered_word_keys)


def load_selected_word_keys():
    print('loading selected word keys')
    with open(locations.selected_word_keys, 'rb') as f:
        return pickle.load(f)


def load_selected_words():
    selected_word_keys = load_selected_word_keys()
    print(f'found {len(selected_word_keys)} selected word keys, loading words')
    return stores.load_phraser_cgn_store().load_many(selected_word_keys)


def filter_word_tokens_for_lexicon(words=None, min_freq=100,
    max_freq=None, min_dur=100, max_dur=1500, filter_out_overlap=True,
    filter_out_flemish=True, n_syllables=1):
    if words is None: words = load_words()
    if filter_out_overlap: words = lexicon_helper.remove_overlap(words)
    words = lexicon_helper.filter_duration(words, min_dur, max_dur)
    words = lexicon_helper.filter_components(words, remove_components='cdm')
    if filter_out_flemish:
        words = lexicon_helper.filter_dutch_variant(words, strict=True)
    words = lexicon_helper.filter_on_n_syllables(words, n_syllables)
    frequency_dict = lexicon_helper.make_frequency_dict(words, min_len_label=2)
    return lexicon_helper.filter_frequency(frequency_dict, words,
        min_freq, max_freq)


def select_words_for_lexicon(filtered_words, n_tokens=100, random_seed=0):
    random.seed(random_seed)
    selection = []
    labels = list(set([x.label for x in filtered_words]))
    print(f'found {len(labels)} unique labels after filtering')
    label_to_token = {label: [] for label in labels}
    print('grouping tokens by label')
    for word in progressbar(filtered_words):
        label_to_token[word.label].append(word)
    print('selecting tokens for lexicon')
    for label in progressbar(labels):
        selection += random.sample(label_to_token[label], n_tokens)
    print(f'Selected {len(selection)} tokens for lexicon')
    print(f'Unique labels: {len(set([x.label for x in selection]))}')
    return selection


def load_words():
    cgn = stores.load_phraser_cgn_store()
    print(f'loading all words from cgn database {cgn}')
    return list(cgn.words.all())


def load_phones():
    return list(stores.load_phraser_cgn_store().phones.all())


def load_syllables():
    return list(stores.load_phraser_cgn_store().syllables.all())
