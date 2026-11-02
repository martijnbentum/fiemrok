'''Compute echoframe embeddings for experiment stimuli and the CGN lexicon.'''

from echoframe import segment_features
from progressbar import progressbar

import model_paths
import stores

default_layers = (3, 6, 9, 12)


def make_stimulus_embeddings(experiment, echoframe_store=None,
    layers=default_layers, model_name=model_paths.wav2vec2_nl1_name,
    gpu=False):
    '''Compute embeddings for the filler and target words of the experiment.
    experiment:         fiemrok.Experiment with a phraser stimuli store added
    echoframe_store:    defaults to the echoframe stimuli store
    '''
    if echoframe_store is None:
        echoframe_store = stores.load_echoframe_stimuli_store()
    for tag, stimuli in [('filler', experiment.fillers),
        ('target', experiment.targets)]:
        for stimulus in progressbar(stimuli):
            make_stimulus_embedding(stimulus, echoframe_store, layers=layers,
                tags=[tag], model_name=model_name, gpu=gpu)


def make_stimulus_embedding(stimulus, echoframe_store, layers=default_layers,
    tags=None, model_name=model_paths.wav2vec2_nl1_name, gpu=False):
    '''Compute embeddings for the phraser word of a single stimulus.'''
    if not hasattr(stimulus, 'word'):
        m = f'stimulus {stimulus} does not have a word attribute, '
        m += 'add phraser stimuli store (see stores.py) to experiment object'
        raise ValueError(m)
    segment_features.compute_embeddings(stimulus.word, store=echoframe_store,
        layers=layers, tags=tags, model_name=model_name, gpu=gpu)


def make_lexicon_embeddings(lexicon=None, echoframe_store=None,
    layers=default_layers, tags=('cgn',),
    model_name=model_paths.wav2vec2_nl1_name, gpu=False):
    '''Compute embeddings for the CGN word tokens of the lexicon.
    lexicon:            list of phraser words, defaults to
                        select_lexicon.make_lexicon()
    echoframe_store:    defaults to the echoframe CGN lexicon store
    '''
    if lexicon is None:
        from . import select_lexicon
        lexicon = select_lexicon.make_lexicon()
    if echoframe_store is None:
        echoframe_store = stores.load_echoframe_cgn_store()
    for word in progressbar(lexicon):
        segment_features.compute_embeddings(word, store=echoframe_store,
            layers=layers, tags=list(tags), model_name=model_name, gpu=gpu)
