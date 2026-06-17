import echoframe
import locations
import phraser
from progressbar import progressbar

from echoframe import segment_features

def load_echoframe_stimuli_store(path = locations.echoframe_stimuli_store_root):
    return echoframe.store.Store(path)

def compute_stimuli_embeddings_for_experiment(experiment, echoframe_store, 
    layers = [3,6,12]):
    for stimulus in progressbar(experiment.fillers):
        compute_stimulus_embeddings(echoframe_store, stimulus, layers = layers,
        tags = ['filler'])
    for stimulus in progressbar(experiment.targets):
        compute_stimulus_embeddings(echoframe_store, stimulus, layers = layers,
        tags = ['target'])

def compute_stimulus_embeddings(echoframe_store, stimulus, layers = [3,6,12],
    tags = None, model_name = 'wav2vec2_nl1_checkpoint-200000'):
    if not hasattr(stimulus, 'word'):
        m = f'stimulus {stimulus} does not have a word attribute'
        m += f'add phraser stimuli store (see stores.py) to experiment object'
        raise ValueError(m)
    word = stimulus.word
    segment_features.compute_embeddings(word, store = echoframe_store, 
        layers = layers, tags = tags, model_name = model_name)
    
