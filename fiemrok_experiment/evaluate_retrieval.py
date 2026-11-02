'''Query stimulus nucleus vectors against the CGN lexicon vector index.'''

import model_paths
import stores
from . import fiemrok
from . import make_lexicon_vector_index as mlvi

name_prefix = 'cgn_center_nucleus'
model_names = (model_paths.wav2vec2_nl1_name, model_paths.hubert_nl1_name)
aligned_types = ('aligned', 'misaligned')


def make_all_compare_embeddings(layers=(3, 6, 9, 12), experiment=None):
    '''Return accuracies per model, layer and aligned type, and the
    CompareEmbeddings objects that produced them.
    '''
    if experiment is None: experiment = load_experiment()
    ces = []
    output = {}
    for model_name in model_names:
        output[model_name] = {}
        for layer in layers:
            ce = CompareEmbeddings(layer=layer, model_name=model_name,
                experiment=experiment)
            ce.compute_results()
            ces.append(ce)
            output[model_name][layer] = {x: ce.accuracies(x)
                for x in aligned_types}
    return output, ces


class CompareEmbeddings:
    def __init__(self, layer=6, model_name=model_paths.wav2vec2_nl1_name,
        experiment=None, search_vector=None, search_depth=10):
        self.layer = layer
        self.model_name = model_name
        if experiment is None: experiment = load_experiment()
        self.experiment = experiment
        if search_vector is None:
            search_vector = mlvi.load_search_vector_index(layer, name_prefix,
                model_name=model_name)
        self.search_vector = search_vector
        self.search_depth = search_depth
        self.stimuli = {x: experiment.select(position='Final',
            word_type='word', aligned_type=x) for x in aligned_types}
        self.results = {}

    def __repr__(self):
        m = f'CompareEmbeddings(layer = {self.layer}, '
        m += f'model_name = {self.model_name}, '
        m += f'search_depth = {self.search_depth})'
        return m

    def compute_results(self):
        for aligned_type, stimuli in self.stimuli.items():
            results = [self._compute_result_for_stimulus(x) for x in stimuli]
            self.results[aligned_type] = results

    def _compute_result_for_stimulus(self, stimulus):
        speech_vector = load_center_frame_nucleus_speech_vector(stimulus,
            layer=self.layer, model_name=self.model_name)
        return self.search_vector.query(speech_vector, self.search_depth)

    def accuracies(self, aligned_type):
        '''Return the accuracy per stimulus of aligned or misaligned.'''
        pairs = zip(self.stimuli[aligned_type], self.results[aligned_type])
        return [compute_accuracy(result, stimulus.trial.word)
            for stimulus, result in pairs]

    def mean_accuracy(self, aligned_type):
        accuracies = self.accuracies(aligned_type)
        return sum(accuracies) / len(accuracies) if accuracies else 0.0


def compute_accuracy(result, label):
    '''Return the fraction of retrieved neighbours whose label matches.'''
    metadatas = result['metadatas']
    matches = sum(1 for md in metadatas if md.label == label)
    return matches / len(metadatas) if metadatas else 0.0


def print_neighbors(label, result, precision=2):
    '''Print `label` (blue) followed by each neighbor's metadata label with
    its score (light grey) in brackets. Neighbor labels matching `label` are
    shown in green. Accuracy (fraction of matches) is printed at the end in
    red between [], e.g.:  loon: lijn (0.72), loon (0.71), ... [0.10]
    '''
    BLUE = '\033[34m'
    GREY = '\033[90m'   # bright black = light grey
    GREEN = '\033[32m'
    RED = '\033[31m'
    RESET = '\033[0m'

    metadatas = result['metadatas']
    parts = []
    for md, score in zip(metadatas, result['scores']):
        if md.label == label: name = f'{GREEN}{md.label}{RESET}'
        else: name = md.label
        parts.append(f'{name} {GREY}({score:.{precision}f}){RESET}')

    accuracy = compute_accuracy(result, label)
    print(f'{BLUE}{label}{RESET}: ' + ', '.join(parts)
        + f' {RED}[{accuracy:.{precision}f}]{RESET}')


def load_experiment():
    stimuli = stores.load_phraser_stimuli_store()
    efs = stores.load_echoframe_stimuli_store()
    return fiemrok.Experiment(phraser_store=stimuli, echoframe_store=efs)


def experiment_stimulus_to_embedding(stimulus, layer=6,
    model_name=model_paths.wav2vec2_nl1_name):
    if not hasattr(stimulus, 'metadatas'):
        m = f'stimulus {stimulus} has no echoframe_metadata,'
        m += ' add echoframe stimuli store to experiment'
        raise ValueError(m)
    metadata = mlvi.filter_metadatas(stimulus.metadatas, layer=layer,
        model_name=model_name)
    if len(metadata) != 1:
        m = f'stimulus {stimulus} has {len(metadata)} entries: '
        m += f' {metadata} for layer {layer} and model {model_name}'
        raise ValueError(m)
    return metadata[0].load_embedding()


def load_sub_embedding_for_nucleus(stimulus, layer=6,
    model_name=model_paths.wav2vec2_nl1_name):
    '''Return the final-syllable nucleus sub embedding and print its context.'''
    check_stimulus_ok(stimulus)
    embedding = experiment_stimulus_to_embedding(stimulus, layer=layer,
        model_name=model_name)
    word = stimulus.word
    nucleus = word.syllables[1].nucleus[0]
    print(f'loading sub embedding for nucleus:\n {nucleus!r}')
    print(f'in syllable:\n {word.syllables[1]!r}')
    print(f'in word:\n {word!r}')
    print(f'in stimulus:\n {stimulus!r}')
    return mlvi.nucleus_sub_embedding(embedding, word, 1)


def load_center_frame_nucleus_speech_vector(stimulus, layer=6,
    model_name=model_paths.wav2vec2_nl1_name):
    '''Return the centre frame of the final-syllable nucleus of a stimulus.'''
    check_stimulus_ok(stimulus)
    embedding = experiment_stimulus_to_embedding(stimulus, layer=layer,
        model_name=model_name)
    return mlvi.center_frame_nucleus_vector(embedding, stimulus.word, 1)


def check_stimulus_ok(stimulus):
    syllables = stimulus.word.syllables
    if len(syllables) != 2:
        m = f'stimulus {stimulus} should have a word with 2 syllables, '
        m += f'but has {len(syllables)} {syllables!r}'
        raise ValueError(m)
    if stimulus.position != 'Final':
        m = f'stimulus {stimulus} should have position Final, '
        m += f'but has {stimulus.position}'
        raise ValueError(m)
    if stimulus.word_type != 'word':
        m = f'stimulus {stimulus} should have word_type word, '
        m += f'but has {stimulus.word_type}'
        raise ValueError(m)
    if not stimulus.target:
        raise ValueError('stimulus should be a target')
    nucleus = syllables[1].nucleus
    if len(nucleus) != 1:
        m = f'stimulus {stimulus} should have a single segment nucleus, '
        m += f'but has {len(nucleus)} {nucleus!r}'
        raise ValueError(m)
