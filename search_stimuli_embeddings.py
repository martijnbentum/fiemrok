import speech_vector_search as svs
import stores
import fiemrok
import cgn_lexicon_vector_search as clvs
from progressbar import progressbar

final_wav2vec2_model_name = 'wav2vec2_nl1_checkpoint-200000'
final_hubert_model_name = 'huibert_nl1_checkpoint-200000'
name_prefix = 'cgn_center_nucleus'

def make_all_compare_embeddings(layers = [3,6,9,12], experiment = None):
    if experiment is None: experiment = load_experiment()
    ces = []
    output = {}
    for model_name in [final_wav2vec2_model_name, final_hubert_model_name]:
        output[model_name] = {}
        for layer in layers:
            output[model_name][layer] = {}
            ce = CompareEmbeddings(layer = layer, model_name = model_name, 
                experiment = experiment)
            ce.compute_results()
            ces.append(ce)
            output[model_name][layer]['aligned'] = ce.aligned_accuracies
            output[model_name][layer]['misaligned'] = ce.misaligned_accuracies
    return output, ces
    

class CompareEmbeddings:
    def __init__(self, layer = 6, model_name = final_wav2vec2_model_name,
        experiment = None, search_vector = None, search_depth = 10):
        self.layer = layer
        self.model_name = model_name
        if experiment is None: experiment = load_experiment()
        self.experiment = experiment
        if search_vector is None:
            search_vector = clvs.load_search_vector_index(layer, name_prefix,
                model_name = model_name)
        self.search_vector = search_vector
        self.search_depth = search_depth
        self._set_aligned_and_misaligned()

    def __repr__(self):
        m = f'CompareEmbeddings(layer = {self.layer}, '
        m += f'model_name = {self.model_name}, '
        m += f'search_depth = {self.search_depth})'
        return m

    def _set_aligned_and_misaligned(self):
        self.aligned = []
        self.misaligned = []
        for trial in self.experiment.target_trials:
            if trial.position != 'Final': continue
            for stimulus in trial.stimuli:
                if stimulus.word_type != 'word': continue
                if stimulus.aligned_type == 'aligned':
                    self.aligned.append(stimulus)
                if stimulus.aligned_type == 'misaligned':
                    self.misaligned.append(stimulus)

    def compute_results(self):
        self.aligned_results = []
        self.misaligned_results = []
        for stimulus in self.aligned:
            result = self._compute_result_for_stimulus(stimulus)
            self.aligned_results.append(result)
        for stimulus in self.misaligned:
            result = self._compute_result_for_stimulus(stimulus)
            self.misaligned_results.append(result)

    def _compute_result_for_stimulus(self, stimulus):
        check_stimulus_ok(stimulus)
        speech_vector = load_center_frame_nucleus_speech_vector(stimulus, 
            layer = self.layer, model_name = self.model_name)
        result = self.search_vector.query(speech_vector, self.search_depth)
        return result

    @property
    def aligned_accuracies(self):
        accuracies = []
        for stimulus, result in zip(self.aligned, self.aligned_results):
            accuracy = compute_accuracy(result, stimulus.trial.word)
            accuracies.append(accuracy)
        return accuracies

    @property
    def misaligned_accuracies(self):
        accuracies = []
        for stimulus, result in zip(self.misaligned, self.misaligned_results):
            accuracy = compute_accuracy(result, stimulus.trial.word)
            accuracies.append(accuracy)
        return accuracies

    @property
    def mean_aligned_accuracy(self):
        accuracies = self.aligned_accuracies
        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    @property
    def mean_misaligned_accuracy(self):
        accuracies = self.misaligned_accuracies
        return sum(accuracies) / len(accuracies) if accuracies else 0.0

def compute_accuracy(result, label):
    matches = 0
    metadatas = result['metadatas']
    for md, score in zip(metadatas, result['scores']):
        if md.label == label:
            matches += 1
    return matches / len(metadatas) if metadatas else 0.0
    
    
def print_neighbors(label, result, precision=2):
    """Print `label` (blue) followed by each neighbor's metadata label with
    its score (light grey) in brackets. Neighbor labels matching `label` are
    shown in green. Accuracy (fraction of matches) is printed at the end in
    red between [], e.g.:  loon: lijn (0.72), loon (0.71), ... [0.10]
    """
    BLUE = '\033[34m'
    GREY = '\033[90m'   # bright black = light grey
    GREEN = '\033[32m'
    RED = '\033[31m'
    RESET = '\033[0m'

    metadatas = result['metadatas']
    parts = []
    for md, score in zip(metadatas, result['scores']):
        if md.label == label:
            name = f'{GREEN}{md.label}{RESET}'
        else:
            name = md.label
        parts.append(f'{name} {GREY}({score:.{precision}f}){RESET}')

    accuracy = compute_accuracy(result, label)
    print(f'{BLUE}{label}{RESET}: ' + ', '.join(parts)
          + f' {RED}[{accuracy:.{precision}f}]{RESET}')

def load_experiment():
    stimuli = stores.load_phraser_stimuli_store()
    efs = stores.load_echoframe_stimuli_store()
    e = fiemrok.Experiment(phraser_store = stimuli, echoframe_store = efs)
    return e

def experiment_stimulus_to_embedding(stimulus, layer = 6, 
    model_name = final_wav2vec2_model_name):
    if not hasattr(stimulus, 'metadatas'):
        m = f'stimulus {stimulus} has no echoframe_metadata'
        m += f' add echoframe stimuli store to experiment'
        raise ValueError(m)
    metadata = clvs.filter_metadatas(stimulus.metadatas, layer = layer, 
        model_name = model_name)
    if len(metadata) != 1:
        m = f'stimulus {stimulus} has {len(metadata)} entries: '
        m = f' {metadata} for layer {layer} and model {model_name}'
        raise ValueError(m) 
    metadata = metadata[0]
    return metadata.load_embedding()

def load_sub_embedding_for_nucleus(stimulus, layer = 6, 
    model_name = final_wav2vec2_model_name):
    check_stimulus_ok(stimulus)
    embedding = experiment_stimulus_to_embedding(stimulus, layer = layer, 
        model_name = model_name)
    nucleus = stimulus.word.syllables[1].nucleus[0]
    print(f'loading sub embedding for nucleus:\n {nucleus!r}')
    print(f'in syllable:\n {stimulus.word.syllables[1]!r}')
    print(f'in word:\n {stimulus.word!r}')
    print(f'in stimulus:\n {stimulus!r}')
    sub_embedding = embedding.sub_embedding(nucleus)
    return sub_embedding

def load_center_frame_nucleus_speech_vector(stimulus, layer = 6,
    model_name = final_wav2vec2_model_name):
    sub_embedding = load_sub_embedding_for_nucleus(stimulus, layer = layer, 
        model_name = model_name)
    speech_vector = svs.util_math.pool_frames(sub_embedding.data, 'center')
    return speech_vector


def check_stimulus_ok(stimulus):
    if len(stimulus.word.syllables) != 2:
        m = f'stimulus {stimulus} should have a word with 2 syllables, '
        m += f'but has {len(word.syllables)} {word.syllables!r}'
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
    nucleus = stimulus.word.syllables[1].nucleus
    if len(nucleus) != 1:
        m = f'stimulus {stimulus} should have a single segment nucleus, '
        m += f'but has {len(nucleus)} {nucleus!r}'
        raise ValueError(m)
        
    
    
    


