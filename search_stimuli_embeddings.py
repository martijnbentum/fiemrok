import speech_vector_search as svs
import stores
import fiemrok
import cgn_lexicon_vector_search as clvs

final_wav2vec2_model_name = 'wav2vec2_nl1_checkpoint-200000'
final_hubert_model_name = 'huibert_nl1_checkpoint-200000'
name_prefix = 'cgn_center_nucleus'

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
        
    
    
    


