from echoframe import store
import numpy as np
import speech_vector_search as svs
from progressbar import progressbar
import stores

final_wav2vec2_model_name = 'wav2vec2_nl1_checkpoint-200000'
final_hubert_model_name = 'huibert_nl1_checkpoint-200000'


def filter_metadatas(metadatas, layer = 6, 
    model_name = final_wav2vec2_model_name):
    metadatas = [x for x in metadatas if x.layer == layer]
    metadatas = [x for x in metadatas if x.model_name == model_name]
    m = f'found {len(metadatas)} metadata entries for layer {layer}'
    m += f' and model {model_name}'
    print(m)
    return metadatas

def metadata_to_center_frame_nucleus_speech_vector(metadata):
    '''Returns the center frame of the nucleus of a single syllable word.
    metadata    echoframe metadata for a word, word must have a single syllable
    '''
    word = metadata.phraser_object
    if not word.object_type == 'Word':
        raise ValueError(f'metadata {metadata} is not for a Word object')
    if len(word.syllables) != 1:
        raise ValueError(f'word {word} is not a single syllable word')
    nucleus = word.syllables[0].nucleus
    if len(nucleus) !=1:
        raise NotImplementedError(f'nucleus {nucleus} is not a single segment')
    nucleus = nucleus[0]
    embedding = metadata.load_embedding()
    nucleus_embedding = embedding.sub_embedding(nucleus)
    speech_vector = svs.util_math.pool_frames(nucleus_embedding.data, 'center')
    return speech_vector

def metadatas_to_center_frame_nucleus_speech_vectors(metadatas):
    '''Returns the center frame of the nucleus of a single syllable word.
    metadatas       list of echoframe metadata for words, 
                    words must have a single syllable
    '''
    vectors = []
    errors = []
    succesfull_metadatas = []
    for m in progressbar(metadatas):
        try:v = metadata_to_center_frame_nucleus_speech_vector(m)
        except Exception as e:
            errors.append((m, e))
            continue
        vectors.append(v)
        succesfull_metadatas.append(m)
    return np.asarray(vectors), succesfull_metadatas, errors
    
def make_vector_metadata(metadatas, name, directory = 'svs_experiment_data'):
    print(f'making vector metadata for {len(metadatas)} entries')
    cls = svs.metadata.VectorMetadata
    vmds = [cls(x.label, 'word', [x.echoframe_key], 0) for x in metadatas]
    svs_metadata = svs.metadata.VectorMetadatas(vmds, name, directory)
    return svs_metadata

def make_vector_search_index(vectors, vector_metadata):
    print(f'making vector search index for {len(vectors)} vectors')
    index = svs.search.VectorIndex(vectors, vector_metadata)
    return index

def make_vector_search_index_for_layer(store, layer, name, 
    model_name = final_wav2vec2_model_name, save = True, overwrite = False):
    metadatas = filter_metadatas(store.metadatas, layer = layer, 
        model_name = model_name)
    vector_metadata = make_vector_metadata(metadatas[:3], name = name)
    if vector_metadata.path.exists() and not overwrite: 
        print(f'files already exist {vector_metadata.path}, doing nothing ')
        return
    vectors, mds, e = metadatas_to_center_frame_nucleus_speech_vectors(metadatas)
    metadatas, errors = mds, e
    vector_metadata = make_vector_metadata(metadatas, name = name)
    m = f'found {len(metadatas)} succesfull metadatas '
    m += f'and {len(errors)} errors'
    print(m)
    vector_index = make_vector_search_index(vectors, vector_metadata)
    print('saving vector search index')
    if save: vector_index.save()
    return vector_index, metadatas, errors



# OLD

def OLD_load_speech_vectors(metadatas):
    print(f'loading center frame vectors for {len(metadatas)} metadata entries')
    vectors = []
    for m in progressbar(metadatas):
        v = m.load_payload()
        v = svs.util_math.pool_frames(v, 'center')
        vectors.append(v)
    print(f'loaded {len(vectors)} vectors')
    return np.asarray(vectors)

def OLD_make_vsi_for_layers(layers, model = 'w2v', save = True, 
    overwrite = False):
    if model == 'w2v':store = load_store_w2v()
    elif model == 'hubert1':store = load_store_hubert1()
    else:raise ValueError(f'unknown model {model}')
    print(f'making vector search indices for layer {layers} and model {model}')
    vsis = {}
    for layer in progressbar(layers):
        name = f'{model}_{layer}'
        vsi, emd = make_vector_search_index_for_layer(store, layer, name, save, 
            overwrite)
        vsis[layer] = [vsi, emd]
    return vsis
   

def OLD_load_store_w2v(root = 'lexicon100_2466_cgn'):
    print(f'loading store from {root}')
    s= store.Store(root)
    return s

def OLD_load_store_hubert1(root = 'lexicon100_2466_cgn_hubert'):
    print(f'loading store from {root}')
    s= store.Store(root)
    return s

