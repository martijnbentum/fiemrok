'''Build the CGN lexicon vector index that stimuli are compared against.'''

import numpy as np
from progressbar import progressbar
import speech_vector_search as svs

import locations
import model_paths


def filter_metadatas(metadatas, layer=6,
    model_name=model_paths.wav2vec2_nl1_name):
    metadatas = [x for x in metadatas if x.layer == layer]
    metadatas = [x for x in metadatas if x.model_name == model_name]
    m = f'found {len(metadatas)} metadata entries for layer {layer}'
    m += f' and model {model_name}'
    print(m)
    return metadatas


def nucleus_sub_embedding(embedding, word, syllable_index):
    '''Return the part of embedding covering the first nucleus phone.'''
    nucleus = word.syllables[syllable_index].nucleus[0]
    return embedding.sub_embedding(nucleus)


def center_frame_nucleus_vector(embedding, word, syllable_index):
    '''Return the centre frame of a syllable nucleus in word.'''
    sub_embedding = nucleus_sub_embedding(embedding, word, syllable_index)
    return svs.util_math.pool_frames(sub_embedding.data, 'center')


def metadata_to_center_frame_nucleus_speech_vector(metadata):
    '''Returns the center frame of the nucleus of a single syllable word.
    metadata:   echoframe metadata for a word, word must have a single syllable
    '''
    word = metadata.phraser_object
    if not word.object_type == 'Word':
        raise ValueError(f'metadata {metadata} is not for a Word object')
    if len(word.syllables) != 1:
        raise ValueError(f'word {word} is not a single syllable word')
    nucleus = word.syllables[0].nucleus
    if len(nucleus) != 1:
        raise NotImplementedError(f'nucleus {nucleus} is not a single segment')
    embedding = metadata.load_embedding()
    return center_frame_nucleus_vector(embedding, word, 0)


def metadatas_to_center_frame_nucleus_speech_vectors(metadatas):
    '''Returns the center frame of the nucleus of a single syllable word.
    metadatas:  list of echoframe metadata for words,
                words must have a single syllable
    '''
    vectors = []
    errors = []
    succesfull_metadatas = []
    for m in progressbar(metadatas):
        try: v = metadata_to_center_frame_nucleus_speech_vector(m)
        except Exception as e:
            errors.append((m, e))
            continue
        vectors.append(v)
        succesfull_metadatas.append(m)
    return np.asarray(vectors), succesfull_metadatas, errors


def make_vector_metadata(metadatas, name,
    directory=locations.svs_fiemrok_experiment_data):
    print(f'making vector metadata for {len(metadatas)} entries')
    cls = svs.metadata.VectorMetadata
    vmds = [cls(x.label, 'word', [x.echoframe_key], 0) for x in metadatas]
    return svs.metadata.VectorMetadatas(vmds, name, directory)


def make_vector_search_index(vectors, vector_metadata):
    print(f'making vector search index for {len(vectors)} vectors')
    return svs.search.VectorIndex(vectors, vector_metadata)


def make_vector_search_index_name(layer, model_name, name_prefix=''):
    return f'{name_prefix}_layer-{layer}_model-{model_name}'


def load_search_vector_index(layer, name_prefix='',
    directory=locations.svs_fiemrok_experiment_data,
    model_name=model_paths.wav2vec2_nl1_name):
    name = make_vector_search_index_name(layer, model_name, name_prefix)
    return svs.VectorIndex.load(directory, name)


def make_vector_search_index_for_layer(store, layer, name_prefix='',
    directory=locations.svs_fiemrok_experiment_data,
    model_name=model_paths.wav2vec2_nl1_name, save=True, overwrite=False):
    name = make_vector_search_index_name(layer, model_name, name_prefix)
    metadatas = filter_metadatas(store.metadatas, layer=layer,
        model_name=model_name)
    path = svs.metadata.make_metadata_path(directory, name)
    if path.exists() and not overwrite:
        print(f'files already exist {path}, doing nothing ')
        return
    output = metadatas_to_center_frame_nucleus_speech_vectors(metadatas)
    vectors, metadatas, errors = output
    vector_metadata = make_vector_metadata(metadatas, name=name,
        directory=directory)
    m = f'found {len(metadatas)} succesfull metadatas '
    m += f'and {len(errors)} errors'
    print(m)
    vector_index = make_vector_search_index(vectors, vector_metadata)
    print('saving vector search index')
    if save: vector_index.save()
    return vector_index, metadatas, errors
