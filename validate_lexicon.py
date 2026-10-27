from pathlib import Path

from echoframe import store
import numpy as np
import speech_vector_search as svs
from progressbar import progressbar

import stores

def load_store_w2v(root='lexicon100_2466_cgn'):
    print(f'loading store from {root}')
    s = store.Store(root)
    if stores.cgn is not None:
        s.attach_phraser_store(stores.CGN_SOURCE_ID, stores.cgn)
    return s

def load_store_hubert1(root='lexicon100_2466_cgn_hubert'):
    print(f'loading store from {root}')
    s = store.Store(root)
    if stores.cgn is not None:
        s.attach_phraser_store(stores.CGN_SOURCE_ID, stores.cgn)
    return s

def load_metadata(store):
    print('loading echoframe metadata')
    metadata = store.metadatas
    return metadata

def load_metadata_layer(store, layer):
    print(f'loading echoframe metadata for layer {layer}')
    metadatas = load_metadata(store)
    layer_metadata = [x for x in metadatas if x.layer == layer]
    print(f'found {len(layer_metadata)} metadata entries for layer {layer}')
    return layer_metadata

def load_speech_vectors(metadatas, pooling='center'):
    '''Pool each token's stored word frames into one vector.'''
    if pooling not in ('center', 'mean'):
        raise ValueError(f'unknown pooling method {pooling}')
    print(f'loading {pooling} vectors for {len(metadatas)} metadata entries')
    vectors = []
    for m in progressbar(metadatas):
        v = m.load_payload()
        v = svs.util_math.pool_frames(v, pooling)
        vectors.append(v)
    print(f'loaded {len(vectors)} vectors')
    return np.asarray(vectors)
    
def make_vector_metadata(metadatas, name, directory = 'svs_data'):
    print(f'making vector metadata for {len(metadatas)} entries')
    cls = svs.metadata.VectorMetadata
    vmds = [cls(x.label, 'word', [x.echoframe_key], 0) for x in metadatas]
    svs_metadata = svs.metadata.VectorMetadatas(vmds, name, directory)
    return svs_metadata

def make_vector_search_index(vectors, vector_metadata):
    print(f'making vector search index for {len(vectors)} vectors')
    index = svs.search.VectorIndex(vectors, vector_metadata)
    return index

def make_vector_search_index_for_layer(store, layer, name, save=True,
    overwrite=False, pooling='center', directory='svs_data'):
    '''Build or load a token index; mean pooling uses a separate name.'''
    if pooling not in ('center', 'mean'):
        raise ValueError(f'unknown pooling method {pooling}')
    if pooling == 'mean' and not name.endswith('_mean'):
        name += '_mean'
    metadata = load_metadata_layer(store, layer)
    if not metadata:
        raise ValueError(f'no embeddings found for layer {layer}')
    vector_metadata = make_vector_metadata(metadata, name=name,
        directory=directory)
    vector_path = Path(directory) / f'{name}.npy'
    if vector_metadata.path.exists() and vector_path.exists() and not overwrite:
        print(f'loading existing index {name}')
        vector_index = svs.VectorIndex.load(directory, name)
        by_key = {md.echoframe_key: md for md in metadata}
        metadata = [by_key[md.source_echoframe_keys[0]]
            for md in vector_index.metadatas]
        return vector_index, metadata
    if save and not overwrite and (
        vector_metadata.path.exists() or vector_path.exists()):
        raise ValueError(f'incomplete index {name}; use overwrite=True')
    vectors = load_speech_vectors(metadata, pooling=pooling)
    vector_index = make_vector_search_index(vectors, vector_metadata)
    print('saving vector search index')
    if save: vector_index.save(overwrite=overwrite)
    echoframe_metadata = metadata
    return vector_index, echoframe_metadata

def make_vsi_for_layers(layers, model='w2v', save=True,
    overwrite=False, pooling='center', directory='svs_data'):
    if model == 'w2v':store = load_store_w2v()
    elif model == 'hubert1':store = load_store_hubert1()
    else:raise ValueError(f'unknown model {model}')
    print(f'making vector search indices for layer {layers} and model {model}')
    vsis = {}
    for layer in progressbar(layers):
        name = f'{model}_{layer}'
        vsi, emd = make_vector_search_index_for_layer(store, layer, name, save,
            overwrite, pooling=pooling, directory=directory)
        vsis[layer] = [vsi, emd]
    return vsis
   


