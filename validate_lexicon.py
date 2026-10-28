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

def load_speech_vectors(store, metadatas, pooling='center'):
    '''Return ordered token vectors, opening each HDF5 shard once.
    store:      embedding store containing the requested metadata
    metadatas:  token metadata in the desired vector order
    pooling:    center or mean, using Echoframe's existing arithmetic
    '''
    if pooling not in ('center', 'mean'):
        raise ValueError(f'unknown pooling method {pooling}')
    print(f'loading {pooling} vectors for {len(metadatas)} metadata entries')
    keys = [metadata.echoframe_key for metadata in metadatas]
    if not keys: return np.asarray([], dtype=float)
    vectors = store.load_many_frames(keys, frame=pooling, keep_missing=True)
    if len(vectors) != len(keys) or any(v is None for v in vectors):
        raise ValueError('missing vectors for requested metadata')
    print(f'loaded {len(vectors)} vectors')
    return np.asarray(vectors, dtype=float)
    
def make_vector_metadata(metadatas, name, directory = 'svs_data',
    unit_type='word'):
    print(f'making vector metadata for {len(metadatas)} entries')
    cls = svs.metadata.VectorMetadata
    vmds = [cls(x.label, unit_type, [x.echoframe_key], 0) for x in metadatas]
    svs_metadata = svs.metadata.VectorMetadatas(vmds, name, directory)
    return svs_metadata

def make_vector_search_index(vectors, vector_metadata):
    print(f'making vector search index for {len(vectors)} vectors')
    index = svs.search.VectorIndex(vectors, vector_metadata)
    return index

def make_vector_search_index_for_layer(store, layer, name, save=True,
    overwrite=False, pooling='center', directory='svs_data',
    return_metadata=True, unit_type='word'):
    '''Return (index, metadata); mean pooling uses a separate index name.
    return_metadata: False returns None for metadata and skips its loading
                     when an existing complete index can be reused
    unit_type:       category for newly built vector metadata; defaults to word
    '''
    if pooling not in ('center', 'mean'):
        raise ValueError(f'unknown pooling method {pooling}')
    if pooling == 'mean' and not name.endswith('_mean'):
        name += '_mean'
    metadata_path = svs.metadata.make_metadata_path(directory, name)
    vector_path = Path(directory) / f'{name}.npy'
    metadata_exists = metadata_path.exists()
    vectors_exist = vector_path.exists()
    if metadata_exists and vectors_exist and not overwrite:
        print(f'loading existing index {name}')
        vector_index = svs.VectorIndex.load(directory, name)
        if not return_metadata: return vector_index, None
        metadata = load_metadata_layer(store, layer)
        by_key = {md.echoframe_key: md for md in metadata}
        metadata = [by_key[md.source_echoframe_keys[0]]
            for md in vector_index.metadatas]
        return vector_index, metadata
    if save and not overwrite and (metadata_exists or vectors_exist):
        raise ValueError(f'incomplete index {name}; use overwrite=True')
    metadata = load_metadata_layer(store, layer)
    if not metadata:
        raise ValueError(f'no embeddings found for layer {layer}')
    vectors = load_speech_vectors(store, metadata, pooling=pooling)
    vector_metadata = make_vector_metadata(metadata, name=name,
        directory=directory, unit_type=unit_type)
    vector_index = make_vector_search_index(vectors, vector_metadata)
    if save:
        print('saving vector search index')
        vector_index.save(overwrite=overwrite)
    return vector_index, metadata if return_metadata else None

def make_vsi_for_layers(layers, model='w2v', save=True,
    overwrite=False, pooling='center', directory='svs_data',
    embedding_store=None, name_prefix='', unit_type='word'):
    '''Build layer indexes, using the original word stores/names by default.
    Supply embedding_store and name_prefix to index another dataset.
    The caller owns a supplied store; this function does not close it.
    '''
    if embedding_store is None:
        if model == 'w2v':embedding_store = load_store_w2v()
        elif model == 'hubert1':embedding_store = load_store_hubert1()
        else:raise ValueError(f'unknown model {model}')
    print(f'making vector search indices for layer {layers} and model {model}')
    vsis = {}
    for layer in progressbar(layers):
        name = f'{name_prefix}{model}_{layer}'
        vsi, emd = make_vector_search_index_for_layer(embedding_store, layer,
            name, save, overwrite, pooling=pooling, directory=directory,
            unit_type=unit_type)
        vsis[layer] = [vsi, emd]
    return vsis
   


