'''Build pooled speech-vector indexes from saved token selections.'''

from pathlib import Path

import numpy as np
from progressbar import progressbar
import speech_vector_search as svs

from . import lv_locations


def load_metadata(store):
    '''Load the embedding store's metadata records.'''
    return store.metadatas


def load_metadata_layer(store, layer, selected_keys=None, model_name=None,
    collar=500):
    '''Return hidden states in saved selection order for one model/layer.'''
    records = [md for md in load_metadata(store)
        if md.layer == layer and md.output_type == 'hidden_state'
        and md.collar == collar
        and (model_name is None or md.model_name == model_name)]
    if selected_keys is None: return records
    selected_keys = list(selected_keys)
    requested = set(selected_keys)
    by_key = {}
    for md in records:
        if md.phraser_key not in requested: continue
        if md.phraser_key in by_key:
            raise ValueError('multiple embeddings for one selected token')
        by_key[md.phraser_key] = md
    selected = []
    for key in selected_keys:
        if key not in by_key:
            message = f'missing embedding for selected token {key.hex()}'
            raise ValueError(message)
        selected.append(by_key[key])
    return selected


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
    
def make_vector_metadata(metadatas, name, directory=lv_locations.svs_data,
    unit_type='word'):
    print(f'making vector metadata for {len(metadatas)} entries')
    cls = svs.metadata.VectorMetadata
    vmds = [cls(x.label, unit_type, [x.echoframe_key], 0) for x in metadatas]
    svs_metadata = svs.metadata.VectorMetadatas(vmds, name, str(directory))
    return svs_metadata

def make_vector_search_index(vectors, vector_metadata):
    print(f'making vector search index for {len(vectors)} vectors')
    index = svs.search.VectorIndex(vectors, vector_metadata)
    return index

def make_vector_search_index_for_layer(store, layer, name, selected_keys,
    model_name, save=True, overwrite=False, pooling='center',
    directory=lv_locations.svs_data, unit_type='word', collar=500):
    '''Build a fresh index for the ordered selection and requested condition.
    selected_keys: ordered Phraser word or marker keys
    model_name:    registered model checkpoint name
    pooling:       center or mean over each token's stored frames
    name:          complete index name, including its pooling condition
    '''
    if pooling not in ('center', 'mean'):
        raise ValueError(f'unknown pooling method {pooling}')
    if save and not overwrite:
        for suffix in ('.json', '.npy'):
            path = Path(directory) / f'{name}{suffix}'
            if path.exists(): raise FileExistsError(path)
    metadata = load_metadata_layer(store, layer, selected_keys, model_name,
        collar=collar)
    if not metadata:
        raise ValueError(f'no embeddings selected for layer {layer}')
    vectors = load_speech_vectors(store, metadata, pooling=pooling)
    vector_metadata = make_vector_metadata(metadata, name=name,
        directory=directory, unit_type=unit_type)
    index = make_vector_search_index(vectors, vector_metadata)
    if save:
        Path(directory).mkdir(parents=True, exist_ok=True)
        index.save(overwrite=overwrite)
    return index, metadata


def make_vsi_for_layers(layers, embedding_store, selected_keys, model_name,
    name_prefix='', save=True, overwrite=False, pooling='center',
    directory=lv_locations.svs_data, unit_type='word', collar=500):
    '''Build one pooled index per layer; caller owns the embedding store.'''
    selected_keys = list(selected_keys)
    indexes = {}
    for layer in progressbar(layers):
        name = f'{name_prefix}{layer}_{pooling}'
        indexes[layer] = make_vector_search_index_for_layer(embedding_store,
            layer, name, selected_keys, model_name, save=save,
            overwrite=overwrite, pooling=pooling, directory=directory,
            unit_type=unit_type, collar=collar)
    return indexes
