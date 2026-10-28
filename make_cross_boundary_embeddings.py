'''Compute full-frame EchoFrame embeddings for sampled CGN markers.'''

from echoframe.batch_segment_features import compute_embeddings_batch

import cross_boundary_lexicon as cbl
import locations
import model_paths
import stores


def compute_embeddings_for_hubert(**kwargs):
    '''Compute HuBERT embeddings; return the open store for the caller to close.'''
    embedding_store = stores.load_echoframe_cross_boundary_store(
        locations.cross_boundary_hubert_store)
    compute_embeddings(embedding_store, **kwargs)
    return embedding_store


def compute_embeddings_for_wv2(**kwargs):
    '''Compute wav2vec embeddings; return the open store for the caller to close.'''
    embedding_store = stores.load_echoframe_cross_boundary_store(
        locations.cross_boundary_wav2vec_store)
    compute_embeddings(embedding_store, **kwargs)
    return embedding_store


def compute_embeddings(embedding_store, layers=(3, 6, 9, 12), collar=500,
    gpu=False, batch_size=32, verbose=True):
    '''Compute missing full-frame matrices for the saved marker sample.
    The open store supplies CGN and its sole registered model. The caller
    owns its lifetime. Collar is context on each side in milliseconds.
    '''
    cgn = embedding_store.load_phraser_store(stores.CGN_SOURCE_ID)
    markers = cbl.load_markers(cgn, keys_path=locations.cross_boundary_marker_keys)
    model_name = embedding_store.model_registry.model_metadatas[0].model_name
    compute_embeddings_batch(markers, layers=layers, model_name=model_name,
        store=embedding_store, collar=collar, gpu=gpu, batch_size=batch_size,
        tags=['cgn', 'cross_boundary'], verbose=verbose)


def create_store(path, model_name, architecture):
    '''Open a store and register its version 1/checkpoint 200000 model if absent.'''
    embedding_store = stores.load_echoframe_cross_boundary_store(path)
    if embedding_store.load_model_metadata(model_name) is None:
        checkpoint = model_paths.get_pretrained_model(architecture,
            checkpoint=200_000, version=1)
        embedding_store.register_model(model_name,
            local_path=checkpoint.as_posix(), language='Dutch', size='base',
            architecture=architecture)
    return embedding_store


def create_wv2_and_hubert():
    '''Return the configured wav2vec and HuBERT stores; the caller closes them.'''
    wav2vec_store = create_store(locations.cross_boundary_wav2vec_store,
        'wav2vec2_nl1_checkpoint-200000', 'wav2vec2')
    hubert_store = create_store(locations.cross_boundary_hubert_store,
        'huibert_nl1_checkpoint-200000', 'hubert')
    return wav2vec_store, hubert_store
