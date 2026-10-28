'''Compute full-frame EchoFrame embeddings for sampled CGN markers.'''

from echoframe.batch_segment_features import compute_embeddings_batch

import cross_boundary_lexicon as cbl
import locations
import model_paths
import stores


def create_stores():
    '''Open both cross-boundary stores and register their 200000 checkpoints.
    Existing registrations are retained. The caller owns both open stores.
    '''
    wav2vec_store = stores.load_echoframe_cross_boundary_store(
        locations.cross_boundary_wav2vec_store)
    hubert_store = stores.load_echoframe_cross_boundary_store(
        locations.cross_boundary_hubert_store)
    wav2vec_model = 'wav2vec2_nl1_checkpoint-200000'
    hubert_model = 'huibert_nl1_checkpoint-200000'
    if wav2vec_store.load_model_metadata(wav2vec_model) is None:
        checkpoint = model_paths.get_pretrained_wav2vec2_model_path()
        wav2vec_store.register_model(wav2vec_model,
            local_path=checkpoint.as_posix(), language='Dutch', size='base',
            architecture='wav2vec2')
    if hubert_store.load_model_metadata(hubert_model) is None:
        checkpoint = model_paths.get_pretrained_hubert_model_path()
        hubert_store.register_model(hubert_model,
            local_path=checkpoint.as_posix(), language='Dutch', size='base',
            architecture='hubert')
    return wav2vec_store, hubert_store


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
