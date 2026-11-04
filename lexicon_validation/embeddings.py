'''Compute full-frame Echoframe embeddings for selected words and markers.'''

from echoframe import Store
from echoframe.batch_segment_features import compute_embeddings_batch

import model_paths
from . import lv_locations
from . import lv_stores


MODELS = {'w2v': (model_paths.wav2vec2_nl1_name, 'wav2vec2'),
    'hubert': (model_paths.hubert_nl1_name, 'hubert')}


def compute_embeddings_for_hubert(items, condition='word', n_types=None,
    workspace=lv_locations.workspace, **kwargs):
    '''Compute HuBERT embeddings; caller closes the returned store.'''
    return _compute_for_model(items, 'hubert', condition, n_types, workspace,
        **kwargs)


def compute_embeddings_for_wv2(items, condition='word', n_types=None,
    workspace=lv_locations.workspace, **kwargs):
    '''Compute Wav2Vec embeddings; caller closes the returned store.'''
    return _compute_for_model(items, 'w2v', condition, n_types, workspace,
        **kwargs)


def _compute_for_model(items, model, condition, n_types, workspace, **kwargs):
    items = list(items)
    if not items: raise ValueError('no selected items to embed')
    if n_types is None: n_types = len({item.label for item in items})
    path = lv_locations.store_path(model, condition, n_types,
        workspace=workspace)
    model_name, architecture = MODELS[model]
    embedding_store = create_store(path, model_name, architecture,
        cgn=items[0].store)
    try:
        compute_embeddings(embedding_store, items, condition=condition,
            model_name=model_name, **kwargs)
    except Exception:
        embedding_store.close()
        raise
    return embedding_store


def compute_embeddings(embedding_store, items, condition='word',
    layers=(3, 6, 9, 12), collar=500, gpu=False, batch_size=32, verbose=True,
    model_name=None):
    '''Compute selected items with context on each side in milliseconds.
    items:           words or markers in saved selection order
    condition:       word or cross_boundary, used as an embedding tag
    embedding_store: caller-owned store with CGN attached
    '''
    if condition not in ('word', 'cross_boundary'):
        raise ValueError('condition must be word or cross_boundary')
    if model_name is None:
        models = embedding_store.model_registry.model_metadatas
        if len(models) != 1:
            raise ValueError('provide model_name for a store without one model')
        model_name = models[0].model_name
    compute_embeddings_batch(items, layers=layers, model_name=model_name,
        store=embedding_store, collar=collar, gpu=gpu, batch_size=batch_size,
        tags=['cgn', condition], verbose=verbose)


def create_store(path, model_name, architecture, cgn):
    '''Open a CGN store and register the version 1/checkpoint 200000 model.'''
    embedding_store = Store(path)
    try:
        embedding_store.attach_phraser_store(lv_stores.CGN_SOURCE_ID, cgn)
        if embedding_store.load_model_metadata(model_name) is None:
            checkpoint = model_paths.get_pretrained_model(architecture,
                checkpoint=200_000, version=1)
            if checkpoint is None:
                raise ValueError(f'checkpoint not found for {architecture}')
            embedding_store.register_model(model_name,
                local_path=checkpoint.as_posix(), language='Dutch', size='base',
                architecture=architecture)
    except Exception:
        embedding_store.close()
        raise
    return embedding_store


def create_wv2_and_hubert(condition, n_types, cgn,
    workspace=lv_locations.workspace):
    '''Return both configured stores for a selection; caller closes them.'''
    wav2vec_path = lv_locations.store_path('w2v', condition, n_types,
        workspace=workspace)
    hubert_path = lv_locations.store_path('hubert', condition, n_types,
        workspace=workspace)
    wav2vec_store = create_store(wav2vec_path, *MODELS['w2v'], cgn=cgn)
    try:
        hubert_store = create_store(hubert_path, *MODELS['hubert'], cgn=cgn)
    except Exception:
        wav2vec_store.close()
        raise
    return wav2vec_store, hubert_store
