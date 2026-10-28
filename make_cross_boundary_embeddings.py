'''Compute full-frame EchoFrame embeddings for sampled CGN markers.'''

import json
from pathlib import Path

import echoframe

import cross_boundary_lexicon as cbl
import stores


ROOT = Path(__file__).resolve().parent
MODEL_SETTINGS = {
    'wav2vec2': ('wav2vec2_nl1_checkpoint-200000',
        'lexicon100_2466_cgn', 'cross_boundary100_2200_cgn'),
    'hubert': ('huibert_nl1_checkpoint-200000',
        'lexicon100_2466_cgn_hubert', 'cross_boundary100_2200_cgn_hubert'),
}


def load_store(model='wav2vec2', root=None, reference_root=None,
    phraser_store=None):
    '''Open a cross-boundary store, registering its checkpoint and CGN source.
    model:          wav2vec2 or hubert
    root:           destination; defaults to this module's model-specific path
    reference_root: word-lexicon store supplying the checkpoint registration
    phraser_store:  source CGN store; defaults to the shared stores.py instance
    The caller owns the returned EchoFrame store and should close it.
    '''
    if model not in MODEL_SETTINGS:
        raise ValueError('model must be wav2vec2 or hubert')
    model_name, reference_name, destination = MODEL_SETTINGS[model]
    root = ROOT / destination if root is None else Path(root)
    reference_root = ROOT / reference_name if reference_root is None \
        else Path(reference_root)
    embedding_store = echoframe.Store(root)
    try:
        if embedding_store.load_model_metadata(model_name) is None:
            with (reference_root / 'config.json').open() as f:
                models = json.load(f).get('models', {})
            if model_name not in models:
                raise ValueError(f'{model_name} is not registered in '
                    f'{reference_root / "config.json"}')
            record = models[model_name]
            fields = ('local_path', 'huggingface_id', 'language', 'size',
                'architecture')
            embedding_store.register_model(model_name,
                **{field: record.get(field) for field in fields})
        if phraser_store is None:
            phraser_store = stores.load_phraser_cgn_store()
        embedding_store.attach_phraser_store(stores.CGN_SOURCE_ID, phraser_store)
    except BaseException:
        embedding_store.close()
        raise
    return embedding_store


def compute_embeddings(markers=None, model='wav2vec2', root=None,
    layers=(3, 6, 9, 12), collar=500, gpu=False, batch_size=32,
    keys_path='svs_data/cross_boundary_marker_keys.pkl', reference_root=None,
    phraser_store=None, verbose=True):
    '''Compute missing marker-layer matrices and return the open EchoFrame store.
    markers:        persisted Marker objects; None loads the saved key file
    model:          wav2vec2 or hubert
    root:           destination; defaults to the model-specific store directory
    layers:         hidden-state layers, matching the word lexicon by default
    collar:         context on each side in milliseconds; not stored as frames
    gpu:            run inference on the GPU
    batch_size:     markers per inference/write batch
    keys_path:      pickle containing ordered saved marker keys
    reference_root: word-lexicon store supplying the checkpoint registration
    phraser_store:  CGN store containing the markers
    Store all frames fully inside each marker, without pooling. Existing
    marker/model/layer/collar entries are skipped, so calls can be resumed.
    Close the returned store when finished; it retains the loaded model cache.
    '''
    if model not in MODEL_SETTINGS:
        raise ValueError('model must be wav2vec2 or hubert')
    if phraser_store is None:
        phraser_store = stores.load_phraser_cgn_store()
    if markers is None:
        markers = cbl.load_markers(phraser_store, keys_path=keys_path)
    else:
        markers = list(markers)
    if not markers:
        raise ValueError('no markers to compute embeddings for')
    if any(marker.object_type != 'Marker' for marker in markers):
        raise ValueError('all segments must be persisted phraser markers')

    from echoframe.batch_segment_features import compute_embeddings_batch

    embedding_store = load_store(model, root=root, reference_root=reference_root,
        phraser_store=phraser_store)
    try:
        compute_embeddings_batch(markers, layers=layers,
            model_name=MODEL_SETTINGS[model][0], store=embedding_store,
            collar=collar, gpu=gpu, batch_size=batch_size,
            tags=['cgn', 'cross_boundary'], verbose=verbose)
    except BaseException:
        embedding_store.close()
        raise
    return embedding_store
