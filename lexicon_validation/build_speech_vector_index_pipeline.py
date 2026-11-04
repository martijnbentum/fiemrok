'''Select validation datasets, extract embeddings, and build vector indexes.'''

from contextlib import contextmanager
from pathlib import Path
import pickle

import lexicon_helper
import model_paths
import stores
from . import cross_boundary_lexicon as cbl
from . import lv_locations
from . import lv_stores
from . import word_lexicon


LAYERS = (3, 6, 9, 12)
MODELS = {'w2v': model_paths.wav2vec2_nl1_name,
    'hubert': model_paths.hubert_nl1_name}
CONDITIONS = ('word', 'cross_boundary')


def build_local_phraser_store(components=lv_locations.default_components,
    language=lv_locations.default_language, source_store=None,
    workspace=lv_locations.workspace):
    '''Copy eligible recordings into a new local material store.
    components:   included CGN component letters, e.g. 'befgijklno'
    language:     CGN variant directory, nl or vl
    source_store: source Phraser Store; defaults to the shared CGN store
    Returns an open local Store; the caller closes it.
    '''
    from phraser import copy_audio_materials

    components = ''.join(sorted(set(components)))
    root = lv_locations.phraser_store_path(components, language, workspace)
    if root.exists(): raise FileExistsError(f'store already exists: {root}')
    print(f'Selecting source recordings: components={components}, '
        f'language={language}', flush=True)
    if source_store is None: source_store = stores.load_phraser_cgn_store()
    audio_keys = []
    for audio in source_store.audios.all():
        component = lexicon_helper.filename_to_component(audio.filename)
        variant = lexicon_helper.filename_to_dutch_variant(audio.filename)
        if component in components and variant == language:
            audio_keys.append(audio.key)
    if not audio_keys:
        raise ValueError('no recordings match the components and language')
    print(f'Selected {len(audio_keys):,} recordings for {root}', flush=True)
    return copy_audio_materials(sorted(audio_keys), root, source_store)


@contextmanager
def _local_phraser_store(cgn, workspace, components, language):
    '''Use a caller-owned store or open and close the named local store.'''
    if cgn is not None:
        yield cgn
        return
    local_store = lv_stores.load_phraser_store(components, language, workspace)
    try:
        yield local_store
    finally:
        local_store.close()


def prepare_samples(n_tokens=100, min_types=2000, random_seed=0,
    tolerance=10, max_attempts=100, tries_per_round=3, cgn=None,
    words=None, phrases=None, workspace=lv_locations.workspace,
    components=lv_locations.default_components,
    language=lv_locations.default_language):
    '''Select both datasets and save their ordered Phraser keys.
    n_tokens:   exact number of selected occurrences per type
    min_types:  minimum accepted cross-boundary type count
    workspace: validation output directory, containing both key files
    '''
    with _local_phraser_store(cgn, workspace, components, language) as cgn:
        if words is None: words = cgn.words.all()
        if phrases is None: phrases = cgn.phrases.all()
        eligible = word_lexicon.filter_word_tokens(words, n_tokens=n_tokens)
        selected_words = word_lexicon.select_words(eligible, n_tokens=n_tokens,
            random_seed=random_seed)
        if not selected_words:
            raise ValueError('no word types have enough eligible tokens')
        targets, _ = cbl.inspect_lexicon_phonemes(selected_words)
        phrases = cbl.filter_phrases_for_cross_boundary_lexicon(phrases)
        phrases = sorted(phrases, key=lambda phrase: phrase.key)
        lexicon = cbl.CrossBoundaryLexicon(phrases)
        plan = cbl.plan_sampling(lexicon, targets, n_tokens=n_tokens,
            tolerance=tolerance)
        lexicon.sample_lexicon(plan, random_seed=random_seed,
            tries_per_round=tries_per_round, max_attempts=max_attempts,
            min_types=min_types)
        marker_path = lv_locations.selection_path('cross_boundary', workspace)
        markers = lexicon.make_markers(cgn, save=True, keys_path=marker_path)
        word_keys = [word.key for word in selected_words]
        word_lexicon.save_selected_word_keys(word_keys,
            lv_locations.selection_path('word', workspace))
        print(f'Saved {len(word_keys)} word tokens and {len(markers)} markers')
        return {'word': word_keys,
            'cross_boundary': [marker.key for marker in markers]}


def _load_items(condition, cgn, workspace):
    '''Load the frozen sample in saved key order.'''
    path = lv_locations.selection_path(condition, workspace)
    with path.open('rb') as handle: keys = pickle.load(handle)
    if not keys: raise ValueError(f'empty {condition} selection')
    items = list(cgn.load_many(keys))
    if len(items) != len(keys) or any(item is None for item in items):
        raise ValueError(f'missing selected {condition} items')
    return keys, items


def build_embeddings(layers=LAYERS, collar=500, gpu=False, batch_size=32,
    verbose=True, cgn=None, workspace=lv_locations.workspace,
    components=lv_locations.default_components,
    language=lv_locations.default_language):
    '''Extract both saved datasets into four model/condition stores.'''
    from . import make_embeddings

    with _local_phraser_store(cgn, workspace, components, language) as cgn:
        layers = tuple(layers)
        paths = {}
        for condition in CONDITIONS:
            _, items = _load_items(condition, cgn, workspace)
            n_types = len({item.label for item in items})
            for model, model_name in MODELS.items():
                path = lv_locations.store_path(model, condition, n_types,
                    workspace=workspace)
                architecture = 'wav2vec2' if model == 'w2v' else 'hubert'
                embedding_store = make_embeddings.create_store(path, model_name,
                    architecture, cgn=cgn)
                try:
                    make_embeddings.compute_embeddings(embedding_store, items,
                        condition=condition, layers=layers, collar=collar,
                        gpu=gpu, batch_size=batch_size, verbose=verbose)
                finally:
                    embedding_store.close()
                paths[model, condition] = path
        return paths


def build_indexes(layers=LAYERS, collar=500, overwrite=False, cgn=None,
    workspace=lv_locations.workspace, directory=None,
    components=lv_locations.default_components,
    language=lv_locations.default_language):
    '''Build center/mean indexes from exactly the two saved selections.
    overwrite: explicitly allow replacing existing vector-index files
    directory: vector output folder; defaults to workspace/svs_data
    '''
    from . import build_speech_vector_index as builder

    build_index = builder.make_vector_search_index_for_layer
    with _local_phraser_store(cgn, workspace, components, language) as cgn:
        if directory is None: directory = Path(workspace) / 'svs_data'
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        layers = tuple(layers)
        paths = {}
        for condition in CONDITIONS:
            keys, items = _load_items(condition, cgn, workspace)
            n_types = len({item.label for item in items})
            unit_type = 'word' if condition == 'word' else 'marker'
            for model, model_name in MODELS.items():
                prefix = f'{model}_{condition}_{n_types}'
                embedding_store = lv_stores.load_echoframe_store(model,
                    condition, materials=cgn, workspace=workspace)
                try:
                    for layer in layers:
                        for pooling in ('center', 'mean'):
                            name = f'{prefix}_{layer}_{pooling}'
                            index, _ = build_index(embedding_store, layer,
                                name, selected_keys=keys,
                                model_name=model_name, directory=directory,
                                pooling=pooling, unit_type=unit_type,
                                collar=collar,
                                overwrite=overwrite)
                            paths[model, condition, layer, pooling] = (
                                directory / f'{name}.npy')
                            del index
                finally:
                    embedding_store.close()
        return paths
