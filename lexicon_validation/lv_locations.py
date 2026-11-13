'''Paths for the lexicon-validation experiment.'''

from numbers import Integral
from pathlib import Path


workspace = Path(__file__).resolve().parents[2] / 'lexicon_validation'
word_keys = workspace / 'word_keys.pkl'
cross_boundary_marker_keys = workspace / 'cross_boundary_marker_keys.pkl'
echoframe_stores = workspace / 'echoframe_stores'
svs_data = workspace / 'svs_data'
default_components = 'abefghijklno'
default_language = 'nl'


def phraser_store_path(components=default_components,
    language=default_language, workspace=workspace):
    '''Name the local material store by included CGN components and variant.'''
    components = ''.join(sorted(set(components)))
    if not components or not set(components) <= set('abcdefghijklmno'):
        raise ValueError('components must contain CGN letters a through o')
    if language not in ('nl', 'vl'):
        raise ValueError('language must be nl or vl')
    return Path(workspace) / f'cgn_{components}_{language}'


def selection_path(condition, workspace=workspace):
    '''Return the saved ordered-key file for one dataset.'''
    names = {'word': word_keys.name,
        'cross_boundary': cross_boundary_marker_keys.name}
    if condition not in names:
        raise ValueError('condition must be word or cross_boundary')
    return Path(workspace) / names[condition]


def store_path(model, condition, n_types, workspace=workspace):
    '''Name a store using its model, dataset, and actual type count.'''
    if model not in ('w2v', 'hubert'):
        raise ValueError('model must be w2v or hubert')
    selection_path(condition, workspace)
    if (isinstance(n_types, bool) or not isinstance(n_types, Integral)
        or n_types < 1):
        raise ValueError('n_types must be a positive integer')
    name = f'{model}_{condition}_{n_types}'
    return Path(workspace) / 'echoframe_stores' / name


def find_store_path(model, condition, workspace=workspace):
    '''Find the single existing store for one model and condition.'''
    if model not in ('w2v', 'hubert'):
        raise ValueError('model must be w2v or hubert')
    selection_path(condition, workspace)
    directory = Path(workspace) / 'echoframe_stores'
    prefix = f'{model}_{condition}_'
    matches = []
    for path in directory.glob(f'{prefix}*'):
        if path.is_dir() and path.name[len(prefix):].isdigit():
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f'no {model}/{condition} store in {directory}')
    if len(matches) > 1:
        raise ValueError(f'multiple {model}/{condition} stores in {directory}')
    return matches[0]


def _index_condition(model, condition, layer, pooling):
    store_path(model, condition, 1)
    if (isinstance(layer, bool) or not isinstance(layer, Integral)
        or layer not in (3, 6, 9, 12)):
        raise ValueError('layer must be 3, 6, 9, or 12')
    if pooling not in ('center', 'mean'):
        raise ValueError('pooling must be center or mean')


def find_index_path(model, condition, layer, pooling, directory=svs_data):
    '''Find the single current pipeline vector index for one condition.'''
    _index_condition(model, condition, layer, pooling)
    prefix, suffix = f'{model}_{condition}_', f'_{layer}_{pooling}'
    matches = [path for path in Path(directory).glob(f'{prefix}*{suffix}.npy')
        if path.is_file() and path.stem[len(prefix):-len(suffix)].isdigit()]
    if not matches:
        raise FileNotFoundError(f'no vector index for '
            f'{model}/{condition}/{layer}/{pooling} in {directory}')
    if len(matches) > 1:
        raise ValueError(f'multiple vector indexes for '
            f'{model}/{condition}/{layer}/{pooling} in {directory}')
    return matches[0]


def neighbour_mapping_path(condition, workspace=workspace):
    selection_path(condition, workspace)
    return Path(workspace) / 'neighbours' / condition / 'token_mapping.npz'


def neighbour_results_path(model, condition, layer, pooling,
    workspace=workspace):
    _index_condition(model, condition, layer, pooling)
    return (Path(workspace) / 'neighbours' / condition / model /
        f'layer{layer:02d}_{pooling}.h5')
