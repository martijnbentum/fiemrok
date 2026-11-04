'''Paths for the lexicon-validation experiment.'''

from numbers import Integral
from pathlib import Path


workspace = Path(__file__).resolve().parents[2] / 'lexicon_validation'
word_keys = workspace / 'word_keys.pkl'
cross_boundary_marker_keys = workspace / 'cross_boundary_marker_keys.pkl'
echoframe_stores = workspace / 'echoframe_stores'
svs_data = workspace / 'svs_data'
default_components = 'befgijklno'
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
