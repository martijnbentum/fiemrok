'''Per-token frame-offset curves on shared position and time grids.

Each token's per-offset values are linearly interpolated onto fixed grids:
position in reference durations (0 = onset, 1 = offset, one duration of
context per side) and milliseconds from onset or offset. Values outside a
token's valid queries are NaN. Curves are cached per result in the
validation workspace and rebuilt when the selection changes.
'''

import numpy as np
from progressbar import progressbar

from . import lv_locations, token_features
from .neighbour_helper import fingerprint
from .neighbour_results import Result

POSITION_GRID = np.linspace(-1, 2, 91)
ONSET_MS = np.arange(-1000, 501, 20)
OFFSET_MS = np.arange(-500, 1001, 20)
MAX_GAP_MS = 100
CURVES = ('precision', 'self_rank', 'prev', 'next', 'onset', 'offset')


def load_curves(model, condition, layer, pooling, n_neighbours=10,
    overwrite=False, workspace=lv_locations.workspace):
    '''Load cached curves for one frame-offset result, computing if missing.

    model:         hubert or w2v
    condition:     word or cross_boundary
    n_neighbours:  neighbours used for precision (precision@n)
    overwrite:     recompute and replace an existing cache
    Returns a dict of per-token arrays keyed by CURVES plus token_types.
    prev and next are only computed for words; for cross_boundary they
    are all NaN.
    '''
    path = lv_locations.frame_offset_curves_path(model, condition, layer,
        pooling, workspace)
    filename = lv_locations.frame_offset_path(model, condition, layer,
        pooling, workspace)
    with Result(filename) as result:
        expected = result.attributes['selection_fingerprint']
        if path.exists() and not overwrite:
            curves = _load_npz(path)
            if (curves.pop('selection_fingerprint') != expected
                or curves.pop('n_neighbours') != n_neighbours):
                m = f'{path} does not match the results; '
                raise ValueError(m + 'rebuild it with overwrite=True')
            return curves
        adjacency = None
        if condition == 'word':
            adjacency = load_adjacency(condition, workspace=workspace,
                selection_fingerprint=expected)
        curves = compute_curves(result, adjacency, n_neighbours)
    _save_npz(path, curves, selection_fingerprint=expected,
        n_neighbours=n_neighbours)
    return curves


def compute_curves(result, adjacency=None, n_neighbours=10, chunk_size=500):
    '''Interpolate one result's per-offset metrics for every token.

    adjacency:  optional dict with prev_label, prev_gap, next_label and
                next_gap per token; adjacent curves stay NaN without it
    precision is the fraction of the first n_neighbours with the token's
    type; self_rank is log10(global rank of the token's own vector + 1);
    prev/next are the fraction with the adjacent word's type, only for
    tokens whose adjacent word is a lexicon type within MAX_GAP_MS.
    '''
    if result.result_kind != 'frame_offsets':
        raise ValueError('curves need frame-offset results')
    mapping = result._mapping
    token_types = mapping['token_types'].astype('int64')
    n_tokens = len(token_types)
    adjacent = adjacent_types(result, adjacency)
    self_columns = _self_columns(mapping['type_indices'], n_tokens)
    curves = {name: np.full((n_tokens, len(grid)), np.nan, 'float32')
        for name, grid in _grids().items()}
    for start in progressbar(range(0, n_tokens, chunk_size)):
        stop = min(n_tokens, start + chunk_size)
        arrays = result.read(start, stop)
        if arrays['neighbour_indices'].shape[-1] < n_neighbours:
            raise ValueError('results store fewer than n_neighbours neighbours')
        values = _offset_values(arrays, token_types, adjacent, self_columns,
            start, stop, n_neighbours)
        for row in range(stop - start):
            _interpolate_token(curves, arrays, values, row, start + row,
                adjacent)
    curves['token_types'] = token_types
    return curves


def adjacent_types(result, adjacency):
    '''Return previous/next type indices per token, -1 when unusable.'''
    n_tokens = len(result)
    types = dict(prev=np.full(n_tokens, -1), next=np.full(n_tokens, -1))
    if adjacency is None: return types
    lookup = result._type_labels
    for side in ('prev', 'next'):
        labels = adjacency[f'{side}_label']
        gaps = adjacency[f'{side}_gap']
        if len(labels) != n_tokens:
            raise ValueError('adjacency does not match the result tokens')
        types[side] = np.array([lookup.get(label, -1) if gap < MAX_GAP_MS
            else -1 for label, gap in zip(labels, gaps)])
    return types


def load_adjacency(condition, store=None, overwrite=False,
    workspace=lv_locations.workspace, selection_fingerprint=None):
    '''Load the previous/next word per token, building it when missing.

    Only words are supported; cross-boundary adjacency needs a separate
    definition. Gaps are in ms; edges of the speaker's stream have no label
    and a NaN gap.
    '''
    if condition != 'word':
        raise ValueError('adjacency is only defined for word results')
    path = lv_locations.adjacency_path(condition, workspace)
    if path.exists() and not overwrite:
        adjacency = _load_npz(path)
        saved = adjacency.pop('selection_fingerprint')
        if selection_fingerprint not in (None, saved):
            m = f'{path} does not match the selection; '
            raise ValueError(m + 'rebuild it with overwrite=True')
        return adjacency
    keys, items = token_features.load_selected_items(condition, store,
        workspace=workspace)
    expected = fingerprint(keys)
    if selection_fingerprint not in (None, expected):
        raise ValueError('saved selection does not match results')
    adjacency = build_adjacency(items)
    _save_npz(path, adjacency, selection_fingerprint=expected)
    return adjacency


def build_adjacency(items, stream_words=None):
    '''Find each word's neighbouring words in its speaker's stream.

    stream_words: function returning a time-ordered word list for an item;
    defaults to the speaker's words in the recording, across phrases.
    '''
    if stream_words is None: stream_words = _speaker_words()
    n_items = len(items)
    adjacency = {f'{side}_{name}': np.full(n_items, fill, dtype)
        for side in ('prev', 'next')
        for name, fill, dtype in (('label', '', object), ('gap', np.nan, float))}
    order = sorted(range(n_items),
        key=lambda i: token_features._stream_key(items[i]))
    words, positions = None, {}
    for index in progressbar(order):
        item = items[index]
        stream = stream_words(item)
        if stream is not words:
            words = stream
            positions = {word.key: i for i, word in enumerate(words)}
        position = positions[item.key]
        if position > 0:
            word = words[position - 1]
            adjacency['prev_label'][index] = word.label
            adjacency['prev_gap'][index] = item.start - word.end
        if position + 1 < len(words):
            word = words[position + 1]
            adjacency['next_label'][index] = word.label
            adjacency['next_gap'][index] = word.start - item.end
    for side in ('prev', 'next'):
        adjacency[f'{side}_label'] = adjacency[f'{side}_label'].astype(str)
    return adjacency


def type_weighted_mean(values, token_types, n_boot=1000, seed=0):
    '''Return mean, low and high per grid column with a type bootstrap CI.

    The mean weights tokens equally; the 95% interval resamples types.
    NaN values are skipped.
    '''
    values = np.asarray(values, dtype=float)
    present = ~np.isnan(values)
    n_types = int(token_types.max()) + 1
    sums = np.zeros((n_types, values.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, token_types, np.where(present, values, 0))
    np.add.at(counts, token_types, present)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = sums.sum(0) / counts.sum(0)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n_types, (n_boot, n_types))
        weights = np.zeros((n_boot, n_types))
        np.add.at(weights, (np.arange(n_boot)[:, None], draws), 1)
        boot = (weights @ sums) / (weights @ counts)
    if not np.isfinite(boot).any(): return mean, mean.copy(), mean.copy()
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return mean, low, high


def _grids():
    '''Return each curve's grid.'''
    position = dict(precision=POSITION_GRID, self_rank=POSITION_GRID,
        prev=POSITION_GRID, next=POSITION_GRID)
    return {**position, 'onset': ONSET_MS, 'offset': OFFSET_MS}


def _self_columns(type_indices, n_tokens):
    '''Return each token's column within its type's same-type arrays.'''
    columns = np.full(n_tokens, -1, 'int64')
    for row in type_indices: columns[row] = np.arange(len(row))
    if (columns < 0).any(): raise ValueError('mapping misses tokens')
    return columns


def _offset_values(arrays, token_types, adjacent, self_columns, start, stop,
    n_neighbours):
    '''Return per-offset metrics for one chunk of tokens.'''
    valid = arrays['status'] == 0
    neighbours = arrays['neighbour_indices'][..., :n_neighbours]
    neighbours = token_types[np.where(valid[..., None], neighbours, 0)]
    rows = np.arange(stop - start)
    ranks = arrays['same_type_ranks'][rows, :, self_columns[start:stop]]
    ranks = np.where(valid, ranks, 0).astype(float)
    values = dict(self_rank=np.log10(ranks + 1),
        precision=(neighbours == token_types[start:stop, None, None]).mean(-1))
    for side in ('prev', 'next'):
        types = adjacent[side][start:stop, None, None]
        values[side] = (neighbours == types).mean(-1)
    return values


def _interpolate_token(curves, arrays, values, row, token, adjacent):
    '''Fill one token's curves from its valid offsets.'''
    valid = arrays['status'][row] == 0
    if not valid.any(): return
    times = arrays['query_times'][row, valid]
    start = arrays['reference_starts'][row]
    end = arrays['reference_ends'][row]
    position = (times - start) / (end - start)
    for name in ('precision', 'self_rank', 'prev', 'next'):
        if name in adjacent and adjacent[name][token] < 0: continue
        curves[name][token] = _interpolate(POSITION_GRID, position,
            values[name][row, valid])
    precision = values['precision'][row, valid]
    curves['onset'][token] = _interpolate(ONSET_MS, (times - start) * 1000,
        precision)
    curves['offset'][token] = _interpolate(OFFSET_MS, (times - end) * 1000,
        precision)


def _interpolate(grid, x, y):
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def _speaker_words():
    '''Return a cached stream-word lookup shared over sorted items.'''
    streams = {}

    def stream_words(item):
        if token_features._stream_key(item) not in streams: streams.clear()
        return token_features._speaker_stream(item, streams)['words']
    return stream_words


def _save_npz(path, arrays, **attributes):
    '''Write arrays via a temporary file so failures leave no cache.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez(temporary, **attributes, **arrays)
    temporary.replace(path)


def _load_npz(path):
    '''Load a cache into memory, converting scalar attributes.'''
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    for name, value in arrays.items():
        if value.ndim == 0: arrays[name] = value.item()
    return arrays
