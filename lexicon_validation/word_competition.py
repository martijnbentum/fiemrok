'''Word competition for cross-boundary frame-offset queries.

Every cross-boundary query frame was searched in the cross-boundary index
and, separately, in the word index without the marker's own words. Per frame
the two stored top lists are merged by cosine similarity (exact up to the
stored top_k) and the top n_neighbours are split into shares: the marker's
own type, word 1 (a), word 2 (b), the previous and the next word. Curves sit
on a section axis prev | a | b | next with each word stretched to unit
width; frames in gaps between words are dropped.

Markers need a and b in the word lexicon. Per token and slot a status
records why a curve is missing: 0 used, 1 not in the lexicon (or no word),
2 gap of MAX_GAP_MS or more, 3 label shared with another used slot,
4 marker excluded because a or b is unusable.
'''

from pathlib import Path

import h5py
import numpy as np
from progressbar import progressbar
from speech_vector_search.metadata import VectorMetadatas

from . import frame_offset_curves as foc
from . import lv_locations
from .neighbour_helper import INVALID_INDEX, index_identity, type_mapping
from .neighbour_results import Result

SLOTS = ('prev', 'a', 'b', 'next')
SECTION_POINTS = 25
SECTION_GRID = (np.arange(4 * SECTION_POINTS) + .5) / SECTION_POINTS
SHARES = ('own', 'a', 'b', 'prev', 'next', 'marker')
RANKS = ('own_rank', 'a_rank', 'b_rank')
CURVES = SHARES + RANKS
STATUSES = ('used', 'not_in_lexicon', 'gap', 'collision', 'marker_excluded')


def load_curves(model, layer, pooling, n_neighbours=10, overwrite=False,
    workspace=lv_locations.workspace):
    '''Load cached word-competition curves, computing them if missing.

    model:         hubert or w2v
    pooling:       center or mean; selects both result files
    n_neighbours:  size of the merged neighbour list
    overwrite:     recompute and replace an existing cache
    Returns per-token arrays keyed by CURVES on SECTION_GRID, plus
    token_types (marker types), {slot}_status, marker_start, marker_end
    (section positions) and the index sizes n_markers and n_words.
    '''
    path = lv_locations.word_competition_path(model, layer, pooling,
        workspace)
    own_path = lv_locations.frame_offset_path(model, 'cross_boundary', layer,
        pooling, workspace)
    cross_path = lv_locations.cross_frame_offset_path(model, layer, pooling,
        workspace)
    with Result(own_path) as own, h5py.File(cross_path, 'r') as cross:
        expected = own.attributes['selection_fingerprint']
        if path.exists() and not overwrite:
            curves = foc._load_npz(path)
            if (curves.pop('selection_fingerprint') != expected
                or curves.pop('n_neighbours') != n_neighbours):
                m = f'{path} does not match the results; '
                raise ValueError(m + 'rebuild it with overwrite=True')
            return curves
        adjacency = foc.load_adjacency('cross_boundary', workspace=workspace,
            selection_fingerprint=expected)
        curves = compute_curves(own, cross, adjacency, n_neighbours)
    foc._save_npz(path, curves, selection_fingerprint=expected,
        n_neighbours=n_neighbours)
    return curves


def compute_curves(own, cross, adjacency, n_neighbours=10, chunk_size=500):
    '''Interpolate per-frame shares and nearest ranks for every marker.

    own:        cross-boundary frame-offset Result (cross-boundary index)
    cross:      open h5py file of the same queries against the word index
    adjacency:  marker adjacency from foc.load_adjacency('cross_boundary')
    '''
    _check_pair(own, cross, n_neighbours)
    word_types, lookup = _word_types(cross)
    marker_types = own._mapping['token_types'].astype('int64')
    n_tokens = len(marker_types)
    slot_types, statuses = slot_table(adjacency, lookup)
    sections = section_bounds(adjacency, statuses)
    self_columns = foc._self_columns(own._mapping['type_indices'], n_tokens)
    curves = {name: np.full((n_tokens, len(SECTION_GRID)), np.nan, 'float32')
        for name in CURVES}
    markers = dict(marker_start=np.full(n_tokens, np.nan),
        marker_end=np.full(n_tokens, np.nan))
    for start in progressbar(range(0, n_tokens, chunk_size)):
        stop = min(n_tokens, start + chunk_size)
        arrays = own.read(start, stop)
        crossed = {name: cross[name][start:stop] for name in ('status',
            'neighbour_indices', 'neighbour_similarities', 'a_type_ranks',
            'b_type_ranks')}
        types = dict(own=marker_types[start:stop],
            **{slot: slot_types[slot][start:stop] for slot in SLOTS})
        values, valid = offset_values(arrays, crossed, marker_types,
            word_types, types, self_columns[start:stop], n_neighbours)
        for row in range(stop - start):
            token = start + row
            if statuses['a'][token] == 4: continue
            _fill_token(curves, markers, arrays, values, valid, row, token,
                sections[token], slot_types)
    return dict(curves, **markers, token_types=marker_types,
        **{f'{slot}_status': statuses[slot] for slot in SLOTS},
        n_markers=n_tokens, n_words=len(word_types))


def slot_table(adjacency, lookup):
    '''Return word types (-1 when unused) and statuses per slot.'''
    types, statuses = {}, {}
    for slot in SLOTS:
        labels = adjacency[f'{slot}_label']
        types[slot] = np.array([lookup.get(label, -1) for label in labels])
        statuses[slot] = np.where(types[slot] < 0, 1, 0).astype('int8')
    for slot in ('prev', 'next'):
        gap = adjacency[f'{slot}_gap']
        far = (statuses[slot] == 0) & ~(gap < foc.MAX_GAP_MS)
        statuses[slot][far] = 2
    used = {slot: statuses[slot] == 0 for slot in SLOTS}
    for slot in SLOTS:
        label = adjacency[f'{slot}_label']
        for other in SLOTS:
            if other == slot: continue
            same = label == adjacency[f'{other}_label']
            statuses[slot][used[slot] & used[other] & same] = 3
    excluded = (types['a'] < 0) | (types['b'] < 0)
    for slot in SLOTS:
        statuses[slot][excluded] = 4
        types[slot] = np.where(statuses[slot] == 0, types[slot], -1)
    return types, statuses


def section_bounds(adjacency, statuses):
    '''Return per marker the (start, end) ms of each SLOTS section.

    Sections of missing words, or of prev/next words beyond the gap limit,
    are NaN; their frames are dropped.
    '''
    bounds = np.stack([np.stack([adjacency[f'{slot}_start'],
        adjacency[f'{slot}_end']], 1) for slot in SLOTS], 1)
    for i, slot in enumerate(SLOTS):
        unusable = (statuses[slot] == 2) | (adjacency[f'{slot}_label'] == '')
        bounds[unusable, i] = np.nan
    return bounds


def section_positions(times_ms, bounds):
    '''Return the section index and fraction of each time; -1 in gaps.'''
    times_ms = np.asarray(times_ms, dtype=float)
    sections = np.full(len(times_ms), -1)
    fractions = np.full(len(times_ms), np.nan)
    for i, (start, end) in enumerate(bounds):
        inside = (times_ms >= start) & (times_ms <= end)
        sections[inside] = i
        fractions[inside] = (times_ms[inside] - start) / (end - start)
    return sections, fractions


def offset_values(arrays, crossed, marker_types, word_types, types,
    self_columns, n_neighbours):
    '''Return per-offset shares and log ranks for one chunk of markers.'''
    valid = (arrays['status'] == 0) & (crossed['status'] == 0)
    n_own = arrays['neighbour_indices'].shape[-1]
    similarities = np.concatenate([arrays['neighbour_similarities'],
        crossed['neighbour_similarities']], -1)
    similarities = np.where(valid[..., None], similarities, 0)
    top = np.argpartition(-similarities, n_neighbours - 1, -1)
    top = top[..., :n_neighbours]
    marker_rows = np.where(valid[..., None], arrays['neighbour_indices'], 0)
    word_rows = np.where(valid[..., None], crossed['neighbour_indices'], 0)
    labels = np.concatenate([marker_types[marker_rows],
        word_types[word_rows]], -1)
    labels = np.take_along_axis(labels, top, -1)
    is_marker = top < n_own
    own = types['own'][:, None, None]
    values = dict(marker=is_marker.mean(-1),
        own=(is_marker & (labels == own)).mean(-1))
    for slot in SLOTS:
        word = types[slot][:, None, None]
        values[slot] = (~is_marker & (labels == word)).mean(-1)
    own_ranks = arrays['same_type_ranks'].astype(float)
    rows = np.arange(len(own_ranks))
    own_ranks[rows, :, self_columns] = np.inf
    values['own_rank'] = _log_nearest(own_ranks)
    for slot in ('a', 'b'):
        ranks = crossed[f'{slot}_type_ranks'].astype(float)
        values[f'{slot}_rank'] = _log_nearest(ranks)
    return values, valid


def simultaneous_band(values, token_types, n_boot=1000, seed=0, level=.95):
    '''Return mean, low and high of a simultaneous type-bootstrap band.

    The band uses the max-|z| over grid points across bootstrap draws.
    '''
    mean, boot = foc.type_bootstrap(values, token_types, n_boot, seed)
    with np.errstate(invalid='ignore', divide='ignore'):
        sd = np.nanstd(boot, 0)
        z = np.nanmax(np.abs(boot - mean) / sd, 1)
    critical = np.nanpercentile(z, level * 100)
    return mean, mean - critical * sd, mean + critical * sd


def coverage(values):
    '''Return per grid point the share of a curve's tokens with a value.'''
    present = ~np.isnan(values)
    tokens = present.any(1)
    return present[tokens].mean(0) if tokens.any() else present.mean(0)


def regions(x, mask):
    '''Return (first, last) x of each run of True values in mask.'''
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(x[a], x[b - 1]) for a, b in zip(starts, stops)]


def exclusion_table(curves):
    '''Return per slot the marker counts per status.'''
    return {slot: {name: int((curves[f'{slot}_status'] == code).sum())
        for code, name in enumerate(STATUSES)} for slot in SLOTS}


def _check_pair(own, cross, n_neighbours):
    '''Check that both results hold the same queries.'''
    attrs = cross.attrs
    if attrs['result_kind'] != 'cross_frame_offsets':
        raise ValueError('expected cross-boundary vs word results')
    same = (own.attributes['selection_fingerprint']
        == attrs['selection_fingerprint']
        and own.attributes['query_pooling'] == attrs['query_pooling']
        and np.array_equal(own.offsets, cross['offsets'][:]))
    if not same: raise ValueError('results hold different queries')
    stored = min(own._file['neighbour_indices'].shape[-1],
        cross['neighbour_indices'].shape[-1])
    if stored < n_neighbours:
        raise ValueError('results store fewer than n_neighbours neighbours')


def _word_types(cross):
    '''Return word-index token types and a label to type lookup.'''
    attrs = cross.attrs
    directory = Path(cross.filename).parent / attrs['index_directory']
    metadatas = VectorMetadatas.load_json(directory.resolve()
        / f'{attrs["index_name"]}.json')
    identity = index_identity(metadatas)
    if any(attrs[f'index_{key}'] != value for key, value in identity.items()):
        raise ValueError('cross results do not match the word index')
    mapping = type_mapping(metadatas)
    lookup = {metadatas[int(rows[0])].label: i
        for i, rows in enumerate(mapping['type_indices'])}
    return mapping['token_types'].astype('int64'), lookup


def _log_nearest(ranks):
    '''Return log10(nearest rank + 1), NaN without a ranked token.'''
    ranks[ranks == INVALID_INDEX] = np.inf
    nearest = ranks.min(-1)
    with np.errstate(divide='ignore'):
        return np.where(np.isfinite(nearest), np.log10(nearest + 1), np.nan)


def _fill_token(curves, markers, arrays, values, valid, row, token, bounds,
    slot_types):
    '''Fill one marker's curves and marker span on the section axis.'''
    reference = [arrays['reference_starts'][row], arrays['reference_ends'][row]]
    sections, fractions = section_positions(np.array(reference) * 1000, bounds)
    for name, section, fraction in zip(markers, sections, fractions):
        if section >= 0: markers[name][token] = section + fraction
    if not valid[row].any(): return
    times = arrays['query_times'][row, valid[row]] * 1000
    sections, fractions = section_positions(times, bounds)
    for name in CURVES:
        if name in slot_types and slot_types[name][token] < 0: continue
        y = values[name][row, valid[row]]
        curves[name][token] = _interpolate_sections(sections, fractions, y)


def _interpolate_sections(sections, fractions, y):
    '''Interpolate within each section only, so gaps are never bridged.'''
    curve = np.full(len(SECTION_GRID), np.nan)
    grid = SECTION_GRID[:SECTION_POINTS]
    for i in range(len(SLOTS)):
        inside = sections == i
        if not inside.any(): continue
        part = slice(i * SECTION_POINTS, (i + 1) * SECTION_POINTS)
        curve[part] = foc._interpolate(grid, fractions[inside], y[inside])
    return curve
