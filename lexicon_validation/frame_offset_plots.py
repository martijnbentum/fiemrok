'''Plot frame-offset curves for center and mean pooling.'''

from pathlib import Path

import numpy as np

from . import frame_offset_curves as foc
from . import lv_locations
from . import word_competition as wc

COLORS = dict(center='#1b6ca8', mean='#d9622b')
POSITION_TICKS = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
POSITION_LABELS = ['-1', '-0.5', 'onset', '0.5', 'offset', '+0.5', '+1']
SLOT_COLORS = dict(own='#333333', a='#1b6ca8', b='#d9622b', prev='#5e9c3a',
    next='#8e5ea2')
SECTION_LABELS = dict(prev='prev', a='word 1', b='word 2', next='next')
MIN_COVERAGE = .5


def plot_frame_offsets(model, condition, layer, n_neighbours=10,
    overwrite=False, workspace=lv_locations.workspace, output=None):
    '''Plot center vs mean frame-offset curves for one model and layer.

    model:         hubert or w2v
    condition:     word or cross_boundary
    n_neighbours:  neighbours used for precision (precision@n)
    overwrite:     recompute cached curves
    output:        figure path without suffix; defaults to the workspace
                   figures folder. Saves .png and .pdf.
    Returns the matplotlib figure. Cross-boundary figures show the
    word-competition shares (center solid, mean dashed) instead of the
    adjacent-word panel.
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curves = {pooling: foc.load_curves(model, condition, layer, pooling,
        n_neighbours=n_neighbours, overwrite=overwrite, workspace=workspace)
        for pooling in COLORS}
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    axes = dict(precision=figure.add_subplot(grid[0, :2]),
        self_rank=figure.add_subplot(grid[0, 2:]),
        adjacent=figure.add_subplot(grid[1, :2]),
        onset=figure.add_subplot(grid[1, 2]))
    axes['offset'] = figure.add_subplot(grid[1, 3], sharey=axes['onset'])
    for pooling, values in curves.items():
        _plot_pooling(axes, pooling, values)
    if condition == 'cross_boundary':
        _plot_word_shares(axes['adjacent'], model, layer, n_neighbours,
            overwrite, workspace)
    _style_axes(axes, condition, n_neighbours)
    title = (f'{model} layer {layer}, {condition}: frame-offset retrieval '
        '(token-weighted means, 95% CI bootstrapped over types)')
    figure.suptitle(title)
    if output is None:
        name = f'frame_offsets_{model}_{condition}_layer{layer:02d}'
        output = lv_locations.figure_path(name, workspace)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        figure.savefig(output.with_name(f'{output.name}.{suffix}'), dpi=150)
    return figure


def _plot_pooling(axes, pooling, values):
    '''Draw one pooling's curves in every panel.'''
    color = COLORS[pooling]
    types = values['token_types']
    grid = foc.POSITION_GRID
    _band(axes['precision'], grid, values['precision'], types, color, pooling)
    log_ranks = values['self_rank']
    if not np.isnan(log_ranks).all():
        median = 10 ** np.nanmedian(log_ranks, 0)
        q25, q75 = 10 ** np.nanpercentile(log_ranks, [25, 75], 0)
        axes['self_rank'].plot(grid, median, color=color,
            label=f'{pooling} median')
        axes['self_rank'].fill_between(grid, q25, q75, color=color,
            alpha=0.2, lw=0)
        median = 10 ** np.nanmedian(values['type_rank'], 0)
        axes['self_rank'].plot(grid, median, '--', color=color,
            label=f'{pooling} nearest same type')
    for side, style in (('prev', '-'), ('next', '--')):
        used = ~np.isnan(values[side]).all(1)
        if not used.any(): continue
        label = f'{pooling}, {side} word (N={used.sum():,})'
        _band(axes['adjacent'], grid, values[side][used], types[used], color,
            label, style)
    _band(axes['onset'], foc.ONSET_MS, values['onset'], types, color, pooling)
    _band(axes['offset'], foc.OFFSET_MS, values['offset'], types, color,
        pooling)


def _band(ax, x, values, types, color, label, style='-'):
    '''Plot a token-weighted mean with its type-bootstrap interval.'''
    mean, low, high = foc.type_weighted_mean(values, types)
    ax.plot(x, mean, style, color=color, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.25, lw=0)


def _style_axes(axes, condition, n_neighbours):
    '''Add reference lines, ticks, labels and titles.'''
    precision = f'P@{n_neighbours}'
    names = ('precision', 'self_rank', 'adjacent')
    if condition != 'word': names = names[:2]
    for name in names:
        ax = axes[name]
        ax.axvline(0, color='grey', lw=0.8)
        ax.axvline(1, color='grey', lw=0.8)
        ax.axvspan(0, 1, color='grey', alpha=0.07, lw=0)
        ax.set_xticks(POSITION_TICKS, POSITION_LABELS)
        ax.set_xlim(-1, 2)
        ax.set_xlabel('query position (reference durations)')
        if ax.get_legend_handles_labels()[0]: ax.legend(frameon=False, fontsize=8)
    axes['precision'].set_ylabel(f'{precision} (target type)')
    axes['precision'].set_title(f'{precision} by query position')
    axes['self_rank'].set_yscale('log')
    axes['self_rank'].set_ylabel('global rank (+1)')
    title = 'Rank of own pooled vector (median, IQR) and nearest same type'
    axes['self_rank'].set_title(title)
    if condition == 'word':
        axes['adjacent'].set_ylabel(f'{precision} (adjacent word type)')
        title = f'Previous/next word (in lexicon, gap < {foc.MAX_GAP_MS} ms)'
        axes['adjacent'].set_title(title)
    for name, label in (('onset', 'ms from onset'), ('offset', 'ms from offset')):
        axes[name].axvline(0, color='grey', lw=0.8)
        axes[name].set_xlabel(label)
        axes[name].set_title(f'{name.capitalize()} aligned')
    axes['onset'].set_ylabel(f'{precision} (target type)')
    axes['onset'].legend(frameon=False, fontsize=8)


def plot_word_competition(model, layer, n_neighbours=10, overwrite=False,
    workspace=lv_locations.workspace, output=None):
    '''Plot own-type versus word shares for cross-boundary queries.

    model:         hubert or w2v
    n_neighbours:  size of the merged neighbour list
    overwrite:     recompute cached curves
    output:        figure path without suffix; defaults to the workspace
                   figures folder. Saves .png and .pdf.
    Columns are center and mean pooling. Rows: shares of the merged
    neighbours, own minus word 1/word 2 with a simultaneous band, nearest
    ranks, and the marker share with coverage. Grid points covered by
    fewer than MIN_COVERAGE of a curve's tokens are not drawn. Prints the
    exclusion table and the regions where a difference band excludes 0.
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True,
        sharey='row', constrained_layout=True)
    for column, pooling in enumerate(COLORS):
        curves = wc.load_curves(model, layer, pooling, n_neighbours,
            overwrite, workspace)
        print(pooling, wc.exclusion_table(curves))
        _plot_competition(axes[:, column], curves, pooling, n_neighbours)
    title = (f'{model} layer {layer}, cross_boundary queries: own type vs '
        'words (token-weighted means, bands bootstrapped over marker types)')
    figure.suptitle(title)
    if output is None:
        name = f'word_competition_{model}_layer{layer:02d}'
        output = lv_locations.figure_path(name, workspace)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        figure.savefig(output.with_name(f'{output.name}.{suffix}'), dpi=150)
    return figure


def _plot_competition(axes, curves, pooling, n_neighbours):
    '''Draw one pooling's column.'''
    x, types = wc.SECTION_GRID, curves['token_types']
    shares, differences, ranks, pools = axes
    _plot_shares(shares, curves)
    for name in ('a', 'b'):
        values = curves['own'] - curves[name]
        mean, low, high = wc.simultaneous_band(values, types)
        hidden = wc.coverage(values) < MIN_COVERAGE
        for array in (mean, low, high): array[hidden] = np.nan
        label = f'own - {SECTION_LABELS[name]}'
        differences.plot(x, mean, color=SLOT_COLORS[name], label=label)
        differences.fill_between(x, low, high, color=SLOT_COLORS[name],
            alpha=.25, lw=0)
        found = wc.regions(x, (low > 0) | (high < 0))
        print(pooling, label, 'band excludes 0 at',
            [(round(a, 2), round(b, 2)) for a, b in found])
    for name, pool, label in (('own', 'n_markers', 'own type'),
        ('a', 'n_words', 'word 1'), ('b', 'n_words', 'word 2')):
        values = curves[f'{name}_rank']
        median = 10 ** np.nanmedian(values, 0)
        median[wc.coverage(values) < MIN_COVERAGE] = np.nan
        color = SLOT_COLORS[name]
        ranks.plot(x, median, color=color, label=f'nearest {label}')
        ranks.axhline(curves[pool] / 31, color=color, ls=':', lw=.8)
    _clipped_band(pools, x, curves['marker'], types, SLOT_COLORS['own'],
        'marker share')
    pools.plot(x, wc.coverage(curves['own']), color='grey', ls='--',
        label='coverage (own curve)')
    precision = f'top {n_neighbours}'
    shares.set_title(f'{pooling}: share of merged {precision}')
    shares.set_ylabel(f'share of merged {precision}')
    differences.axhline(0, color='grey', lw=.8)
    differences.set_ylabel('own minus word share')
    differences.set_title('own type minus word (95% simultaneous band)')
    ranks.set_yscale('log')
    ranks.set_ylabel('median nearest rank (+1)')
    ranks.set_title('nearest own / word token (dotted: chance N/31)')
    pools.set_ylim(0, 1)
    pools.set_ylabel('fraction')
    pools.set_title('marker share of merged list; query coverage')
    _style_sections(axes, curves)


def _plot_word_shares(ax, model, layer, n_neighbours, overwrite, workspace):
    '''Draw word-competition shares, center solid and mean dashed.'''
    for pooling, style in (('center', '-'), ('mean', '--')):
        curves = wc.load_curves(model, layer, pooling, n_neighbours,
            overwrite, workspace)
        _plot_shares(ax, curves, style, f'{pooling}: ')
    _style_sections([ax], curves)
    # Leave room above the curves for the two-column legend.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
    ax.set_ylabel(f'share of merged top {n_neighbours}')
    ax.set_title('Word competition: own type vs words '
        '(cross-boundary + word index)')


def _plot_shares(ax, curves, style='-', prefix=''):
    '''Plot own-type and word shares on the section axis.'''
    for name, color in SLOT_COLORS.items():
        values = curves[name]
        n = int((~np.isnan(values).all(1)).sum())
        label = 'own type' if name == 'own' else SECTION_LABELS[name]
        label = f'{prefix}{label} (N={n:,})'
        _clipped_band(ax, wc.SECTION_GRID, values, curves['token_types'],
            color, label, style)


def _clipped_band(ax, x, values, types, color, label, style='-'):
    '''Plot a mean band, hiding grid points with low coverage.'''
    mean, low, high = foc.type_weighted_mean(values, types)
    hidden = wc.coverage(values) < MIN_COVERAGE
    for array in (mean, low, high): array[hidden] = np.nan
    ax.plot(x, mean, style, color=color, label=label)
    ax.fill_between(x, low, high, color=color, alpha=.25, lw=0)


def _style_sections(axes, curves):
    '''Mark section edges, the marker span and label the sections.'''
    starts, ends = curves['marker_start'], curves['marker_end']
    for ax in axes:
        for edge in (1, 2, 3): ax.axvline(edge, color='grey', lw=.8)
        for values in (starts, ends):
            if np.isnan(values).all(): continue
            q25, median, q75 = np.nanpercentile(values, [25, 50, 75])
            ax.axvspan(q25, q75, color='grey', alpha=.1, lw=0)
            ax.axvline(median, color='grey', ls='--', lw=.8)
        ax.set_xlim(0, 4)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xticks([.5, 1.5, 2.5, 3.5], list(SECTION_LABELS.values()))
    axes[-1].set_xlabel('query position (each word stretched to unit width; '
        'dashed: median marker edges, IQR shaded)')
