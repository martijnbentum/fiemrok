'''Plot frame-offset curves for center and mean pooling.'''

from pathlib import Path

import numpy as np

from . import frame_offset_curves as foc
from . import lv_locations

COLORS = dict(center='#1b6ca8', mean='#d9622b')
POSITION_TICKS = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
POSITION_LABELS = ['-1', '-0.5', 'onset', '0.5', 'offset', '+0.5', '+1']


def plot_frame_offsets(model, condition, layer, n_neighbours=10,
    overwrite=False, workspace=lv_locations.workspace, output=None):
    '''Plot center vs mean frame-offset curves for one model and layer.

    model:         hubert or w2v
    condition:     word or cross_boundary
    n_neighbours:  neighbours used for precision (precision@n)
    overwrite:     recompute cached curves
    output:        figure path without suffix; defaults to the workspace
                   figures folder. Saves .png and .pdf.
    Returns the matplotlib figure. Cross-boundary figures leave the
    adjacent-word panel empty.
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
    for name in ('precision', 'self_rank', 'adjacent'):
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
    axes['self_rank'].set_ylabel('global rank of own pooled vector (+1)')
    axes['self_rank'].set_title('Self rank (median, IQR band)')
    axes['adjacent'].set_ylabel(f'{precision} (adjacent word type)')
    title = f'Previous/next word (in lexicon, gap < {foc.MAX_GAP_MS} ms)'
    if condition != 'word': title = 'Previous/next word: words only'
    axes['adjacent'].set_title(title)
    for name, label in (('onset', 'ms from onset'), ('offset', 'ms from offset')):
        axes[name].axvline(0, color='grey', lw=0.8)
        axes[name].set_xlabel(label)
        axes[name].set_title(f'{name.capitalize()} aligned')
    axes['onset'].set_ylabel(f'{precision} (target type)')
    axes['onset'].legend(frameon=False, fontsize=8)
