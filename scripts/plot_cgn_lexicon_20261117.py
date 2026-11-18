'''Plot paired retrieval conditions from sorted per-type P@10 summaries.'''

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import bootstrap

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


LAYERS = (3, 6, 9, 12)
SEED = 20261029
N_RESAMPLES = 10000
DATASETS = (('word', 'Word-aligned', 5675),
    ('cross_boundary', 'Across-word', 5391))
MODELS = (('w2v', 'Wav2Vec 2.0', '#0072B2'),
    ('hubert', 'HuBERT', '#D55E00'))
POOLINGS = (('mean', '-', 'o'), ('center', '--', 's'))


def summarize(values, n_types):
    '''Bootstrap the mean over whole type means, pairing draws by seed.'''
    if values.shape != (n_types,):
        raise ValueError('unexpected number of type means')
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('type means must be finite and between zero and one')
    if not np.allclose(values * 310, np.round(values * 310),
        rtol=0, atol=1e-8):
        raise ValueError('means do not match 31 tokens scored at P@10')
    result = bootstrap((values,), np.mean, n_resamples=N_RESAMPLES,
        batch=128, confidence_level=0.95, method='BCa',
        rng=np.random.default_rng(SEED))
    low, high = result.confidence_interval
    mean = values.mean()
    if not 0 <= low <= mean <= high <= 1:
        raise ValueError('invalid confidence interval')
    return float(mean), float(low), float(high)


def plot(data_path, output):
    '''Recreate the original two-panel figure with the new fixed selection.'''
    rows = []
    identities = {}
    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True)
    with np.load(data_path, allow_pickle=False) as data:
        if (int(data['precision_k']) != 10
            or int(data['tokens_per_type']) != 31
            or int(data['collar_ms']) != 2000):
            raise ValueError('unexpected metric, quota or collar')
        for ax, (condition, title, n_types) in zip(axes, DATASETS):
            labels = data[condition + '__labels']
            if len(labels) != n_types or len(set(labels)) != n_types:
                raise ValueError('unexpected or duplicate type labels')
            if not np.array_equal(labels, np.sort(labels)):
                raise ValueError('type identities must be sorted')
            identities[condition] = str(
                data[condition + '__selection_fingerprint'])
            for model, model_title, color in MODELS:
                for pooling, style, marker in POOLINGS:
                    summaries = []
                    for layer in LAYERS:
                        key = f'{condition}__{model}__{pooling}__{layer}'
                        mean, low, high = summarize(data[key], n_types)
                        summaries.append((mean, low, high))
                        rows.append({'dataset': condition, 'model': model,
                            'pooling': pooling, 'layer': layer,
                            'types': n_types, 'tokens': n_types * 31,
                            'mean': mean, 'ci_low': low, 'ci_high': high})
                        print(f'{title} | {model_title} | {pooling} | '
                            f'{layer} | {100 * mean:.2f}% | '
                            f'[{100 * low:.2f}, {100 * high:.2f}]',
                            flush=True)
                    means, lows, highs = np.asarray(summaries).T
                    intervals = np.asarray([means - lows, highs - means])
                    ax.plot(LAYERS, means, color=color, linestyle=style,
                        linewidth=1.95, zorder=2)
                    ax.plot(LAYERS, means, color=color, linestyle='none',
                        marker=marker, markersize=2.5, alpha=0.65, zorder=4)
                    ax.errorbar(LAYERS, means, yerr=intervals, fmt='none',
                        ecolor=color, elinewidth=1.8, capsize=5, capthick=1.8,
                        zorder=3)
            ax.set_title(title, fontsize=15, pad=12)
            ax.text(0.03, 0.96, f'{n_types:,} types; {n_types * 31:,} tokens',
                transform=ax.transAxes, va='top', fontsize=9, color='#59636D')
            ax.set_xlabel('Layer', labelpad=8)
            ax.set_xticks(LAYERS)
            ax.set_xlim(2.5, 12.5)
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color='#E3E7EB', linewidth=0.8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#BCC3CA')
            ax.tick_params(axis='both', length=3, color='#BCC3CA')
    axes[0].set_ylabel('Mean precision at 10', labelpad=10)
    handles = []
    for _, title, color in MODELS:
        handles.append(Line2D([], [], color=color, linewidth=4, label=title))
    for label, style in (('Mean pooling', '-'), ('Centre pooling', '--')):
        handles.append(Line2D([], [], color='#444444', linestyle=style,
            linewidth=1.95, label=label))
    fig.suptitle('Word-aligned and across-word retrieval', fontsize=18,
        y=0.97)
    subtitle = '31 tokens per type | 2,000 ms context per side | '
    subtitle += '95% type-level BCa CI'
    fig.text(0.5, 0.915, subtitle,
        ha='center', fontsize=10, color='#59636D')
    fig.legend(handles=handles, loc='upper center', ncol=4,
        bbox_to_anchor=(0.5, 0.88), frameon=False, handlelength=2.5,
        columnspacing=1.7)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.12, top=0.76,
        wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg', '.pdf'):
        fig.savefig(output.with_suffix(suffix), dpi=200)
    plt.close(fig)
    report = {'metric': 'precision_at_10', 'tokens_per_type': 31,
        'collar_ms': 2000, 'bootstrap_method': 'BCa',
        'bootstrap_resamples': N_RESAMPLES, 'bootstrap_seed': SEED,
        'bootstrap_unit': 'type', 'confidence_level': 0.95,
        'input_sha256': sha256(data_path.read_bytes()).hexdigest(),
        'selection_fingerprints': identities, 'conditions': rows}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.data, args.output)


if __name__ == '__main__':
    main()
