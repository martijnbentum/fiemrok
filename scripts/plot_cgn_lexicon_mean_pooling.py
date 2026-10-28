'''Plot mean P@10 with word-level bootstrap confidence intervals.'''

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import bootstrap

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notes/assets/cgn_lexicon_mean_pooling'
LAYERS = [3, 6, 9, 12]
N_RESAMPLES = 10000
SEED = 20261028
FILES = {
    'Wav2Vec 2.0': 'cgn_lexicon_w2v_mean_results.json',
    'HuBERT': 'cgn_lexicon_hubert1_mean_results.json',
}


def word_summary(scores):
    '''Return the word mean and its 95% BCa bootstrap bounds.'''
    values = np.asarray([scores[word] for word in sorted(scores)], dtype=float)
    if values.shape != (2466, 100):
        raise ValueError('expected 2,466 words with 100 scores each')
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('scores must be finite and between 0 and 1')
    word_means = values.mean(axis=1)
    result = bootstrap((word_means,), np.mean, n_resamples=N_RESAMPLES,
        batch=128, confidence_level=0.95, method='BCa',
        rng=np.random.default_rng(SEED))
    low, high = result.confidence_interval
    return word_means.mean(), low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
        default=ROOT.parent / 'svs_data', help='directory of saved scores')
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ['steelblue', '#c97732']
    width = 0.36
    words = None
    for index, (model, filename) in enumerate(FILES.items()):
        results = json.loads((args.data_dir / filename).read_text())
        summaries = []
        for layer in LAYERS:
            scores = results[str(layer)]
            if words is None: words = set(scores)
            if set(scores) != words:
                raise ValueError('word sets must match across all conditions')
            summary = word_summary(scores)
            summaries.append(summary)
            mean, low, high = summary
            print(f'{model} | {layer} | {mean:.4f} | '
                f'[{low:.4f}, {high:.4f}]')
        means, lows, highs = np.asarray(summaries).T
        intervals = np.asarray([means - lows, highs - means])
        positions = [x + (index - 0.5) * width for x in range(len(LAYERS))]
        bars = ax.bar(positions, means, width=width, yerr=intervals,
            capsize=4, label=model, color=colors[index], alpha=0.9)
        ax.bar_label(bars, labels=[f'{mean:.4f}' for mean in means],
            padding=9, fontsize=9)
    ax.set_xticks(range(len(LAYERS)), labels=LAYERS)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean precision at 10')
    ax.set_ylim(0, 1)
    ax.set_title('CGN lexicon: mean-pooled word embeddings')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    fig.text(0.5, 0.015,
        '95% BCa bootstrap CIs over 2,466 word types; 10,000 resamples.',
        ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=180)
    svg_path = OUTPUT.with_suffix('.svg')
    fig.savefig(svg_path)
    lines = svg_path.read_text().splitlines()
    svg_path.write_text('\n'.join(line.rstrip() for line in lines) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
