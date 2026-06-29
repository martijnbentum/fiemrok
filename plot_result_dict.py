import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# default results file, resolved relative to this module (../all_results.json)
_RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'all_results.json')

# display name and panel order, matched on a substring of the model key
_PANELS = [
    ('wav2vec', 'Wav2Vec 2.0'),
    ('huibert', 'HuBERT'),
]

# colorblind-friendly (Okabe-Ito) color + distinct marker per condition
_CONDITIONS = [
    ('aligned', 'Aligned', '#0072B2', 'o'),
    ('misaligned', 'Misaligned', '#D55E00', 's'),
]


def _mean_ci(values, confidence=0.95):
    '''return (mean, half-width of the confidence interval).
    values                 sequence of accuracy values
    confidence             confidence level for the interval
    '''
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = a.mean()
    if n < 2:
        return mean, 0.0
    sem = a.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.5 + confidence / 2, df=n - 1)
    return mean, crit * sem


def _find_model_key(o, substring):
    '''return the model key in o whose name contains substring.
    o                      results dict keyed by model name
    substring              lowercase fragment to match
    '''
    for key in o:
        if substring in key.lower():
            return key
    return None


def plot_accuracy(o=None, confidence=0.95):
    '''plot mean accuracy per layer, one panel per model (side by side),
    with a line per condition (aligned, misaligned) and a 95% CI band.
    o                      results dict model -> layer -> condition -> values;
                           if None, loaded from ../all_results.json
    confidence             confidence level for the interval band
    '''
    if o is None:
        with open(_RESULTS_PATH) as f:
            o = json.load(f)

    fig, axes = plt.subplots(1, len(_PANELS), figsize=(7 * len(_PANELS), 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (substring, panel_title) in zip(axes, _PANELS):
        model_key = _find_model_key(o, substring)
        model = o[model_key]
        layers = sorted(int(layer) for layer in model)  # ascending x

        for condition, label, color, marker in _CONDITIONS:
            means, errs = [], []
            for layer in layers:
                mean, ci = _mean_ci(model[str(layer)][condition], confidence)
                means.append(mean)
                errs.append(ci)
            means, errs = np.array(means), np.array(errs)

            ax.plot(layers, means, marker=marker, color=color, label=label)
            ax.fill_between(layers, means - errs, means + errs,
                            color=color, alpha=0.2)

        ax.set_xticks(layers)
        ax.set_xlabel('layer')
        ax.set_title(panel_title)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('accuracy')
    axes[0].legend()  # legend only in the left panel
    fig.suptitle('Word embedding retrieval based on central vowel frame')
    fig.tight_layout()
    return fig
