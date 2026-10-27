'''Summarise per-word frame counts and identify inspection candidates.'''

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'cgn_lexicon_frame_counts.json'
OUTPUT = ROOT / 'data/cgn_lexicon_frame_count_analysis.json'
FIGURE = ROOT / 'notes/assets/cgn_lexicon_frame_counts.png'


def analyse_word(counts):
    '''Return descriptive statistics and Tukey outliers for one word.'''
    values = np.asarray(counts)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outer_low, outer_high = q1 - 3 * iqr, q3 + 3 * iqr
    outliers = []
    for index, value in enumerate(counts):
        if low <= value <= high: continue
        outliers.append({
            'token_index': index,
            'frames': value,
            'direction': 'short' if value < low else 'long',
            'ratio_to_word_median': value / float(median),
            'outside_outer_fence': bool(
                value < outer_low or value > outer_high),
        })
    return {
        'n_tokens': len(counts),
        'min': int(values.min()),
        'q1': float(q1),
        'median': float(median),
        'q3': float(q3),
        'max': int(values.max()),
        'mean': float(values.mean()),
        'standard_deviation': float(values.std(ddof=1)),
        'coefficient_of_variation': float(values.std(ddof=1) / values.mean()),
        'iqr': float(iqr),
        'relative_iqr': float(iqr / median),
        'median_absolute_deviation': float(np.median(abs(values - median))),
        'lower_fence': float(low),
        'upper_fence': float(high),
        'outer_lower_fence': float(outer_low),
        'outer_upper_fence': float(outer_high),
        'n_short_outliers': sum(x['direction'] == 'short' for x in outliers),
        'n_long_outliers': sum(x['direction'] == 'long' for x in outliers),
        'n_outer_outliers': sum(x['outside_outer_fence'] for x in outliers),
        'outliers': outliers,
    }


def word_table(words, stats):
    '''Format selected word summaries as a Markdown table.'''
    lines = ['| Word | Median | Mean | Range | IQR | Short flags | Long flags |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for word in words:
        row = stats[word]
        lines.append(f"| {word} | {row['median']:g} | {row['mean']:.2f} | "
            f"{row['min']}–{row['max']} | {row['iqr']:g} | "
            f"{row['n_short_outliers']} | {row['n_long_outliers']} |")
    return '\n'.join(lines)


def token_table(events):
    '''Format flagged tokens with their positions in the source JSON.'''
    lines = ['| Word | Token index (0-based) | Frames | Word median | Ratio |',
        '| --- | ---: | ---: | ---: | ---: |']
    for word, event, median in events:
        lines.append(f"| {word} | {event['token_index']} | "
            f"{event['frames']} | {median:g} | "
            f"{event['ratio_to_word_median']:.2f} |")
    return '\n'.join(lines)


def plot_counts(data, stats, selected):
    '''Plot per-word medians/spread and selected token distributions.'''
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 7),
        gridspec_kw={'width_ratios': [1, 1.3]})
    medians = [row['median'] for row in stats.values()]
    spreads = [row['iqr'] for row in stats.values()]
    flags = [row['n_short_outliers'] + row['n_long_outliers']
        for row in stats.values()]
    scatter = axes[0].scatter(medians, spreads, c=flags, s=13,
        alpha=.65, cmap='viridis', edgecolors='none')
    for word in sorted(stats, key=lambda w: -stats[w]['iqr'])[:2]:
        axes[0].annotate(word, (stats[word]['median'], stats[word]['iqr']),
            xytext=(5, 0), textcoords='offset points', fontsize=9)
    axes[0].set(xlabel='Word median (frames)', ylabel='Within-word IQR (frames)',
        title='Each point is one word type')
    fig.colorbar(scatter, ax=axes[0], label='Flagged tokens out of 100')
    positions = np.arange(len(selected))
    rng = np.random.default_rng(0)
    for position, word in zip(positions, selected):
        counts = np.asarray(data[word])
        row = stats[word]
        mask = (counts < row['lower_fence']) | (counts > row['upper_fence'])
        jitter = rng.uniform(-.16, .16, len(counts))
        axes[1].scatter(counts, position + jitter, s=10,
            c=np.where(mask, '#c43c39', '#999999'), alpha=.65)
        axes[1].plot(row['median'], position, '|', color='black',
            markersize=17, markeredgewidth=2)
    axes[1].set_yticks(positions, selected)
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Stored frames per token',
        title='Selected inspection candidates\nRed: 1.5-IQR flags; black: median')
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


def main():
    data = json.loads(SOURCE.read_text())
    for word, counts in data.items():
        if not counts or any(type(x) is not int or x <= 0 for x in counts):
            raise ValueError(f'invalid frame counts for {word}')
    stats = {word: analyse_word(counts) for word, counts in data.items()}
    events = [(word, event, row['median']) for word, row in stats.items()
        for event in row['outliers']]
    short = sorted((e for e in events if e[1]['direction'] == 'short'),
        key=lambda e: (e[1]['ratio_to_word_median'], e[0]))
    long = sorted((e for e in events if e[1]['direction'] == 'long'),
        key=lambda e: (-e[1]['ratio_to_word_median'], e[0]))
    variable = sorted(stats,
        key=lambda w: -stats[w]['coefficient_of_variation'])
    robust_variable = sorted(stats, key=lambda w: -stats[w]['relative_iqr'])
    shortest = sorted(stats, key=lambda w: (stats[w]['median'], w))
    longest = sorted(stats, key=lambda w: (-stats[w]['median'], w))
    n_tokens = sum(len(v) for v in data.values())
    n_flagged_words = sum(bool(r['outliers']) for r in stats.values())
    outer = sum(r['n_outer_outliers'] for r in stats.values())
    summary = {
        'n_word_types': len(stats), 'n_tokens': n_tokens,
        'n_flagged_tokens': len(events), 'n_flagged_words': n_flagged_words,
        'n_short_outliers': len(short), 'n_long_outliers': len(long),
        'n_outer_outliers': outer,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        'source': SOURCE.name,
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'method': 'Per-word Tukey fences: Q1 - 1.5 IQR, Q3 + 1.5 IQR; '
            'outer fences use 3 IQR; strict inequalities; '
            'NumPy linear percentile interpolation; sample standard deviation.',
        'token_index': 'Zero-based position in that word’s source JSON list; '
            'not an embedding key or global index.',
        'summary': summary, 'word_types': stats,
    }, ensure_ascii=False, indent=2) + '\n')
    selected = list(dict.fromkeys([e[0] for e in long[:8]]
        + [e[0] for e in short[:6]]))
    plot_counts(data, stats, selected)
    report = f'''# CGN lexicon frame counts by word type

Source: `cgn_lexicon_frame_counts.json`, {len(stats):,} word types and
{n_tokens:,} tokens, with 100 tokens per type. All counts are positive integers.
Counts describe stored embedding rows, not independently verified acoustic
durations. The source does not record model, layer, frame stride or token keys.

## Method

Each token is compared with other tokens of the **same word type**.
A short outlier is below Q1 − 1.5 × IQR; a long outlier is above
Q3 + 1.5 × IQR. Values exactly on a fence are not flagged. Quartiles use
NumPy's linear interpolation. Outer fences at 3 × IQR provide a stricter
inspection subset. No word has zero IQR in this dataset.

This flags {len(events):,} tokens ({len(events) / n_tokens:.2%}):
{len(short):,} short and {len(long):,} long, spread across
{n_flagged_words:,} word types. Of these, {outer:,} tokens also exceed an
outer fence. With 100 observations per word, an occasional flag is common;
the presence of any outlier does not establish a faulty word type.

These are inspection candidates, not confirmed alignment errors. Duration
distributions are asymmetric, and long tails may reflect legitimate prosodic
lengthening, reduction differences or annotation problems. The 0.1–1.5-second
selection window also truncates the distribution. Short-token detection is
particularly constrained by the four-frame floor observed in the data.

## Overview

![Per-word spread and selected token distributions](assets/cgn_lexicon_frame_counts.png)

The left panel compares word medians and within-word IQRs. Colour is the
number of flagged tokens, not retrieval accuracy. The right panel shows
tokens from selected extreme-ratio words; red points are flags and black
ticks are word medians. Jitter separates overlapping points and has no
substantive meaning.

## Unusually long tokens relative to their word

Prioritised by token count divided by its word median. Token indices are
zero-based positions in each word's original JSON list. For instance, index
14 is the fifteenth count for that word, not global index 14.

{token_table(long[:20])}

## Unusually short tokens relative to their word

{token_table(short[:20])}

## Word types with the largest relative variability

Ranked by sample standard deviation / mean (coefficient of variation),
which is sensitive to extremes. IQR is included as a more robust measure;
high variability alone is not evidence of error.

{word_table(variable[:20], stats)}

### Broad central distributions

Ranked by IQR / median, which is less driven by isolated extremes than the
coefficient of variation. A broad or mixed distribution can have few or no
Tukey flags because its fences widen. In particular, `hm` has a median of
31 frames, IQR 37.25 and range 5–70, but no flagged tokens. This makes it
interesting to inspect despite passing the token-level screen. Acronyms
such as `cvp`, `vvd` and `vtm` also warrant checking for different spoken
realisations or annotation conventions; counts alone cannot establish these.

{word_table(robust_variable[:12], stats)}

## Word types with the shortest and longest typical counts

These rankings describe between-word differences, not a second error flag.
Longer words naturally tend to have larger medians; no correction for
phoneme/syllable count or speaking rate is available in this JSON.

### Shortest medians

{word_table(shortest[:12], stats)}

### Longest medians

{word_table(longest[:12], stats)}

## Inspection and reproducibility

Start with the largest long-token ratios (`de`, `der`, `ik`, `en`) and the
smallest short-token ratios (`huilen`, `sfeer`, `lijden`). Check the actual
audio and word boundaries before excluding anything. To locate a token,
recover that word's source-key list in the same order as when the counts
were generated. The count alone cannot identify a token; a newly generated
or differently ordered index must not be assumed to preserve this mapping.

The source counts and complete [per-word analysis](../data/cgn_lexicon_frame_count_analysis.json)
are local JSON files ignored by Git. The analysis contains every word's
statistics, fences and flagged token positions, together with a source hash.
To regenerate the analysis and figure, and print this report to stdout,
run:

```bash
python scripts/analyse_lexicon_frame_counts.py
```
'''
    print(report)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    print(f'Wrote {OUTPUT} and {FIGURE}', file=sys.stderr)


if __name__ == '__main__':
    main()
