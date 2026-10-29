# CGN retrieval for word-aligned and across-word segments

The **Word-aligned** condition uses the CGN lexicon: 2,466 word types
with 100 tokens per type, giving 246,600 queries per model and layer.
CGN components a, c, d, h, and m are excluded, as are tokens shorter than
0.1 seconds or longer than 1.5 seconds. See [CGN lexicon](cgn_lexicon.md)
for the lexicon selection.

The **Across-word** condition uses 2,200 phoneme-sequence types spanning
word boundaries, with 100 tokens per type and 220,000 queries per model
and layer. See the [cross-boundary evaluation procedure](../data/cross_boundary_neighbours.md).

Each query is scored by the proportion of its ten nearest non-self
neighbours with the same label (precision at 10). Word-aligned labels are
words; across-word labels are phoneme sequences. Self-matches are excluded
by source key. Mean pooling averages each token's stored embedding frames;
centre pooling selects its centre frame. Both query and reference vectors
use the chosen pooling method. Vector pooling is implemented in
[validate_lexicon.py](../validate_lexicon.py), and scoring in
[analyse_results.py](../analyse_results.py).

## Retrieval across conditions

![Word-aligned and across-word retrieval](assets/lexicon_cross_boundary_by_dataset.png)

Panels show the two datasets; colour identifies the model. Solid lines show
mean pooling and dashed lines show centre pooling. Error bars show 95%
type-level BCa bootstrap confidence intervals.

HuBERT peaks at layer 9, and Wav2Vec 2.0 at layer 6, for both datasets and
pooling methods. Mean pooling improves all evaluated conditions except
Wav2Vec layer 12. The highest P@10 is 81.09% for word-aligned segments and
56.56% for across-word segments, both using mean-pooled HuBERT layer 9.

The dataset comparison is descriptive: type inventories, label definitions,
token samples, and retrieval indexes differ. The performance gap does not
isolate a causal effect of crossing a word boundary.

## Best-layer results

For each dataset, model, and pooling method, the table selects the highest
mean P@10 among layers 3, 6, 9, and 12. Scores are percentages.

| Dataset | Model | Best layer | Centre pooling (%) | Mean pooling (%) |
|---|---|---:|---:|---:|
| Word-aligned | Wav2Vec 2.0 | 6 | 46.27 | 63.94 |
| Word-aligned | HuBERT | 9 | 61.76 | 81.09 |
| Across-word | Wav2Vec 2.0 | 6 | 30.55 | 44.43 |
| Across-word | HuBERT | 9 | 37.42 | 56.56 |

## Means and error bars

For each type, its 100 token scores are averaged. Each plotted point is the
mean of these type averages, giving each type equal weight. Because every
type has exactly 100 tokens, this also equals the mean over all token scores.

Error bars show 95% bias-corrected and accelerated (BCa) bootstrap confidence
intervals. Each of 10,000 resamples draws types with replacement: 2,466
word types for word-aligned segments, or 2,200 phoneme-sequence types for
across-word segments. Resampling whole type averages keeps each type's
tokens together. The calculation uses seed 20261029, sorted type identities,
and the same resampling indices across models, layers, and pooling methods
within each dataset. Interval endpoints can be asymmetric around the mean.

These intervals treat types as independent sampling units and describe
uncertainty across types, conditional on the sampled tokens and fixed
retrieval index. They do not measure variation from selecting different
speakers or recordings, rebuilding the index, or retraining models. They
are intervals for individual means. A comparison between conditions within
a dataset should use an interval on their paired per-type differences.

## Data and reproduction

All eight input files are in `svs_data/` relative to the repository root:

- `cgn_lexicon_{w2v,hubert1}_{centre,mean}_results.json`
- `cross_boundary_lexicon100_2200_{w2v,hubert1}_{center,mean}_results.json`

Braces indicate filename alternatives. Each file maps layer to type to
100 token scores. All files contain layers 3, 6, 9, and 12, with matching
type sets across conditions within each dataset. See the
[download and validation note](lexicon_cross_boundary_results.md).

From the repository root, run:

```bash
python scripts/plot_lexicon_cross_boundary.py
```

Use `--data-dir /path/to/svs_data` for another input directory. The script
requires NumPy, SciPy, and Matplotlib, validates score counts, ranges, and
type sets, and prints means and confidence interval endpoints. It generates
[PNG](assets/lexicon_cross_boundary_by_dataset.png),
[SVG](assets/lexicon_cross_boundary_by_dataset.svg), and
[PDF](assets/lexicon_cross_boundary_by_dataset.pdf) from saved scores
without rerunning retrieval.

## All layer results

Mean P@10 is expressed as a percentage for every dataset, model, pooling
method, and evaluated layer.

| Dataset | Model | Layer | Centre pooling (%) | Mean pooling (%) |
|---|---|---:|---:|---:|
| Word-aligned | Wav2Vec 2.0 | 3 | 11.89 | 19.40 |
| Word-aligned | Wav2Vec 2.0 | 6 | 46.27 | 63.94 |
| Word-aligned | Wav2Vec 2.0 | 9 | 21.18 | 27.21 |
| Word-aligned | Wav2Vec 2.0 | 12 | 5.31 | 4.57 |
| Word-aligned | HuBERT | 3 | 11.30 | 18.64 |
| Word-aligned | HuBERT | 6 | 25.49 | 54.27 |
| Word-aligned | HuBERT | 9 | 61.76 | 81.09 |
| Word-aligned | HuBERT | 12 | 29.92 | 62.23 |
| Across-word | Wav2Vec 2.0 | 3 | 14.06 | 21.86 |
| Across-word | Wav2Vec 2.0 | 6 | 30.55 | 44.43 |
| Across-word | Wav2Vec 2.0 | 9 | 20.46 | 27.89 |
| Across-word | Wav2Vec 2.0 | 12 | 6.28 | 5.51 |
| Across-word | HuBERT | 3 | 12.99 | 20.01 |
| Across-word | HuBERT | 6 | 25.47 | 47.00 |
| Across-word | HuBERT | 9 | 37.42 | 56.56 |
| Across-word | HuBERT | 12 | 22.81 | 46.56 |
