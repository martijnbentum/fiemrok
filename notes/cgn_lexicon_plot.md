# CGN lexicon retrieval with centre- and mean-pooled word embeddings

The evaluation uses 2,466 CGN word types with 100 tokens per type, giving
246,600 queries for each model and layer. CGN components a, c, d, h, and m
are excluded, as are tokens shorter than 0.1 seconds or longer than
1.5 seconds. See [CGN lexicon](cgn_lexicon.md) for the lexicon selection.

Each query is scored by the proportion of its ten nearest non-self
neighbours with the same word label (precision at 10). Self-matches are
excluded by source key.
Vector pooling is implemented in [validate_lexicon.py](../validate_lexicon.py),
and scoring in [analyse_results.py](../analyse_results.py).

## Mean-pooling results

Each token is represented by the mean of its stored word embedding frames.
Both queries and reference vectors use this mean pooling.

![Mean P@10 with 95% word-level bootstrap confidence intervals](assets/cgn_lexicon_mean_pooling.png)

| Model | Layer | Mean P@10 | 95% bootstrap CI |
|---|---:|---:|---:|
| Wav2Vec 2.0 | 3 | 0.1940 | [0.1892, 0.1992] |
| Wav2Vec 2.0 | 6 | 0.6394 | [0.6318, 0.6468] |
| Wav2Vec 2.0 | 9 | 0.2721 | [0.2663, 0.2779] |
| Wav2Vec 2.0 | 12 | 0.0457 | [0.0440, 0.0477] |
| HuBERT | 3 | 0.1864 | [0.1813, 0.1917] |
| HuBERT | 6 | 0.5427 | [0.5347, 0.5507] |
| HuBERT | 9 | 0.8109 | [0.8047, 0.8167] |
| HuBERT | 12 | 0.6223 | [0.6149, 0.6297] |

HuBERT has its highest mean P@10 at layer 9 (0.8109), and Wav2Vec 2.0 at
layer 6 (0.6394). The difference between these highest means is
17.16 percentage points, calculated from the unrounded scores.

### Means and error bars

For each word type, the 100 token scores are averaged. Each bar shows the
mean of these 2,466 word averages, giving every word type equal weight.
Because each type has exactly 100 tokens, this also equals the mean over
all token scores.

Error bars show 95% bias-corrected and accelerated (BCa) bootstrap
confidence intervals. Each of 10,000 resamples selects 2,466 word types
with replacement and averages their word scores. Resampling whole word
averages keeps the tokens of each word together. The calculation uses a
fixed random seed of 20261028 and the same ordered word types and resampling
indices across all models and layers. Interval endpoints can be asymmetric
around the mean.

These intervals treat word types as independent sampling units and describe
uncertainty in mean performance across word types, conditional on the
sampled tokens and the fixed retrieval index. They do not measure variation
from selecting different speakers or recordings, rebuilding the reference
index, or retraining models. A comparison between two conditions should use
an interval on their paired per-word differences.

## Centre-pooling results


![Centre-pooled CGN lexicon precision by layer](https://github.com/user-attachments/assets/025b5358-3c1e-4337-a0e7-4b839ef8b80b)

The figure reports 95% confidence intervals. Their calculation cannot be
verified from the figure alone; the word-level bootstrap method described
above applies to the mean-pooling figure.



## Mean-pooling data and reproduction

The inputs are `cgn_lexicon_w2v_mean_results.json` and
`cgn_lexicon_hubert1_mean_results.json` in `../svs_data/`, relative to the
repository root. Each file maps layer to word type to 100 token scores.
Both files contain layers 3, 6, 9, and 12 with identical sets of 2,466 words.

From the repository root, run:

```bash
python scripts/plot_cgn_lexicon_mean_pooling.py
```

Use `--data-dir /path/to/svs_data` for another input directory. The script
requires NumPy, SciPy, and Matplotlib, validates score counts, ranges, and
word sets, and prints the means and interval endpoints. It generates the
[PNG](assets/cgn_lexicon_mean_pooling.png) and
[SVG](assets/cgn_lexicon_mean_pooling.svg) directly from saved scores without
rerunning retrieval.
