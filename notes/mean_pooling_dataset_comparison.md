# Word-aligned versus cross-boundary retrieval: mean pooling

Mean pooling only; computed from the four existing score files. Here
word-aligned means complete word tokens, not arbitrary within-word spans.
The gap is word-aligned minus cross-boundary P@10, in percentage points.

## Dataset differences

| Model | Layer | Word (%) | Cross-boundary (%) | Gap (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| Wav2Vec 2.0 | 3 | 19.40 | 21.86 | -2.46 | [-3.24, -1.67] |
| Wav2Vec 2.0 | 6 | 63.94 | 44.43 | 19.51 | [18.37, 20.65] |
| Wav2Vec 2.0 | 9 | 27.21 | 27.89 | -0.68 | [-1.58, 0.21] |
| Wav2Vec 2.0 | 12 | 4.57 | 5.51 | -0.93 | [-1.22, -0.65] |
| HuBERT | 3 | 18.64 | 20.01 | -1.38 | [-2.13, -0.59] |
| HuBERT | 6 | 54.27 | 47.00 | 7.27 | [6.10, 8.49] |
| HuBERT | 9 | 81.09 | 56.56 | 24.53 | [23.46, 25.62] |
| HuBERT | 12 | 62.23 | 46.56 | 15.67 | [14.54, 16.83] |

The word advantage emerges at the stronger retrieval layers; it is absent
at layer 3 in both models. This is compatible with a layer-dependent
advantage for whole-word identity, but does not establish lexical encoding
or a causal boundary effect. Inventories, label definitions, token samples,
and candidate indexes differ.

Intervals use 10,000 independent type bootstrap resamples per dataset,
95% BCa intervals, and seed 20261029. Each type contributes the mean of
its 100 token scores. Types cannot be paired across datasets. These are
pointwise, exploratory intervals, without adjustment for eight comparisons
or selecting the strongest tested layers. They assume independent types
and independent dataset samples; dependencies from shared recordings,
speakers, and retrieval candidates are not modeled. They are conditional
on the existing tokens and indexes, not estimates over new corpus samples.

## How failures are distributed

At each model's strongest tested layer (Wav2Vec 6; HuBERT 9):

| Model | Dataset | Type median P@10 (%) | Type Q1–Q3 (%) | Queries with 0/10 correct (%) | Queries with 10/10 correct (%) | Types averaging below 50% (%) |
|---|---|---:|---:|---:|---:|---:|
| Wav2Vec 2.0 | Word | 65.60 | 50.63–79.50 | 8.30 | 26.21 | 23.88 |
| Wav2Vec 2.0 | Cross-boundary | 41.80 | 28.08–58.50 | 16.26 | 11.37 | 63.45 |
| HuBERT | Word | 85.30 | 72.80–92.90 | 5.28 | 56.20 | 4.01 |
| HuBERT | Cross-boundary | 55.60 | 39.80–73.52 | 11.09 | 22.49 | 41.09 |

The gap extends across the type-score distribution, rather than being
confined to a few exceptionally difficult types. Cross-boundary queries
have roughly twice the rate of complete retrieval failure, and far fewer
perfect neighbour sets. For HuBERT, the cross-boundary upper quartile
(73.52%) is close to the word lower quartile (72.80%).

Type difficulty is strongly shared between models: Spearman correlation
between Wav2Vec layer 6 and HuBERT layer 9 type means is 0.944 for words
and 0.936 for cross-boundary sequences. This compares types within each
dataset, not types across datasets, and does not establish shared wrong
neighbours or token-level agreement.

Among HuBERT's hardest word types are `hum` (11.0%), `hé` (11.7%),
`be` (17.5%), `hu` (17.8%), and `ja` (18.9%). Its hardest cross-boundary
types include `ə n w ə` (4.1%), `ə m d ɑ` (6.6%), and `ɑ t ə m` (7.5%).
The latter three also occur among Wav2Vec's five hardest types. These are
useful targets for inspection; the scores alone do not explain the failures.

## Can we establish different kinds of mistakes?

Not from these saved scores. `analyse_results.evaluate_index()` retains
only label-to-token-score lists. The local `svs_data/` contains the eight
score files, without query source keys, neighbour labels, neighbour source
keys, distances, or phoneme alignments. Historical word-neighbour examples
in `cgn_lexicon.md` are not a matched comparison for these mean-pooling runs.

Next, export neighbour-level results for HuBERT layer 9 in both datasets,
starting with a reproducible random sample for prevalence estimates and
the difficult types above for diagnosis. Use the existing mean-pooled
indexes if available on the server; rebuilding embeddings is not necessary
if those indexes retain their vectors and metadata. Rerun retrieval for
the selected queries and retain the same self-exclusion rule.

Record query and neighbour source keys, labels, ranks, distances, and
correctness. Join source metadata for speaker/recording, duration, word
and phone alignments, source words, and boundary position. Then compare:

- Phonetic proximity of wrong neighbours, using phoneme edit distance
  rather than comparing word spelling to phone strings.
- Shared prefixes/suffixes versus substitutions and span truncation;
  identify which side of the boundary is preserved or confused.
- Alignment problems, reductions, and pronunciation variants through
  inspection of a balanced audio sample.
- Concentration of mistakes within speakers or recordings.

Report error categories as proportions of wrong neighbours, alongside
overall error rates, so the larger number of cross-boundary errors does
not alone imply a different error composition. Use source pronunciations
and account for homophones when comparing the two label schemes.

## Reproduction

```bash
python scripts/compare_mean_pooling_datasets.py
```

Requires NumPy and SciPy. Inputs default to `svs_data/`; `--data-dir`
overrides this. The script validates type identities, counts, ranges, and
P@10 score increments. It writes all gaps, confidence intervals, score
histograms, hardest-type lists, and correlations to
[`data/mean_pooling_dataset_comparison.json`](../data/mean_pooling_dataset_comparison.json).
No retrieval is rerun. The analysis script is
[`scripts/compare_mean_pooling_datasets.py`](../scripts/compare_mean_pooling_datasets.py).
