# Neighbour retrieval overview

Nearest-neighbour exports for word-aligned and across-word segments were
scanned for two models (HuBERT, Wav2Vec 2.0), four layers (3, 6, 9, 12), and
two pooling methods (centre, mean). Word-aligned segments are retrieved more
accurately than across-word segments at every metric in the best conditions,
and mean pooling improves retrieval almost everywhere. The best condition for
both datasets is mean-pooled HuBERT layer 9: P@10 is 81.09% for word-aligned
and 56.56% for across-word segments.

P@10 with type-level BCa confidence intervals, the plot, and the summary of
the word-aligned/across-word comparison are in
[CGN retrieval for word-aligned and across-word segments](cgn_lexicon_plot.md).
This note adds P@1, P@30, match-count distributions, and top-1 confusions.
It reports no intervals of its own.

## Datasets and metrics

The word-aligned index has 2,466 word types and the across-word index has
2,200 phoneme-sequence types. Every type has exactly 100 tokens, every token
is a query against its own index, and every query has 30 non-self
neighbours. Word-aligned labels are orthographic words; across-word labels
are phoneme sequences.

- **P@k:** percentage of the first k neighbours with the exact query label.
  P@1 is nearest-neighbour label accuracy.
- **Hit@30:** percentage of queries with at least one matching label among
  the first 30 neighbours. Each query has 99 same-label tokens in its index,
  so a high Hit@30 is expected and says little on its own. R-precision
  (P@99) would be a better summary but needs a top-100 export.
- **No match in top 10:** percentage of queries with none of the first 10
  neighbours matching.
- **All top 10 match:** percentage of queries with all of the first 10
  neighbours matching.

## Best-layer results

Mean pooling, best of the sampled layers 3, 6, 9, and 12:

| Dataset | Model / layer | P@1 | P@10 | P@30 | Hit@30 | No match in top 10 | All top 10 match |
|---|---|---:|---:|---:|---:|---:|---:|
| Word-aligned | HuBERT / 9 | 86.25% | 81.09% | 75.42% | 95.97% | 5.28% | 56.20% |
| Word-aligned | Wav2Vec 2.0 / 6 | 74.74% | 63.94% | 53.71% | 94.17% | 8.30% | 26.21% |
| Across-word | HuBERT / 9 | 67.62% | 56.56% | 47.16% | 92.74% | 11.09% | 22.49% |
| Across-word | Wav2Vec 2.0 / 6 | 57.55% | 44.43% | 34.80% | 89.45% | 16.26% | 11.37% |

Among the sampled layers, HuBERT is best at layer 9 and Wav2Vec 2.0 at
layer 6 for P@1, P@10, and P@30 in both datasets and pooling methods. The
true peaks may lie between sampled layers. Mean pooling improves P@10 at
every layer except Wav2Vec 2.0 layer 12. At the best layers, mean pooling
raises P@10 over centre pooling by 19.34 percentage points for HuBERT
word-aligned, 19.14 for HuBERT across-word, 17.67 for Wav2Vec 2.0
word-aligned, and 13.88 for Wav2Vec 2.0 across-word. Differences are
computed from unrounded scores.

The word-aligned/across-word difference is descriptive: inventories, labels,
sampled tokens, and indexes differ. It does not isolate a causal effect of
crossing a word boundary.

## Top-1 confusions

The tables show the six most frequent directed top-1 label confusions in
each best mean-pooled condition. Counts are out of 100 queries of the source
type. These are selected common pairs, not a classification of all errors.
The relation column is assigned by hand for these pairs only. No listening,
edit-distance analysis of all errors, or annotation audit was performed.

In these pairs, word-aligned confusions are all between inflected forms of
one lemma. Several are probably homophones or near-homophones in connected
speech: *verandert*/*veranderd* through final devoicing, and plural or past
tense *-n* pairs because final *-n* after schwa is often not pronounced.
Across-word confusions differ by one phone: a substitution, an insertion, a
transposition, or one label nested within the other.

**Word-aligned: HuBERT layer 9, mean pooling**

| Query label | Nearest-neighbour label | Relation | Count |
|---|---|---|---:|
| voorwaarde | voorwaarden | -n inflection | 40 |
| maakte | maakten | -n inflection | 40 |
| verandert | veranderd | homophone (final devoicing) | 40 |
| maakten | maakte | -n inflection | 39 |
| gemeenten | gemeente | -n inflection | 38 |
| voorwaarden | voorwaarde | -n inflection | 36 |

**Word-aligned: Wav2Vec 2.0 layer 6, mean pooling**

| Query label | Nearest-neighbour label | Relation | Count |
|---|---|---|---:|
| manier | manieren | -en inflection | 49 |
| onderwerp | onderwerpen | -en inflection | 46 |
| interessant | interessante | -e inflection | 44 |
| raakte | raakten | -n inflection | 44 |
| economisch | economische | -e inflection | 43 |
| persoonlijk | persoonlijke | -e inflection | 42 |

**Across-word: HuBERT layer 9, mean pooling**

| Query label | Nearest-neighbour label | Relation | Count |
|---|---|---|---:|
| m aː n t ɑ | n aː n t ɑ | substitution | 49 |
| ə ɣ ə l eː | ə x ə l eː | substitution | 48 |
| n ə r ə d | n d ə r ə d | insertion | 47 |
| t x r oː | t x r oː t | nested | 47 |
| ə r eː ɣ ə l | ə r eː ɣ ə | nested | 44 |
| i d ɑ l eː | i t ɑ l eː | substitution | 43 |

**Across-word: Wav2Vec 2.0 layer 6, mean pooling**

| Query label | Nearest-neighbour label | Relation | Count |
|---|---|---|---:|
| n b eː t j | m b eː t j | substitution | 45 |
| ə r k ɑ n | r ə k ɑ n | transposition | 45 |
| t s ɑ n ə | t s ɑ n d ə | insertion | 44 |
| t x r oː | t x r oː t | nested | 42 |
| ə t eː l | ə t eː l ə | nested | 41 |
| ə m oː ɣ ə l ə | ə m oː ɣ ə l ə k | nested | 41 |

## Not yet checked

- How many neighbours share the query's speaker or recording, per dataset.
  Same-speaker or same-recording neighbours could inflate retrieval.
- Duration and phone-count distributions per dataset.
- Whether across-word tokens from the same recording can overlap in time.
  Self-matches are excluded by source key only.
- How many near-identical competitor types each inventory contains, such as
  nested sequences or inflection pairs.

## Validation

All 32 neighbour files passed the integrity checks. No duplicate queries
within a file, duplicate neighbours within a query, self-matches by source
key, non-finite scores, or descending-rank violations were found. These
checks validate the exports, not the underlying annotations.

The files contain 7,465,600 query rows and 223,968,000 neighbour records
(32,750,910,950 bytes). Each word-aligned file has 246,600 queries and each
across-word file has 220,000 queries.

Recomputed P@10 agrees with the older score exports to two decimal places.
Twelve files have small differences in the per-type score distributions (at
most 11 types in one file). The largest absolute net difference is five
matching neighbours per file, or 0.000203 percentage points. The cause was
not established; tied similarity scores ordered differently between runs are
a likely candidate. Net totals can hide offsetting changes.

## All conditions

Percentages; best-layer P@10 with confidence intervals is in
[cgn_lexicon_plot.md](cgn_lexicon_plot.md).

| Dataset | Model | Pooling | Layer | P@1 | P@10 | P@30 |
|---|---|---|---:|---:|---:|---:|
| Word-aligned | HuBERT | centre | 3 | 20.71 | 11.30 | 7.59 |
| Word-aligned | HuBERT | centre | 6 | 39.32 | 25.49 | 17.75 |
| Word-aligned | HuBERT | centre | 9 | 72.80 | 61.76 | 51.45 |
| Word-aligned | HuBERT | centre | 12 | 47.05 | 29.92 | 19.43 |
| Word-aligned | HuBERT | mean | 3 | 34.83 | 18.64 | 11.90 |
| Word-aligned | HuBERT | mean | 6 | 68.10 | 54.27 | 42.65 |
| Word-aligned | HuBERT | mean | 9 | 86.25 | 81.09 | 75.42 |
| Word-aligned | HuBERT | mean | 12 | 75.23 | 62.23 | 48.42 |
| Word-aligned | Wav2Vec 2.0 | centre | 3 | 21.58 | 11.89 | 7.98 |
| Word-aligned | Wav2Vec 2.0 | centre | 6 | 58.89 | 46.27 | 36.00 |
| Word-aligned | Wav2Vec 2.0 | centre | 9 | 35.72 | 21.18 | 13.72 |
| Word-aligned | Wav2Vec 2.0 | centre | 12 | 11.15 | 5.31 | 3.37 |
| Word-aligned | Wav2Vec 2.0 | mean | 3 | 35.69 | 19.40 | 12.33 |
| Word-aligned | Wav2Vec 2.0 | mean | 6 | 74.74 | 63.94 | 53.71 |
| Word-aligned | Wav2Vec 2.0 | mean | 9 | 45.66 | 27.21 | 17.52 |
| Word-aligned | Wav2Vec 2.0 | mean | 12 | 10.41 | 4.57 | 2.85 |
| Across-word | HuBERT | centre | 3 | 20.84 | 12.99 | 9.33 |
| Across-word | HuBERT | centre | 6 | 35.71 | 25.47 | 19.23 |
| Across-word | HuBERT | centre | 9 | 48.39 | 37.42 | 29.83 |
| Across-word | HuBERT | centre | 12 | 33.16 | 22.81 | 16.49 |
| Across-word | HuBERT | mean | 3 | 33.50 | 20.01 | 13.59 |
| Across-word | HuBERT | mean | 6 | 59.22 | 47.00 | 37.54 |
| Across-word | HuBERT | mean | 9 | 67.62 | 56.56 | 47.16 |
| Across-word | HuBERT | mean | 12 | 60.23 | 46.56 | 35.61 |
| Across-word | Wav2Vec 2.0 | centre | 3 | 22.34 | 14.06 | 10.13 |
| Across-word | Wav2Vec 2.0 | centre | 6 | 40.85 | 30.55 | 23.86 |
| Across-word | Wav2Vec 2.0 | centre | 9 | 30.75 | 20.46 | 14.59 |
| Across-word | Wav2Vec 2.0 | centre | 12 | 11.79 | 6.28 | 4.18 |
| Across-word | Wav2Vec 2.0 | mean | 3 | 35.80 | 21.86 | 14.85 |
| Across-word | Wav2Vec 2.0 | mean | 6 | 57.55 | 44.43 | 34.80 |
| Across-word | Wav2Vec 2.0 | mean | 9 | 43.78 | 27.89 | 18.94 |
| Across-word | Wav2Vec 2.0 | mean | 12 | 11.67 | 5.51 | 3.54 |

## Reproduction

The scan ran on 29 October 2026 on `fancypants.science.ru.nl` in the
`fiemrok_env` environment, over all 32 JSONL files in
`/vol/mlusers/mbentum/fiemrok/repo/svs_data/`. It took 386.6 seconds with
two worker processes. The raw files remain remote; only the summary was
downloaded.

- [Detailed JSON summary](../svs_data/neighbour_overview_20261029.json)
- [Analysis script](../svs_data/inspect_neighbour_results_20261029.py)

Both are in the ignored `svs_data/` directory. The script reads the remote
data path and takes the summary output path as its argument. The JSON also
includes differences against the prior local score files. The remote run
directory was in `/tmp` and may be cleared; the local copies above are the
retained record.
