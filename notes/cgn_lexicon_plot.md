selected 2466 word types from cgn with minimally 100 word token (excluded components a,c,d,h,m,
excluded words shorter than .1 second and longer than 1.5 seconds)

This plot summarises the same CGN lexicon as [cgn_lexicon.md](cgn_lexicon.md),
evaluating all 100 tokens per word type rather than one token per word type.
Word embeddings used the centre frame of each word's embedding frames,
as implemented by `load_speech_vectors()` in [validate_lexicon.py](../validate_lexicon.py).
The exact generating run was not recorded, so the centre-frame method is inferred from the code.
The per-token and per-word precision calculations are in [analyse_results.py](../analyse_results.py).

For a mean-pooling comparison, pass `pooling='mean'` to
`validate_lexicon.make_vsi_for_layers()` or
`make_vector_search_index_for_layer()`. This averages each token's stored
word frames and saves separate indexes such as `svs_data/w2v_6_mean.npy`
and `svs_data/w2v_6_mean.json`. The default `pooling='center'` retains the
original index names. Both query and reference vectors use the chosen method.
`analyse_results.evaluate_index()` returns per-token precision grouped by
word type, suitable for `per_word()`, `overall_result()`, and
`plot_precision()`. It requests 11 neighbours and excludes self-matches by
source key before scoring ten neighbours. The existing `analyse_results()`
function also excludes self-matches by key and scores exactly ten remaining
neighbours. Saved results with ten neighbours already excluding self can
be scored directly; results with fewer than ten non-self neighbours raise
an error.

Checks on 26 October 2026 found that `svs_data/results_w2v_results.pkl`
and `svs_data/hubert_results.pkl` contain 246,600 queries per layer
(layers 3, 6, 9, and 12). Each query has ten saved neighbours. A source-key
check for Wav2Vec 2.0 layer 6 found zero self-matches across all 246,600
queries. These saved results already exclude self; centre pooling retrieval
does not need rerunning because of the self-match concern. The original
centre pooling scores remain the baseline for this comparison.

The separate `analyse_results()` function was corrected on 26 October to
score ten non-self neighbours instead of assuming one neighbour must be
discarded. That code correction does not establish an error in the original
plotted centre pooling scores. Mean pooling uses `evaluate_index()`, which
already scored ten non-self neighbours correctly.

Both saved-result analysis and new retrieval assume the correct
query-to-metadata correspondence. This comparison uses the available store
entries; reproducing the original token set requires checking source keys
against the original saved index.

Wav2Vec 2.0 mean pooling results reported on 27 October 2026:

| Layer | Centre pooling (approx.) | Mean pooling | Gain (percentage points) |
|---|---:|---:|---:|
| 3 | 0.125 | 0.1940 | 6.9 |
| 6 | 0.470 | 0.6394 | 16.9 |
| 9 | 0.220 | 0.2721 | 5.2 |

Centre pooling values are approximate readings from the plot below, so the
gains are approximate too. Mean pooling improves all three reported layers;
layer 6 remains strongest and has the largest absolute gain. Layer 12 mean
pooling has not yet been reported.

search is based on each item in the lexicon (100 items per word type, requested 10 nearest neighbours) \
per word score is average precision at 10 (average over 100 items) \
bar plot shows average precision over all word types and 95 % CI

<img width="519" height="379" alt="Screenshot 2026-05-05 at 10 12 10" src="https://github.com/user-attachments/assets/025b5358-3c1e-4337-a0e7-4b839ef8b80b" />
