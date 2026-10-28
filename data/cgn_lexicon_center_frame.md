
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

For a mean-pooling comparison, pass `pooling='mean'` to
`validate_lexicon.make_vsi_for_layers()` or
`make_vector_search_index_for_layer()`. This averages each token's stored
word frames and saves separate indexes such as `svs_data/w2v_6_mean.npy`
and `svs_data/w2v_6_mean.json`. The default `pooling='center'` retains the
original index names. Both query and reference vectors use the chosen method.
Frame loading now uses Echoframe's shard-grouped batch reader. See
[Resume HuBERT mean pooling](../data/hubert_mean_pooling.md) for environment setup,
saved-index reuse without metadata loading, and a resumable evaluation block.
`analyse_results.evaluate_index()` returns per-token precision grouped by
word type, suitable for `per_word()`, `overall_result()`, and
`plot_precision()`. It requests 11 neighbours and excludes self-matches by
source key before scoring ten neighbours. The existing `analyse_results()`
function also excludes self-matches by key and scores exactly ten remaining
neighbours. Saved results with ten neighbours already excluding self can
be scored directly; results with fewer than ten non-self neighbours raise
an error.
