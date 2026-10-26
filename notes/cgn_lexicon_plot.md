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
an error. This can change old scores that dropped a neighbour or used a
denominator of nine. Both saved-result analysis and new retrieval assume
the correct query-to-metadata correspondence. This comparison uses the available
store entries; reproducing the original token set requires checking source
keys against the original saved index.

search is based on each item in the lexicon (100 items per word type, requested 10 nearest neighbours) \
per word score is average precision at 10 (average over 100 items) \
bar plot shows average precision over all word types and 95 % CI

<img width="519" height="379" alt="Screenshot 2026-05-05 at 10 12 10" src="https://github.com/user-attachments/assets/025b5358-3c1e-4337-a0e7-4b839ef8b80b" />
