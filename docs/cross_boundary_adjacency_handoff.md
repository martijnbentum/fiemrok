# Handoff: word competition for cross-boundary frame offsets

Status: design agreed 2026-11-20; not implemented yet.

## Question
For cross-boundary (cb) queries, at which positions does word identity
(word 1 = A, word 2 = B, prev, next) compete with or beat the marker's own
cb type in nearest-neighbour retrieval, if ever?

## Runs
- New result kind: cb queries against the word index. Stored separately as
  `frame_offsets/cross_boundary_vs_word/{model}/layer{LL}_{pooling}.h5`
  with a new `lv_locations` path function; records both selection
  fingerprints and the index name.
- Queries are identical to the cb-vs-cb runs: same selection (all 167,121
  markers, saved order), offsets (±100 frames, 20 ms), 2000 ms collar.
  center: single frames; mean: the marker's fully contained frames, shifted
  together by the offset (current `select_query_frames`; files with a
  `query_pooling` attribute). No need to wait for the running cb reruns,
  but merge only with the new cb mean files, not the 19 Nov ones.
- Exclusion: per query, exclude the word-index tokens that are the marker's
  A, B, prev or next occurrence (by Phraser key).
- Ranks: store ranks/similarities of the other A-type and B-type tokens in
  the word index (30 or 31 each, NA padded; NA when not in the lexicon),
  counted after the exclusions. Compute cost is unchanged (one sort per
  row); files grow to roughly 17-19 GB.
- Writer change: `write_frame_offsets` assumes queries and candidates are
  the same tokens (identity check, self exclusion by row, own-type ranks).
  Extend it with separate query keys, per-query excluded rows and per-query
  rank targets.
- First conditions: HuBERT L9 center and mean, on wildfire
  (`mbentum@wildfire.science.ru.nl`), on an idle GPU.

## Analysis
- Merge per query frame the stored top-30 of the cb-vs-cb and cb-vs-word
  runs by cosine similarity; take the top 10 (exact for k <= 30). Curves are
  shares of these 10 slots: own, word 1, word 2, prev, next.
- Adjacency: prev = word before A, next = word after B in the speaker
  stream, gap < 100 ms; store labels, gaps and word start/end times.
- Populations (option 2): require A and B in the word lexicon; prev/next
  curves only where valid. Show n per curve in the legend. Exclude markers
  with label collisions (A = B, prev = A, etc.) from the affected curves and
  report counts.
- x-axis: four equal sections prev | word 1 | word 2 | next, each word
  stretched to unit width, no time labels. Drop gap frames between words
  (report if this matters). Show the marker span (median, IQR of edges).
  Measure coverage of query frames per position; clip below 50%.
- Statistics: per token d(x) = own - word1 and own - word2; type-weighted
  (marker type) bootstrap with a simultaneous band; report regions where
  it excludes 0. Robustness: also weight by word-1 / word-2 type.
- Secondary panel: type-weighted median log10 rank of the nearest A/B token
  (word index) and of the nearest own token (cb index), with chance level
  ~N/31 (about 5,700). Ranks come from different pools: approximate.

## Check plots
1. Duration terciles: rows word 1 / word 2, columns short / mid / long.
2. Pool share: fraction of merged top-10 that are markers, center vs mean
   (mean queries are marker-shaped and may favour the cb pool).
3. Coverage per section position.
4. All-four subset (option 1) overlaid on option 2.
5. Table of exclusions per reason (lexicon, gap, collision).

## Numbers (2026-11-20, label and gap check)
- 167,121 markers, 5391 cb types, 5675 word types.
- In lexicon: A 91.2%, B 89.4%, prev 91.7%, next 90.8%.
- Gap < 100 ms: prev 89.9%, next 79.7%.
- A and B valid: 137,096. All four valid: 86,214 (51.6%).
- Collisions among A&B-valid markers: 0.3-1.3% per pair, e.g. prev = next
  1,763 (mostly "van").
- One-off script and output: `/tmp/fiemrok-cb-gap-20261120/` on mlp06.

## Pointers
- `frame_offset_curves.py` (`build_adjacency`, `type_weighted_mean`),
  `frame_offset_evaluation.py` (`evaluate_words`, `select_query_frames`),
  `neighbour_computation.py` (`write_frame_offsets`),
  `token_features.py` (`_target_phones_and_words`, stream helpers),
  speech-vector-search `gpu_search.global_ranks_rows`.
