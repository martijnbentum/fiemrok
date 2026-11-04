# Lexicon validation pipeline changes

Implemented plan; see [usage](build_speech_vector_index_pipeline.md). All generated validation data will be rebuilt from scratch.
This assumption does not authorize deletion. fiemrok_experiment is separate
and remains unchanged.

## Module organization

- Add word_lexicon.py for validation word selection.
- Keep cross_boundary_lexicon.py and cross_boundary_lexicon_sampling.py
  separate from word selection.
- Rename make_cross_boundary_embeddings.py to make_embeddings.py; extend
  its existing functions to support words and markers.
- Renamed validate_lexicon.py to build_speech_vector_index.py.
- Add build_speech_vector_index_pipeline.py as the entry point.
- Add package-local lv_locations.py for validation-specific paths;
  use explicit relative imports. Keep shared source locations in the root
  module and do not alter fiemrok_experiment paths.
- Reuse lexicon_helper filters with explicit validation parameters; leave
  shared defaults unchanged.

## Word selection

Add filter_word_tokens(), select_words(), save_selected_word_keys(), and
load_selected_word_keys() in word_lexicon.py.

Apply nl-only and acdhm exclusions; require at least two phones; allow any
syllable count; apply inclusive 100-1500 ms bounds; group by orthographic
label; apply per-type 5 * IQR duration filtering; check frequency; sample
exactly n_tokens=100 per type, configurable. Save ordered Phraser keys.
Use stable candidate ordering and a local random generator. No additional
annotated-overlap filter is introduced.

Reuse lexicon_helper.filter_outlier_tokens_from_type with an explicit
multiplier of 5. Tests cover filters, boundary values, multisyllabic words,
per-type outlier removal before frequency checks, configurable quotas,
reproducibility, and ordered key round trips.

## Cross-boundary selection

Change CrossBoundaryLexicon.__init__() and collect_candidate_tokens() to
apply per-type iqr_multiplier=5 after absolute duration bounds and before
frequency eligibility. Keep nl/acdhm filters and no shared source phones.

Change inspect_lexicon_phonemes() to accept newly selected word objects,
not an old layer-6 store. Retain modal pronunciation per word type with
deterministic tie-breaking.

Expose n_tokens=100 through plan_sampling(). Set min_types=2000 on both
CrossBoundaryLexicon.sample_lexicon() and sampling.sample_lexicon(); pass it
through the pipeline. Retain adaptive quotas: fewer cross-boundary types
are acceptable, subject to the floor. Report failure if sampling cannot
satisfy the floor and token quota; do not relax either silently.

Create markers once through make_markers(), save ordered keys, and load
those keys in later stages. Tests cover outlier filtering, word-derived
length targets, configurable quotas/floor, smaller feasible totals,
no shared phones, and explicit sampling failure.

## Embeddings and vector indexes

Extend compute_embeddings(), create_store(), and the model convenience
functions in make_embeddings.py to accept either saved selection. Generate
four fresh Echoframe stores: word/cross-boundary times Wav2Vec 2.0/HuBERT,
with layers 3, 6, 9, 12.

Do not add validate_embedding_coverage() or old-cache compatibility logic.
Use the existing batch extractor. Downstream construction must report
missing requested embeddings rather than silently dropping selected items.

In the proposed build_speech_vector_index.py, change load_metadata_layer(),
make_vector_search_index_for_layer(), and make_vsi_for_layers() to process
the requested model/layer and saved selection order. Build every index
fresh. Use the correct word/marker unit_type in make_vector_metadata().
Retain existing center/mean pooling arithmetic. Tests cover selected-item
inputs, order, pooling, unit types, and missing requested embeddings.

## Pipeline and placement

Add prepare_samples(), build_embeddings(), and build_indexes() in
build_speech_vector_index_pipeline.py. prepare_samples() exposes both
n_tokens=100 and min_types=2000.

Put the two selection PKLs directly in the external
fiemrok/lexicon_validation workspace. Put the four Echoframe stores under
its echoframe_stores/ directory. Name each store
{model}_{condition}_{n_types}, with word/cross_boundary as the conditions
and the actual selected type count, not the sampling floor. Derive the
store path after selection. Both models use the same count per condition.
Echoframe stores keep their own internal layout. Keep the validation path
module named lv_locations.py and use explicit relative imports within the package.
No run directories, config files, manifests, or summary files. Vector-index
output placement and naming remain open.

## Deferred

Revisit analyse_results.py, neighbour_metadata.py, and analysis/plotting
scripts after agreeing retrieval storage and metrics. Prefer annotation
lookups through Echoframe keys. Exclude only the query token itself.
Thirty saved neighbours and precision@10 remain undecided.

## Implementation order

1. Establish validation-local paths and module names.
2. Select words with per-type 5 * IQR filtering and save ordered keys.
3. Derive phone-count targets, sample cross-boundary items using the new
   floor, create markers, and save ordered marker keys.
4. Generate the four fresh Echoframe stores from those selections.
5. Build all 32 dataset/model/layer/pooling index conditions from scratch.
6. Connect the pipeline functions and update affected imports/tests/examples.
7. Revisit retrieval output contents and metrics before implementing them.
