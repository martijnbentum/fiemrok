# Lexicon validation pipeline changes

Implemented plan; see [usage](lexicon_validation_pipeline.md). All generated validation data will be rebuilt from scratch.
This assumption does not authorize deletion. fiemrok_experiment is separate
and remains unchanged.

## Module organization

- Add word_lexicon.py for validation word selection.
- Keep cross_boundary_lexicon.py and cross_boundary_sampling.py
  separate from word selection.
- Rename make_cross_boundary_embeddings.py to embeddings.py; extend
  its existing functions to support words and markers.
- Renamed validate_lexicon.py to vector_indices.py.
- Add pipeline.py as the entry point.
- Add package-local lv_locations.py for validation-specific paths;
  use explicit relative imports. Keep shared source locations in the root
  module and do not alter fiemrok_experiment paths.
- Reuse lexicon_helper filters with explicit validation parameters; leave
  shared defaults unchanged.

## Word selection

Implemented prerequisite: build one experiment-specific Phraser material store.
Select Audio records from the current CGN source with variant nl and
components outside acdhm. Copy the selected recordings' complete phrase,
word, syllable, and phone hierarchies plus their linked speakers. Preserve
all original keys and identifiers; exclude existing markers. Deduplicate
shared speakers without following their links into unselected recordings.
Keep audio filenames unchanged. Use this material store for both datasets,
all later metadata lookups, and new cross-boundary markers. The generic
`phraser.copy_materials.copy_audio_materials(audio_keys, goal_store_root,
source_store)` operation is implemented in the sibling Phraser repository;
pipeline integration uses build_local_phraser_store(components, language,
source_store=None). The destination name is cgn_{components}_{language},
using included, sorted component letters. Any existing destination is rejected.
Subsequent stages open this local store and never implicitly use shared CGN.

Add filter_word_tokens(), select_words(), save_selected_word_keys(), and
load_selected_word_keys() in word_lexicon.py.

Population filtering happens only in the material-store stage. Require at
least two phones; allow any syllable count; apply inclusive 100-1500 ms bounds; group by orthographic
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
frequency eligibility. Remove phrase-level component/variant filters;
retain the no-shared-source-phones sampling rule.

Change inspect_lexicon_phonemes() to accept newly selected word objects,
not an old layer-6 store. Retain modal pronunciation per word type with
deterministic tie-breaking.

Expose n_tokens=100 through plan_sampling(). The pipeline uses min_types=None
to derive a floor of ceil(0.9 * word_type_count) and passes it to the sampler.
Start at the word-type goal, never request more, and reduce only after
failed rounds (or an insufficient frequency ceiling). Keep a successful
total and improve distribution fit within that total. An explicit floor
cannot exceed the word-type goal. Report failure rather than relaxing the
90% floor or token quota. Generic low-level sampling retains an explicit
floor parameter; the pipeline supplies the experiment-specific bound.

Create markers once through make_markers(), save ordered keys, and load
those keys in later stages. Tests cover outlier filtering, word-derived
length targets, configurable quotas/floor, smaller feasible totals,
no shared phones, and explicit sampling failure.

## Embeddings and vector indexes

Extend compute_embeddings(), create_store(), and the model convenience
functions in embeddings.py to accept either saved selection. Generate
four fresh Echoframe stores: word/cross-boundary times Wav2Vec 2.0/HuBERT,
with layers 3, 6, 9, 12.

Do not add validate_embedding_coverage() or old-cache compatibility logic.
Use the existing batch extractor. Downstream construction must report
missing requested embeddings rather than silently dropping selected items.

In the proposed vector_indices.py, change load_metadata_layer(),
make_vector_search_index_for_layer(), and make_vsi_for_layers() to process
the requested model/layer and saved selection order. Build every index
fresh. Use the correct word/marker unit_type in make_vector_metadata().
Retain existing center/mean pooling arithmetic. Tests cover selected-item
inputs, order, pooling, unit types, and missing requested embeddings.

## Pipeline and placement

Provide build_local_phraser_store(), prepare_samples(), build_embeddings(),
and build_indexes() in
pipeline.py. prepare_samples() exposes both
n_tokens=100 and min_types=None (derive the relative type-count floor).

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

Revisit retrieval_evaluation.py, neighbour_metadata.py, and analysis/plotting
scripts after agreeing retrieval storage and metrics. Prefer annotation
lookups through Echoframe keys. Exclude only the query token itself.
Thirty saved neighbours and precision@10 remain undecided.

## Implementation order

1. Build the new local Phraser recording store, filtering component/variant
   only here. Other stages use it for annotations and new markers.
2. Select words with per-type 5 * IQR filtering and save ordered keys.
3. Derive phone-count targets, sample cross-boundary items using the new
   floor, create markers, and save ordered marker keys.
4. Generate the four fresh Echoframe stores from those selections.
5. Build all 32 dataset/model/layer/pooling index conditions from scratch.
6. Connect the pipeline functions and update affected imports/tests/examples.
7. Revisit retrieval output contents and metrics before implementing them.
