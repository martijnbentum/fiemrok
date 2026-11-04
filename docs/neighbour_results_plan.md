# Compact lexicon-validation neighbour results

## Scope and agreed behaviour

Replace repeated JSONL neighbour records with numeric retrieval arrays whose
token indices refer directly to rows in the searched SVS vector index. Keep
`neighbour_metadata.py` available; obtain annotations through Echoframe and its
attached local Phraser store instead of exporting annotation dictionaries.

This concerns `lexicon_validation` only. Do not change sample selection,
`fiemrok_experiment`, or existing vector-index storage as part of this feature.
Result files depend on the original vector indexes and annotation stores.
They are not standalone annotation exports.

Agreed contents:

- First 30 non-self neighbours: token indices and float32 similarities.
- All same-type tokens, including the query: global ranks and float32
  similarities, aligned with a shared type-to-token-index mapping.
- uint32 for global token indices and ranks; float32 for similarities.
- Query index is implicit: result row `i` refers to vector-index row `i`.
- Both datasets, both models, layers 3/6/9/12, center/mean pooling.
- Keep same-speaker and same-recording neighbours eligible.
- No separate config, manifest, summary, or annotation-export files.

The default same-type width is 100 because selection defaults to 100 tokens
per type. Derive the width from the actual uniform sample quota; do not
hard-code 100 or relax the quota. Default nearest-neighbour count is 30 and
is independent of the per-type sample quota.

## Feature 1: identity and shared type mapping

Requirements:

- Use SVS metadata order as the retrieval coordinate system. Require one
  Echoframe source key per vector, as produced by `vector_indices.py`.
- Resolve a token through vector metadata -> Echoframe key -> Echoframe
  metadata -> attached local Phraser annotations.
- Compare ordered underlying Phraser token keys with the saved selection.
  Reject missing, duplicate, or reordered token identities.
- Build each type's token-index list from vector metadata labels, preserving
  vector-index order. Store these lists once per dataset, shared across
  model/layer/pooling conditions.
- Confirm identical ordered Phraser keys and type grouping before reusing a
  mapping. Echoframe keys themselves differ across embedding conditions.
- Store a selection/order fingerprint in the mapping and each result file.
  Bind each result to its specific vector index using its name and a
  fingerprint of ordered Echoframe source keys. Keep these as internal file
  fields, not a separate manifest.
- These checks identify token/order mismatches, not changed vector values
  under unchanged keys. Vector-content checksums are outside initial scope.

Tests:

- Index-to-key-to-annotation resolution selects the correct token.
- Mapping reuse succeeds across conditions with different Echoframe keys
  but identical Phraser order; reordered tokens or changed labels fail.
- Same-type indices follow vector order and include the query exactly once.
- Wrong result/index pairing fails before returning metadata.

## Feature 2: numeric storage and paths

Requirements:

Use the existing external workspace and selection filenames:

```text
lexicon_validation/
    word_keys.pkl
    cross_boundary_marker_keys.pkl
    neighbours/
        word/
            token_mapping.npz
            hubert/
                layer03_center.h5
                layer03_mean.h5
                ... layers 06, 09, 12
            w2v/
                ... same layer/pooling files
        cross_boundary/
            ... same structure
```

Use existing model identifiers `hubert` and `w2v`. Add path helpers to
`lv_locations.py`; preserve existing SVS index locations and names.

Each HDF5 results file contains:

| Dataset | Shape | dtype |
| --- | --- | --- |
| neighbour_indices | (n_tokens, n_neighbours) | uint32 |
| neighbour_similarities | (n_tokens, n_neighbours) | float32 |
| same_type_ranks | (n_tokens, n_tokens_per_type) | uint32 |
| same_type_similarities | (n_tokens, n_tokens_per_type) | float32 |

The NPZ mapping contains numeric type membership arrays and the identity
information needed to interpret them. Support lookup from query token index
to its type's ordered token indices without reading annotations per query.

- Chunk HDF5 arrays for query/batch reads and writes. Avoid loading the full
  results file for a single query. Choose lossless compression after a small
  representative measurement; estimates below assume no compression.
- Write temporary outputs and publish only after successful completion.
  Refuse existing output by default; allow explicit overwrite.
- At 150,000 tokens in every condition, default arrays occupy 156 MB per
  condition, or 4.992 GB across 32 conditions. Shared mapping token-index
  arrays add about 0.6 MB per dataset, plus small lookup/identity fields.

Tests:

- Round-trip shapes, dtypes, values, and identity fields.
- Query/batch reads, context-manager closure, and failed-write publication.
- Existing-output refusal and explicit overwrite.

## Feature 3: exact retrieval and same-type ranks

Requirements:

- Retrieve 30 nearest non-self tokens and their similarities by default.
- Compute each same-type token's rank among all dataset tokens, including
  the query in the ranked population. This population differs deliberately
  from the non-self nearest-neighbour list.
- Proposed rank convention: zero-based, larger similarity first, ties
  broken by ascending token index. Self participates normally; do not force
  self to rank zero when identical vectors tie.
- Calculate ranks and ordering before writing float32 similarities. Never
  derive ranks from rounded output scores.
- Inspect the installed SVS search implementation first: preserve its actual
  vector normalization and scoring semantics. Define the metric in code,
  with one interpretation for this module; do not add a metric-description
  export or invent a different similarity calculation.
- Calculate exact ranks with bounded query batches, sorting one query at a
  time rather than retaining the complete all-pairs matrix. This avoids 100
  full score comparisons per query when extracting same-type ranks.
- Include progress, elapsed timings, and query counts. Exact ranks still
  require substantial all-candidate comparisons; benchmark before launching
  all conditions. Do not replace exact ranks with approximate ranks silently.

Tests:

- Compare neighbour lists, same-type ranks, and similarities against a small
  exhaustive reference, including ties, identical vectors, and self.
- Batched/block computation agrees with the reference and across block sizes.
- Same-type columns align with the shared mapping for every query.
- Same speaker/recording tokens remain eligible.

## Feature 4: result classes and metadata access

Add `lexicon_validation/neighbour_results.py`. Review the public API with the
user before implementing the classes; names below are provisional.

- `NeighbourResults`: opens arrays and the shared mapping, checks vector-index
  identity, supports context management and query lookup.
- `QueryResult`: a lightweight view for one query, exposing nearest-neighbour
  indices/similarities and same-type indices/ranks/similarities.
- `results[i]` consistently means query at vector-index row `i`.
- Resolve labels and annotations on demand through existing vector metadata
  and `lv_stores` handles. Do not silently open the source CGN store.
- Make caller ownership of passed stores explicit. Close handles opened by
  the results object, while leaving caller-owned handles open.
- Reuse existing context/boundary logic where needed; keep
  `neighbour_metadata.py` intact for now. Avoid building a second metadata
  cache/export system.

Tests:

- Query views expose correctly aligned arrays and metadata.
- Metadata lookup is lazy and uses the provided/local stores.
- Owned handles close and caller-owned handles remain usable.

## Feature 5: export integration

- Add a compact-results writer in the new module and connect it to
  `retrieval_evaluation.export_neighbours`, retaining `condition='word'`.
- Resolve the current pipeline's actual SVS index names rather than the
  legacy hard-coded cross-boundary type count in the existing exporter.
- Keep numerical evaluation possible without annotation-store access.
- Document the new return value and output format explicitly. No backwards
  compatibility layer or automatic migration of old JSONL files is needed.
- Do not change evaluation metrics or remove `neighbour_metadata.py` in this
  feature. Existing precision helpers may remain available.

Tests:

- One export call handles only its requested dataset/model/layer/pooling.
- Export uses the correct current index and local paths.
- End-to-end small-fixture export/read/annotation resolution and metric parity.

## Review sequence and remaining decisions

1. Review identity/storage requirements, including the proposed rank/tie
   convention, before implementing the writer.
2. Review the class API together before implementing public classes.
3. Inspect SVS scoring and benchmark exact-rank computation on a subset.
4. Implement and verify the agreed features with targeted tests.

## Implementation

The format and initial classes are implemented; see
[usage and conventions](neighbour_results.md). Query batches and per-query
stable sorting replace the initial proposal to count ranks in candidate
blocks. Numerical arrays are written in batches with LZF compression after a
small synthetic compression/timing comparison. Full-corpus timing remains
unmeasured. No production data export or remote work is part of this change.
