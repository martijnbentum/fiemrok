# Compact neighbour results

`neighbour_computation.py` computes neighbours and same-type ranks and writes
them to HDF5 through `write_neighbours` and `write_frame_offsets`.
`neighbour_results.py` provides the reader classes: `Result`, `FrameOffsetQueries`,
`Query`, `Match`, `Token`, and `Type`. The evaluation entry points below call
the computation module.

Export one dataset/model/layer/pooling condition after building its vector
index with the validation pipeline:

```python
from lexicon_validation import lv_stores
from lexicon_validation.retrieval_evaluation import export_neighbours
from lexicon_validation.neighbour_results import Result

paths = export_neighbours('hubert', 3, 'center', condition='word')

with Result(paths['word']) as results:
    results.model, results.condition, results.layer, results.pooling
    results.attributes  # general file metadata
    query = results[42]
    assert query is results[42]  # same cached Query
    query.token.label
    query.token.type.label
    query.token.type.tokens  # stable token order, independent of the query

    neighbour = query.neighbours[0]
    neighbour.token.label
    neighbour.similarity
    neighbour.rank  # global, zero-based, including self in the population

    query.ranked_same_type_matches  # all same-type matches, including self, in global rank order
    query.precision(10)

    # Annotation stores open only when annotations are requested.
    results.phraser_store  # same local store attached to Echoframe
    results.items  # all Phraser words or markers in row order
    results.features['duration_ms']  # per-token arrays, row i = token i
    results.vectors[[3, 42]]  # pooled, L2-normalized SVS vectors
    query.token.echoframe_metadata
    query.token.phraser_metadata  # results.items[query.token_index]
    neighbour.token.phraser_metadata
    query.token.context(n_phone_context=10, n_word_context=3)

    batch = results.read(0, 64)  # Four numeric arrays, no annotation lookups.

# After all results and other users have finished accessing shared stores:
lv_stores.close_stores()
```

The interaction API uses five ordinary classes, without dataclasses:

| Class | Responsibility |
| --- | --- |
| `Result` | Owns one file, general metadata, token/type lookups, and query access. |
| `Query` | References its result and query token; exposes `neighbours` and `ranked_same_type_matches`. |
| `Token` | Stable identity, parent type, source keys, and lazy annotation access. |
| `Type` | Label and tokens in vector-index order. |
| `Match` | Query-relative token, similarity, and global rank. |

`query.neighbours` and `query.ranked_same_type_matches` are tuples of `Match` objects, ordered
by ascending global rank. Neighbours exclude self; same-type matches include
self. The reader derives neighbour ranks from their saved positions and the
saved self rank, so ties do not require self to rank first. The position in
either tuple is not necessarily the match's global rank.

Create query, token, type, and match views through `Result`; their constructors
are internal. Use `results.token(i)` for a token and `results.type(label)`
for a type;
`results.types` returns all types in first-token order. Token and type objects
are created on demand and shared within a result. Overlapping neighbour and
same-type entries within a query share the same match object. Token identity
is scoped to one result, even when another file describes the same selection.
Keep these views inside the result's context manager; accessing their data
after closing it raises `ValueError`.

Queries are also created lazily and cached within a result. Use `results[i]`
for indexed access, `for query in results` for iteration, and `len(results)`
for the token count. Negative indices and slices are unsupported. Repeated
query access reuses its cached matches.

Use `results.collect_queries()` to load every query and its `ranked_same_type_matches` and
`neighbours` matches with a progress bar. It returns nothing; access the cached
queries through `results[i]` or by iterating over `results`. This opens no annotation stores. Repeating
it reuses existing objects.
The result kind and frame offsets are cached when the file is opened.
For frame-offset results, `results.offsets` returns the same read-only NumPy
array on every access; query and offset lookups do not reread it from HDF5.
At 150,000 tokens, 31 tokens per type, and 30 neighbours, fully collecting
one result retains 4.65–9.15 million matches, depending on overlap between
neighbours and same-type matches. Python objects require substantial extra
memory beyond the numeric arrays. Closing the result clears its query,
match, token, and type caches; externally retained views still reject data
access after close.

Object reprs show a compact description without disk reads, annotation
lookups, or cache population. A query shows `precision@10` only after its
neighbours have been cached and at least ten are available. Closed objects
show their closed status instead of raising an exception.

`query.precision(n=10)` uses the cached neighbours, loading and caching matches
on its first call if needed. Frame-offset query status is loaded lazily once,
while the saved neighbour count is cached during result validation. Subsequent
calls perform no HDF5 reads or token creation, including for invalid queries.

For numerical work, `query.neighbour_indices`, `neighbour_similarities`,
`same_type_indices`, `same_type_ranks`, and `same_type_similarities` remain
available. These arrays and `results.read()` retain the stored order:
same-type columns use token order, while `query.ranked_same_type_matches` uses rank order. Sorting
the object view does not modify the file.

`model_name` accepts `hubert`, `w2v`, or `wav2vec2`; on disk the model names
are `hubert` and `w2v`. Layers are 3, 6, 9, 12; pooling is `center` or `mean`;
conditions are `word` and `cross_boundary`. The default condition is `word`.
Index names must end in `_center` or `_mean`; the writer rejects other names
before creating result or mapping files.
Use `workspace=...` for a different local validation workspace and
`directory=...` for a different vector-index directory. Existing results are
refused unless `overwrite=True` is passed. Existing shared mappings must
match even when overwriting a result file.

## Files and identity

Files live in the external validation workspace:

```text
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

There are 32 result files and two shared mapping files for the full
experiment. The selection PKLs remain the dataset definitions. Each results
file stores four arrays:

| Array | Default shape | dtype |
| --- | --- | --- |
| neighbour_indices | (n_tokens, 30) | uint32 |
| neighbour_similarities | (n_tokens, 30) | float32 |
| same_type_ranks | (n_tokens, 31) | uint32 |
| same_type_similarities | (n_tokens, 31) | float32 |

Every token index refers to a row in the searched vector index. Query index
is implicit: row `i` contains query `i`. The writer checks index order against
the saved Phraser selection, including uniqueness. The NPZ file stores each
type's ordered token indices once, plus a token-to-type lookup. Its mapping
is reused across conditions only when ordered Phraser keys and labels match.

The reader verifies ordered Echoframe keys, labels, and the specific index
name, and verifies the shared mapping. Fingerprints do not detect changed
vector values under unchanged keys. Rebuild results after rebuilding vectors.
Relative dependency paths and identity fields are stored inside the result
file; no separate manifests or annotation exports are created.

`results.items` loads every selected Phraser word or marker with one
`load_many` call on the saved selection PKL, checked against the result's
selection fingerprint. `token_features.load_selected_items` caches the
items per selection file, so all 16 results of a dataset share them;
`lv_stores.close_stores()` clears that cache. `phraser_metadata` reads from
these items; `echoframe_metadata` still loads per token.

`results.features` returns the per-token feature table from
`token_features.load_token_features`: a dict of NumPy arrays (columns in
`token_features.COLUMNS`) saved as `token_features/{condition}.npz` in the
workspace. A missing table is built from the items and saved; a table that no longer matches the selection raises.
Cross-boundary `n_syllables` counts distinct syllables of the item's phones,
so partially covered syllables count once.
`left_words` and `right_words` hold `n_context_words` (default 3) words of
the same speaker in the recording, crossing phrase boundaries and padded
with `*`. `articulation_rate` is phones per second over up to 30 s of the
speaker's own phones on each side (`RATE_WINDOW_MS`), excluding the token;
pauses and other speakers do not count. `rate_speech_s` is the speech used,
below 60 s near recording edges; the rate is NaN without any. Loading each
speaker's word and phone stream dominates the build, about 15 minutes per
dataset; a table saved with another context size raises.

Corpus frequencies come from `frequencies.py`. `build_frequencies()` counts
every word, syllable and phone in the local store with count-only Phraser
Lexicons (no eligibility filters; reports kept) and every cross-boundary
phoneme sequence as the sampler enumerates them, saving
`frequencies/{word,syllable,phone,cross_boundary_sequence}_counts.pkl`.
`frequency_columns(features, load_frequencies())` returns log10(count + 1)
arrays in row order: `word_log_freq` (word rows), `left_word_log_freq`,
`right_word_log_freq` and `sequence_log_freq` (cross-boundary rows), plus
`syllable_log_freq_mean`, `syllable_log_freq_min` and `phone_log_freq_mean`
over the token's `syllables` and `phones`. They are computed on demand and
not stored in the feature table.

`results.vectors` returns the pooled vectors the results were computed from,
in row order. SVS normalizes vectors when building an index, so the saved
`.npy` holds unit-length rows; unnormalized vectors are not kept. Without a
supplied `index=`, the file is memory-mapped and no FAISS index is built.

Reading numeric results loads SVS metadata JSON and the shared mapping,
but does not load vector arrays or open annotation stores. When metadata is
requested, the reader opens the existing local Echoframe/Phraser stores via
`lv_stores`. Alternatively pass `store=...` to use a caller-owned Echoframe
handle with Phraser attached, and `index=...` to reuse an already-open SVS
index. Closing a result closes only its HDF5 file. Module-owned stores are shared
across results and remain open until `lv_stores.close_stores()` is called.
Caller-owned stores remain the caller’s responsibility. Keep query views inside the results object's context manager.

## Rank and similarity conventions

Nearest neighbours exclude only the query token. Tokens from the same
speaker or recording remain eligible. Same-type arrays include every token
of the query's type, including the query, in the order exposed by
`same_type_indices`. That order is vector-index order, not similarity order.

Same-type ranks are zero-based positions among **all** dataset tokens,
including self. Larger cosine similarity comes first; exact ties are broken
by ascending token index. Self is not forced to rank zero when vectors tie.
These ranks use a population including self, whereas nearest-neighbour
lists exclude self.

The writer uses SVS's query normalization and cosine inner products: float32
for a FAISS index, float64 for its brute-force backend, with stored
similarities always float32. Ranks are calculated before output conversion.
BLAS and FAISS may differ in the last floating-point bits, so nearly tied
tokens can differ from an independent FAISS search. The saved nearest
neighbours and same-type ranks use the same scores and deterministic tie rule.

## Computation and storage

Exact same-type ranks require all-candidate comparisons. Similarities are
computed in query batches (default 64), then sorted one query at a time;
the full pairwise matrix is never retained. Temporary score memory is
approximately `batch_size * n_tokens * 4` bytes for FAISS conditions, about
38.4 MB at the defaults and 150,000 tokens. Loaded vectors, SVS's backend,
and sort buffers use additional memory. Decrease `batch_size` to reduce score
memory. This is an exhaustive computation, not approximate ranking.

At 150,000 tokens in every condition, the four uncompressed arrays total
73.2 MB per condition or 2.3424 GB across all 32, with 31 tokens per type.
Each shared mapping adds roughly 1.2 MB of uncompressed numeric indices
(membership lists plus the token-to-type lookup). HDF5 chunks use lossless
LZF compression by default;
`compression=None` disables it. Temporary outputs are published only after
successful completion.

A synthetic local check with 3,000 tokens, 768 dimensions, and 100 tokens
per type produced 3.15 MB without compression, 2.16 MB with LZF, and 2.00 MB
with gzip, taking approximately 0.5 seconds per export. This small benchmark
does not predict full-corpus timing or compression. Ten random query lists
matched FAISS neighbour indices, with score differences below 5e-7.

See the [Cheerilee benchmark](neighbour_results_performance.md) for measurements
against a real 157,200-token index and recommended BLAS/thread settings.

`neighbour_metadata.py` remains available, including context/boundary helpers
used by the new reader. No old JSONL conversion or legacy-index compatibility
is provided. Existing evaluation functions remain available; the result
format does not introduce a new aggregate evaluation metric.
