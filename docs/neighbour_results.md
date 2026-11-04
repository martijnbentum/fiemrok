# Compact neighbour results

Export one dataset/model/layer/pooling condition after building its vector
index with the validation pipeline:

```python
from lexicon_validation.retrieval_evaluation import export_neighbours
from lexicon_validation.neighbour_results import NeighbourResults

paths = export_neighbours('hubert', 3, 'center', condition='word')

with NeighbourResults(paths['word']) as results:
    query = results[42]
    query.label
    query.neighbour_indices
    query.neighbour_similarities
    query.same_type_indices
    query.same_type_ranks
    query.same_type_similarities
    query.precision_at_k(10)

    # Annotation stores open only when annotations are requested.
    query.metadata.phraser_object
    results.metadata(int(query.neighbour_indices[0])).phraser_object
    results.context(42, n_phone_context=10, n_word_context=3)

    batch = results.read(0, 64)  # Four numeric arrays, no annotation lookups.
```

`model_name` accepts `hubert`, `w2v`, or `wav2vec2`; on disk the model names
are `hubert` and `w2v`. Layers are 3, 6, 9, 12; pooling is `center` or `mean`;
conditions are `word` and `cross_boundary`. The default condition is `word`.
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
| same_type_ranks | (n_tokens, 100) | uint32 |
| same_type_similarities | (n_tokens, 100) | float32 |

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

Reading numeric results loads SVS metadata JSON and the shared mapping,
but does not load vector arrays or open annotation stores. When metadata is
requested, the reader opens the existing local Echoframe/Phraser stores via
`lv_stores`. Alternatively pass `store=...` to use a caller-owned Echoframe
handle with Phraser attached, and `index=...` to reuse an already-open SVS
index. The results object closes stores it opens; caller-owned stores remain
open. Keep query views inside the results object's context manager.

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
156 MB per condition or 4.992 GB across all 32. Each shared mapping adds
roughly 1.2 MB of uncompressed numeric indices (membership lists plus the
token-to-type lookup). HDF5 chunks use lossless LZF compression by default;
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
