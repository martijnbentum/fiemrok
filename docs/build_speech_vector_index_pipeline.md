# Build lexicon-validation speech-vector indexes

This pipeline belongs to `lexicon_validation`, independently of
`fiemrok_experiment`. Run from `repo/` with the workspace environment active.
CGN and the two model checkpoints must be accessible for the full rebuild.

```python
from lexicon_validation import build_speech_vector_index_pipeline as pipeline

pipeline.prepare_samples(n_tokens=100, min_types=2000, random_seed=0)
pipeline.build_embeddings(gpu=True, batch_size=32)
pipeline.build_indexes()
```

The three stages are separate so a completed selection can be used in later
sessions without sampling again. `prepare_samples()` generates a replacement
selection and new cross-boundary markers: call it once for a new dataset.
It saves ordered Phraser keys, which both models and every layer/pooling
condition then use. To choose another validation workspace, pass the same
`workspace=...` to all three calls.

## Selection

Both datasets use Netherlandic Dutch, excluding CGN components `a,c,d,h,m`.
Duration must be within 100-1500 ms, inclusive. Per-type duration fences are
Q1 - 5 * IQR and Q3 + 5 * IQR, calculated after absolute duration filtering.
Frequency eligibility and sampling happen afterward. Each accepted type
supplies exactly `n_tokens` occurrences; the default is 100.

Words may have any syllable count but must contain at least two phones.
Their type labels are orthographic. Cross-boundary types are phoneme
sequences and follow the selected word types' phone-count distribution,
using the modal pronunciation of each word type. Cross-boundary selection
can contain fewer types, with `min_types=2000` by default. A failed bounded
sampling search raises an error; it does not silently reduce that floor or
the per-type token quota. Selected cross-boundary items cannot share source
phone occurrences.

The defaults retain existing quota tolerance and search settings. Override
`tolerance`, `max_attempts`, or `tries_per_round` on `prepare_samples()` when
needed. Shared source annotations remain in the CGN Phraser store; selected
cross-boundary markers are saved there too.

## Files

`lexicon_validation/lv_locations.py` anchors validation paths to the external
workspace next to `repo/`:

```text
fiemrok/lexicon_validation/
  word_keys.pkl
  cross_boundary_marker_keys.pkl
  echoframe_stores/
    w2v_word_{n_types}/
    hubert_word_{n_types}/
    w2v_cross_boundary_{n_types}/
    hubert_cross_boundary_{n_types}/
  svs_data/
    {model}_{condition}_{n_types}_{layer}_{pooling}.npy
    {model}_{condition}_{n_types}_{layer}_{pooling}.json
```

`n_types` is the actual selected count, not the minimum floor. The JSON/NPY
pair is the existing speech-vector-search index format. No configuration,
manifest, summary, or additional annotation export is generated.

## Embeddings and indexes

Both models use version 1/checkpoint 200000, layers 3, 6, 9, 12, and a 500 ms
context collar on each side. Echoframe stores full token frame matrices;
indexes apply center or mean pooling. This produces 32 index conditions.
`build_embeddings()` uses the existing batch extractor, which can skip
already extracted token/model/layer/collar combinations after interruption.
`build_indexes()` builds fresh indexes; use `overwrite=True` explicitly to
replace existing index files. It reports missing requested embeddings
rather than dropping selected tokens.

For individual operations use `word_lexicon`, `cross_boundary_lexicon`,
`make_embeddings`, and `build_speech_vector_index`. The old module names
`make_cross_boundary_embeddings` and `validate_lexicon` have been replaced.

Retrieval output contents, neighbour counts, and metrics remain deferred.
Existing retrieval examples document the previous dataset and naming scheme.
This pipeline neither deletes old remote files nor launches remote jobs.

## Tests

Run the local suite with numerical libraries limited to one thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../fiemrok_env/bin/python -B -m unittest discover -s tests
```

Integration tests build real temporary Phraser/Echoframe stores and all 32
vector-index conditions with synthetic model outputs. They do not load the
remote checkpoints or process the real CGN corpus.
