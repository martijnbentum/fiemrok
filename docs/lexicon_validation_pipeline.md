# Lexicon validation pipeline

This pipeline belongs to `lexicon_validation`, independently of
`fiemrok_experiment`. Run from `repo/` with the workspace environment active.
CGN and the two model checkpoints must be accessible for the full rebuild.
Use Phraser 0.2.84 or newer for its recording copy API.

```python
from lexicon_validation import pipeline

materials = pipeline.build_local_phraser_store(
    components='abefghijklno', language='nl')
materials.close()
pipeline.prepare_samples(n_tokens=100, random_seed=0)
pipeline.build_embeddings(gpu=True, batch_size=32)
pipeline.build_indexes()
```

The four stages are separate so a completed selection can be used in later
sessions without sampling again. `prepare_samples()` generates a replacement
selection and new cross-boundary markers: call it once for a new dataset.
It saves ordered Phraser keys, which both models and every layer/pooling
condition then use. To choose another validation workspace, pass the same
`workspace=...` to all four calls. For a different material population, pass
the same `components` and `language` to every stage. The defaults include
components `b,e,f,g,i,j,k,l,n,o` and variant `nl`.

## Local Phraser materials

`build_local_phraser_store` selects Audio records from the shared CGN store
and copies their complete phrase/word/syllable/phone hierarchy and linked
speakers. Pass `source_store=...` to choose an already open source Store.
Keys, persisted metadata, and audio paths remain unchanged; old markers
are excluded. The returned Store is open and belongs to the caller.

The destination is `cgn_{components}_{language}` directly under the validation
workspace. Components are included letters and are sorted/deduplicated for
the name. `language` is the CGN variant directory (`nl` or `vl`), not the
Audio object's language description. An existing destination raises
`FileExistsError`, including an empty directory. No matching recordings
raises `ValueError` before a destination is created.

All later stages default to this local store and close their own handles.
They never fall back to the shared source when local materials are missing.
An explicit `cgn=...` supplies a caller-owned Phraser Store instead.

## Selection

With the defaults, the local material store contains Netherlandic Dutch
outside CGN components `c,d,m`. Component and variant filtering occurs
only when building that store, and is not repeated during sampling.
Duration must be within 100-1500 ms, inclusive. Words are selected with
Phraser's `build_word_lexicon`: they need at least two phones, the store's
`within_range` duration flag, no recorded timing concern, and no overlap
with another speaker's speech. The duration
flag holds per-type fences assessed over the whole source store, not only
the local material. Cross-boundary sequences have no such flag and keep
per-type fences Q1 - 5 * IQR and Q3 + 5 * IQR, calculated after absolute
duration filtering. Frequency eligibility and sampling happen afterward.
Each accepted type supplies exactly `n_tokens` occurrences; the default is
100.

`word_lexicon.select_words(store)` returns a Phraser `Lexicon`. By default
it reports recording-loading time and scan progress at recording boundaries,
roughly every five seconds, including elapsed time and estimated time remaining.
The final filter table shows tokens and types retained and lost, plus types
newly below the token quota. Eligible counts precede the per-type sampling cap;
the sampled total is reported separately. Use `verbose=False` for a quiet call.
The inventory retains all eligible keys and has no type-frequency minimum
by default (`n_tokens=None`). Pass `n_tokens` explicitly to filter rare types
during building. The pipeline supplies its quota explicitly, then calls
`.sample(n_tokens=n_tokens, random_seed=random_seed)`
to freeze its exact per-type selection. Sampling returns a new lexicon and
drops types with too few retained keys; `.token_keys` gives its ordered keys.

Words may have any syllable count.
Their type labels are orthographic. Cross-boundary types are phoneme
sequences and follow the selected word types' phone-count distribution,
using the modal pronunciation of each word type. Cross-boundary selection
starts with the same type-count goal as the words. Only failed sampling
rounds permit reducing the count, down to 90% of the word types (rounded up).
Once a total succeeds, subsequent attempts keep that total and try to improve
the phone-count distribution. The sampler never requests more cross-boundary
types than word types. A known frequency ceiling can reduce the initial plan
before attempts start.

For 1,655 word types, the allowed range is 1,490-1,655 cross-boundary types.
`min_types=None` derives that lower bound automatically. An explicit
`min_types` can make the floor stricter; one exceeding the word-type count
raises an error. The former 2,000-type floor is not a pipeline default.
Each type still requires exactly `n_tokens` non-overlapping occurrences.
Selected cross-boundary items cannot share source phone occurrences.

There is no word-type cap: all types meeting the eligibility rules are kept.
Progress output announces word/phrase loading, reports elapsed times, and
shows starting and retained type/token counts at each filtering step. The
word report is Phraser's lexicon report: token and type losses per stage,
including types newly falling below the token quota. Unassessed and
out-of-range duration flags are separate stages. Cross-boundary IQR removal
and frequency rejection are reported separately, followed by sampling
counts and the cross-boundary distribution diagnostics.

The defaults retain existing quota tolerance and search settings. Override
`tolerance`, `max_attempts`, or `tries_per_round` on `prepare_samples()` when
needed. Both datasets and new cross-boundary markers use the local Phraser
material store. Sampling does not modify the shared source annotations.

## Files

`lexicon_validation/lv_locations.py` anchors validation paths to the external
workspace next to `repo/`:

```text
fiemrok/lexicon_validation/
  cgn_abefghijklno_nl/             # local Phraser materials, default population
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

## Open stores interactively

Use `lexicon_validation/lv_stores.py` to access this experiment's stores:

```python
from lexicon_validation import lv_stores

materials = lv_stores.load_phraser_store()
embeddings = lv_stores.load_echoframe_store('w2v', 'word',
    materials=materials)

# Metadata resolves through the attached local materials store.
# embeddings.metadatas[0].phraser_object

embeddings.close()
materials.close()
```

`model` is `w2v` or `hubert`; `condition` is `word` or `cross_boundary`.
The loader finds the corresponding named Echoframe directory automatically;
the type count stays in the directory name but is not a loading parameter.
There is one store per model/condition, four in total. Multiple matches
raise an error rather than choosing one arbitrarily.
The loaders open existing stores and leave closing them to the caller.
Both loaders accept `components`, `language`, and `workspace`; the Phraser
loader also accepts an explicit `path`.

When `materials` is omitted from the Echoframe loader, it opens and attaches
the named local Phraser store automatically. In that case, close both with
`embeddings.close_phraser_stores()` followed by `embeddings.close()`.
When you supply `materials`, close that caller-owned store separately after
all embedding stores using it are finished. There is no global store cache.

## Embeddings and indexes

Both models use version 1/checkpoint 200000, layers 3, 6, 9, 12, and a 500 ms
context collar on each side. Echoframe stores full token frame matrices;
indexes apply center or mean pooling. This produces 32 index conditions.
`build_embeddings()` uses the existing batch extractor, which can skip
already extracted token/model/layer/collar combinations after interruption.
To create or resume just one store from its saved selection:

```python
pipeline.build_embeddings(model='hubert', condition='word', gpu=True)
```

The four convenience wrappers accept the other `build_embeddings()` options
as keyword arguments and return the same dictionary of store paths:

```python
pipeline.build_hubert_word_embeddings(gpu=True)
pipeline.build_hubert_cross_boundary_embeddings(gpu=True)
pipeline.build_w2v_word_embeddings(gpu=True)
pipeline.build_w2v_cross_boundary_embeddings(gpu=True)
```

`model` accepts `'hubert'` or `'w2v'`; `condition` accepts `'word'` or
`'cross_boundary'`. Each defaults to `None`, meaning all values: specifying
only a model processes both datasets, and specifying only a condition uses
both models. Only selected datasets and stores are loaded. The returned
dictionary maps `(model, condition)` pairs to the processed store paths.
Missing stores are created using the saved selection's actual type count.
An existing matching store is resumed; a different type count or multiple
matching directories raises an error. Invalid selectors fail before opening
materials. This check compares directory names, not selection contents;
reuse the frozen selection when resuming.

`build_indexes()` builds fresh indexes; use `overwrite=True` explicitly to
replace existing index files. It reports missing requested embeddings
rather than dropping selected tokens.

To build a single index in an independent job:

```python
pipeline.build_indexes(model='hubert', condition='word', layers=(3,),
    pooling='center')
```

`model` accepts `'hubert'` or `'w2v'`, `condition` accepts `'word'` or
`'cross_boundary'`, and `pooling` accepts `'center'` or `'mean'`. Each defaults
to `None`, selecting both values. Only requested selections and embedding
stores are loaded; invalid selectors fail before opening materials or
creating output directories. The returned dictionary maps
`(model, condition, layer, pooling)` tuples to vector-array paths.

For individual operations use `word_lexicon`, `cross_boundary_lexicon`,
`embeddings`, and `vector_indices`. The old module names
`make_cross_boundary_embeddings` and `validate_lexicon` have been replaced.

See [Compact neighbour results](neighbour_results.md) for exporting 30 nearest
neighbours and all same-type global ranks, with float32 similarities and lazy
annotation lookup. Older retrieval examples document the previous dataset
and naming scheme. Aggregate evaluation metrics remain separate.
This pipeline neither deletes old remote files nor launches remote jobs.

## Tests

Run the local suite with numerical libraries limited to one thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../fiemrok_env/bin/python -B -m unittest discover -s tests
```

Integration tests build real temporary Phraser/Echoframe stores and all 32
vector-index conditions with synthetic model outputs, including the initial
Audio selection/copy stage. They do not load the
remote checkpoints or process the real CGN corpus.
