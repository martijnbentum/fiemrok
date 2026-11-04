# Word and cross-boundary embeddings

The shared extraction module is now `lexicon_validation/make_embeddings.py`.
For the complete workflow use the
[speech-vector index pipeline](build_speech_vector_index_pipeline.md).

```python
from lexicon_validation import build_speech_vector_index_pipeline as pipeline

# Uses the existing saved word and cross-boundary selections.
pipeline.build_embeddings(gpu=True, batch_size=32)
```

For a single dataset/model using the saved selection:

```python
pipeline.build_embeddings(model='hubert', condition='cross_boundary',
    gpu=True, batch_size=32)
```

To pass selected objects explicitly instead:

```python
from lexicon_validation import cross_boundary_lexicon as cbl
from lexicon_validation import lv_stores
from lexicon_validation import make_embeddings

cgn = lv_stores.load_phraser_store()
markers = cbl.load_markers(cgn)
hubert_store = make_embeddings.compute_embeddings_for_hubert(
    markers, condition='cross_boundary', gpu=True, batch_size=32)
hubert_store.close()
cgn.close()
```

The wrappers derive the actual type count and name stores under
`echoframe_stores/{model}_{condition}_{n_types}` in the validation workspace.
They return an open store, which the caller closes. `compute_embeddings`
accepts a caller-owned store, selected items, and condition, and leaves that
store open. `create_store` requires the local Phraser store as its `cgn`
argument. Model wrappers attach the store owning their supplied items.

Defaults are layers 3, 6, 9, 12, a 500 ms collar on each side, and batch size
32. Both models use version 1/checkpoint 200000 from `model_paths.py`.
Full frame matrices within the selected token are stored without pooling;
context frames are excluded. Metadata links to the word or marker's own
Phraser key and carries `cgn` plus the dataset condition as tags.

The existing Echoframe batch extractor skips completed token/model/layer/
collar combinations. Repeat extraction after interruption without rerunning
selection. Tests use real Phraser/Echoframe stores and synthetic inference.
