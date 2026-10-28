# Cross-boundary marker embeddings

`make_cross_boundary_embeddings.py` puts computation first and setup at the
bottom of the file:

- `compute_embeddings_for_hubert(**kwargs)` and
  `compute_embeddings_for_wv2(**kwargs)` create or open the corresponding
  store, register its model if needed, forward options to `compute_embeddings`,
  and return the open store.
- `compute_embeddings(embedding_store, ...)` gets CGN from that open store,
  loads the saved markers, reads the store's sole model registration, and
  calls EchoFrame's batch embedding function.
- `create_store(path, model_name, architecture)` opens one store and registers
  its checkpoint through `model_paths.py` if the model is absent.
- `create_wv2_and_hubert()` calls `create_store` for each model and returns
  `(wav2vec_store, hubert_store)`. Setup can be repeated.

## Setup and compute

```python
import make_cross_boundary_embeddings as cb

wav2vec_store, hubert_store = cb.create_wv2_and_hubert()

cb.compute_embeddings(wav2vec_store, gpu=True)
wav2vec_store.remove_cached_model()
wav2vec_store.close()

cb.compute_embeddings(hubert_store, gpu=True)
hubert_store.remove_cached_model()
hubert_store.close()
```

The registrations use version 1/checkpoint 200000:
`wav2vec2_nl1_checkpoint-200000` and `huibert_nl1_checkpoint-200000`.
The checkpoint paths come directly from `model_paths.py`.

Compute a model directly with a wrapper; no separate setup is required:

```python
hubert_store = cb.compute_embeddings_for_hubert(gpu=True, batch_size=32)
hubert_store.remove_cached_model()
hubert_store.close()

wav2vec_store = cb.compute_embeddings_for_wv2(gpu=True)
wav2vec_store.remove_cached_model()
wav2vec_store.close()
```

The wrappers accept the same options as `compute_embeddings`, including
`layers`, `collar`, `gpu`, `batch_size`, and `verbose`.

`locations.py` defines all three repo-relative paths:

- `cross_boundary_wav2vec_store`: `cross_boundary_lexicon100_2200_cgn/`
- `cross_boundary_hubert_store`: `cross_boundary_lexicon100_2200_cgn_hubert/`
- `cross_boundary_marker_keys`: `svs_data/cross_boundary_marker_keys.pkl`

The two store directories are directly under the repo root. These constants
are absolute paths anchored to the location of `locations.py`, so they do
not depend on the current working directory. They do not use the existing
`locations.data` parent-directory path.

## Open configured stores later

```python
import locations
import stores

wav2vec_store = stores.load_echoframe_cross_boundary_store()
hubert_store = stores.load_echoframe_cross_boundary_store(
    locations.cross_boundary_hubert_store)
```

The loader attaches the shared CGN store. It does not register models;
`create_wv2_and_hubert()` handles setup. The caller owns each open store and closes
it when finished. `compute_embeddings` does not open or close stores.

## Frames and resume

Defaults are layers `(3, 6, 9, 12)`, `collar=500`, and `batch_size=32`.
The collar is context on each side in milliseconds. Both inspected word
stores encode a 500 ms collar. Pass `collar=2000` for a different context
window; that produces separate entries.

All frame vectors fully contained in each marker are saved as a matrix
`(n_frames, hidden_size)`, without pooling. The collar provides inference
context, while its frame rows are excluded. Metadata uses the marker's own
key and CGN source, so `metadata.phraser_object` resolves to the marker.
Entries have tags `cgn` and `cross_boundary`.

Existing marker/model/layer/collar combinations are skipped. Repeat the same
compute call after interruption, or add layers to compute missing ones.
A complete 220,000-marker sample at four layers has 880,000 entries per model.

Integration tests use real phraser and EchoFrame stores with synthetic model
outputs to verify exact frame rows, marker linkage after reopening, setup
from model paths, repeated setup, and skipped inference on completed requests.
