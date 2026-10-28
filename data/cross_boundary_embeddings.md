# Cross-boundary marker embeddings

`make_cross_boundary_embeddings.py` computes full-frame hidden states for
the persisted sample indexed by `svs_data/cross_boundary_marker_keys.pkl`.
It uses the same checkpoints and settings as the existing word stores:

| Model argument | Registered checkpoint | Reference store | Default destination |
| --- | --- | --- | --- |
| `wav2vec2` | `wav2vec2_nl1_checkpoint-200000` | `lexicon100_2466_cgn` | `cross_boundary100_2200_cgn` |
| `hubert` | `huibert_nl1_checkpoint-200000` | `lexicon100_2466_cgn_hubert` | `cross_boundary100_2200_cgn_hubert` |

Default store paths are relative to the module directory. The key-file path
is relative to the working directory, as in `cbl.load_markers`.
Pass `root`, `reference_root`, or `keys_path` to override these locations.
The destination names describe the current 2,200-type, 100-token sample;
choose another root when computing embeddings for another sample.

Only the selected checkpoint registration is copied from the reference
store's `config.json`. Its original model ID is not copied: EchoFrame assigns
the destination's ID. A configured destination can be reopened without the
reference store. Both destinations register the existing CGN phraser store
as source `cgn`, using the shared `stores.py` instance unless `phraser_store`
is supplied explicitly.

## Compute

```python
import make_cross_boundary_embeddings as cb

wav2vec_store = cb.compute_embeddings(model='wav2vec2', gpu=True)
wav2vec_store.remove_cached_model()
wav2vec_store.close()

hubert_store = cb.compute_embeddings(model='hubert', gpu=True)
hubert_store.remove_cached_model()
hubert_store.close()
```

Defaults are layers `(3, 6, 9, 12)`, `collar=500`, and `batch_size=32`.
The collar is context on each side in milliseconds. The inspected reference
stores both encode a 500 ms collar in every entry. An alternative such as
`collar=2000` is supported and produces separate entries.

The function loads persisted markers through `cbl.load_markers`. To reuse
markers already loaded, pass `markers=markers`. Supplied objects must be
persisted phraser markers from the source store. The function returns an open
EchoFrame store; the caller should close it when finished. Its loaded model
is cached, so release it with `remove_cached_model()` before computing the
other model if both stores would otherwise retain GPU models.

Existing marker/model/layer/collar combinations are skipped. Rerun the same
call to continue after interruption, or add layers to compute missing ones.
Records receive tags `cgn` and `cross_boundary`. Each entry uses the marker's
own key; `metadata.phraser_object` resolves to that marker, not its phrase.

All frame vectors fully contained in the marker interval are saved as a
matrix `(n_frames, hidden_size)`, with no pooling. The surrounding collar is
used during model inference but its frame rows are not stored. This matches
the word-lexicon extraction rule (100% frame overlap). Base checkpoints have
768 hidden dimensions. A complete 220,000-marker sample at four layers has
880,000 entries in each model's store.

## Small real-model check

Start with a few markers in the final destination. They will be skipped
when the full computation runs later.

```python
import cross_boundary_lexicon as cbl
import stores

cgn = stores.load_phraser_cgn_store()
markers = cbl.load_markers(cgn)
embedding_store = cb.compute_embeddings(markers=markers[:4], gpu=True,
    batch_size=4)
embedding_store.remove_cached_model()
embedding_store.close()

embedding_store = cb.load_store()
embedding = embedding_store.phraser_key_to_embedding(markers[0].key,
    model_name='wav2vec2_nl1_checkpoint-200000', layer=6, collar=500)
assert embedding.phraser_object.key == markers[0].key
assert embedding.object_class == 'Marker'
assert embedding.data.ndim == 2
assert embedding.shape[0] > 0 and embedding.shape[1] == 768
print(embedding)
embedding_store.close()
```

Integration tests exercise real phraser and EchoFrame stores with synthetic
model outputs. They verify exact stored frame rows, marker links after
reopening, model registration, missing-layer resume, and skipped inference
for completed requests. Real checkpoint inference remains the responsibility
of the small check above.
