# Cross-boundary neighbours with mean and center pooling

The existing `validate_lexicon.py` and `analyse_results.py` functions support
both datasets. The index helpers have optional arguments; existing word
calls keep their stores, names, pooling, and `unit_type='word'` defaults.

- `make_vsi_for_layers(..., embedding_store=..., name_prefix=...)` uses an
  already-open store and prefixes its index names. A prefix such as
  `cross_boundary_lexicon100_2200_` separates these indexes from word indexes.
- `unit_type='marker'` is supported by `make_vector_metadata`,
  `make_vector_search_index_for_layer`, and `make_vsi_for_layers`.
- Existing saved indexes are reused. To change the unit type of an existing
  index, rebuild it with `overwrite=True` or choose another name.

The 2,466-type completeness check in `hubert_mean_pooling.md` is inline code
in that example, not a function. For the cross-boundary sample, the check
uses 2,200 types and 100 token scores per type. Evaluation itself is generic.

## Run both models and both pooling methods

Run from the remote repo after pulling the updated code, in a fresh
Python/IPython session. This processes one layer at a time, reuses saved
indexes, skips completed score layers, and saves scores after each layer.
There are four result files, each containing layers 3, 6, 9, and 12.

```python
import json

import faiss

import analyse_results as ar
import locations
import stores
import validate_lexicon as vl

faiss.omp_set_num_threads(1)
output_dir = locations.repo / 'svs_data'
output_dir.mkdir(exist_ok=True)

for model, path in (
    ('w2v', locations.cross_boundary_wav2vec_store),
    ('hubert1', locations.cross_boundary_hubert_store),
):
    embedding_store = stores.load_echoframe_cross_boundary_store(path)
    try:
        for pooling in ('center', 'mean'):
            prefix = f'cross_boundary_lexicon100_2200_{model}_{pooling}'
            filename = output_dir / f'{prefix}_results.json'
            results = json.loads(filename.read_text()) if filename.exists() else {}

            for layer in (3, 6, 9, 12):
                key = str(layer)
                if key in results:
                    scores = results[key]
                    if len(scores) != 2200 or any(
                        len(values) != 100 for values in scores.values()
                    ):
                        raise ValueError(f'Incomplete saved scores: {prefix}, layer {layer}')
                    print(f'Skipping {prefix}, layer {layer}', flush=True)
                    continue

                name = f'cross_boundary_lexicon100_2200_{model}_{layer}_{pooling}'
                index, _ = vl.make_vector_search_index_for_layer(
                    embedding_store, layer, name, pooling=pooling,
                    unit_type='marker', directory=str(output_dir),
                    return_metadata=False,
                )
                scores = ar.evaluate_index(index, top_k=10, batch_size=512)
                results[key] = scores
                temporary = filename.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(results, ensure_ascii=False),
                    encoding='utf-8')
                temporary.replace(filename)
                mean, std = ar.overall_result(scores)
                print(f'{prefix}, layer {layer}: {mean:.4f}; saved', flush=True)
                del index
    finally:
        embedding_store.close()
```

The result files are stored in `svs_data`:

- `cross_boundary_lexicon100_2200_w2v_center_results.json`
- `cross_boundary_lexicon100_2200_w2v_mean_results.json`
- `cross_boundary_lexicon100_2200_hubert1_center_results.json`
- `cross_boundary_lexicon100_2200_hubert1_mean_results.json`

Scores are precision at ten non-self neighbours, grouped by phoneme sequence.
An interrupted layer restarts its queries; its completed index remains
reusable. A partial index requires `overwrite=True` to rebuild that index.
Both stores were checked on fancypants: one model, one 500 ms collar, and
220,000 matching marker keys at each layer, so layer filtering is sufficient
for these stores.
