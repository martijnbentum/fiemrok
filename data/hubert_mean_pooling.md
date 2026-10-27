# Resume HuBERT mean pooling

Run from the remote `repo/` directory with its environment activated and the
updated Fiemrok code available. Install
the Echoframe batch-reader fix (version 0.1.191 or later), then start a fresh
Python/IPython session so the updated code is imported:

```bash
python -m pip install --upgrade git+https://github.com/martijnbentum/echoframe.git
```

The standard layer builder now reads an entire layer through
`Store.load_many_frames()`. Echoframe groups reads by shard, opens each shard
once per call, and returns pooled vectors in input order. There is no
per-token loading progress bar; the loader prints when loading starts and
finishes. FAISS query batches remain separate from frame loading.

`return_metadata=False` returns `(index, None)`. Existing complete indexes
load without opening the embedding metadata or resolving Phraser labels.
The default `True` still returns Echoframe metadata in saved-index order.
Mean pooling uses Echoframe's existing arithmetic; previously saved indexes
are reused without recomputing vectors.

```python
import json
from pathlib import Path

import faiss

import analyse_results as ar
import validate_lexicon as vl

faiss.omp_set_num_threads(1)
filename = Path('cgn_lexicon_hubert1_mean_results.json')
results = json.loads(filename.read_text()) if filename.exists() else {}
hubert_store = vl.load_store_hubert1()

try:
    for layer in [3, 6, 9, 12]:
        key = str(layer)
        if key in results:
            scores = results[key]
            if len(scores) != 2466 or any(
                len(values) != 100 for values in scores.values()):
                raise ValueError(f'Incomplete saved scores for layer {layer}')
            print(f'Skipping completed HuBERT layer {layer}', flush=True)
            continue

        index, _ = vl.make_vector_search_index_for_layer(
            hubert_store, layer, f'hubert1_{layer}', pooling='mean',
            return_metadata=False)
        scores = ar.evaluate_index(index, top_k=10, batch_size=512)
        results[key] = scores

        temporary = filename.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(results, ensure_ascii=False),
            encoding='utf-8')
        temporary.replace(filename)

        mean, std = ar.overall_result(scores)
        print(f'HuBERT layer {layer}: {mean:.4f}; saved', flush=True)
finally:
    hubert_store.close()
```

The score-count check is specific to this lexicon: 2,466 word types and 100
tokens per type. An interrupted evaluation restarts that layer's queries;
its complete saved index remains reusable. A partial index raises an error
before metadata loading; `overwrite=True` explicitly rebuilds it.

The loaders attach `stores.cgn` if it is set. Closing an embedding store
leaves its Phraser store open. Keep a shared Phraser store open until all
its users finish. In a dedicated session, call
`hubert_store.close_phraser_stores()` when it is no longer needed.
