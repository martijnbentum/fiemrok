# CGN mean-pooling evaluation performance

## Observations

The long progress bar after `saving vector search index` belongs to
`analyse_results.evaluate_index()`, not index saving. Evaluation searches all
246,600 tokens, requests 11 neighbours per query, excludes self by source key,
and scores ten neighbours.

The evaluator now accepts `batch_size`, defaulting to 256. A local synthetic
benchmark showed a 23x retrieval speedup over individual queries, but that
speedup did not transfer to the remote server. The cause is not established.

Remote timings for the same 512 queries using the existing index:

| FAISS threads | Batch size | Seconds | Queries/second |
| ---: | ---: | ---: | ---: |
| 1 | 128 | 6.70 | 76.4 |
| 1 | 256 | 6.78 | 75.5 |
| 1 | 512 | 6.40 | 79.9 |
| 8 | 128 | 7.66 | 66.8 |
| 8 | 256 | 7.15 | 71.7 |
| 8 | 512 | 7.31 | 70.0 |
| 16 | 128 | 8.54 | 60.0 |
| 16 | 256 | 7.53 | 68.0 |
| 16 | 512 | 7.36 | 69.6 |
| 64 | 128 | 8.09 | 63.3 |
| 64 | 256 | 7.32 | 69.9 |
| 64 | 512 | 7.22 | 70.9 |

One thread and batch size 512 were fastest in this measurement. At 79.9
queries/second, retrieval over 246,600 queries takes approximately 51 minutes,
plus scoring overhead. These are single measurements, not repeated trials.
CPU percentages and ETA alone do not identify the bottleneck.

## Next diagnostic: separate search and Python overhead

Run in IPython with the existing `index` and `analyse_results` imported as
`ar`. This assumes the current FAISS backend. It reads the index, creates
temporary variables, and restores the original FAISS thread setting. It
does not save or overwrite mean-pooling artifacts or the `scores` and
`results` variables. Run after the current evaluation finishes; interrupting
evaluation loses the unfinished layer's query progress.

```python
import numpy as np
import faiss
from time import perf_counter

q = np.ascontiguousarray(index.vectors[:512], dtype=np.float32)
old_threads = faiss.omp_get_max_threads()

try:
    faiss.omp_set_num_threads(1)
    index.backend.index.search(q[:32], 11)

    start = perf_counter()
    distances, neighbours = index.backend.index.search(q, 11)
    raw_seconds = perf_counter() - start

    start = perf_counter()
    search_results = index.queries(q, top_k=11)
    wrapper_seconds = perf_counter() - start

    start = perf_counter()
    for i, result in enumerate(search_results):
        ar.precision_at_k(result, index.metadatas[i], top_k=10)
    scoring_seconds = perf_counter() - start

    print(f'Raw FAISS search:       {raw_seconds:.3f}s')
    print(f'Search with Python API: {wrapper_seconds:.3f}s')
    print(f'Precision scoring only: {scoring_seconds:.3f}s')
finally:
    faiss.omp_set_num_threads(old_threads)
```

- Raw FAISS dominates: investigate CPU allocation, memory bandwidth, the
  FAISS installation, and shared server contention.
- Python API is substantially slower: investigate normalization, array
  conversion, and metadata lookup.
- Scoring dominates: investigate source-key filtering and label comparisons.

Repeat timings before drawing conclusions; execution order and server load
can affect comparisons. The diagnostic has been proposed but its results
have not yet been reported.

## Check process resource limits

Run on the Linux server in IPython:

```python
import os

print('Available CPU affinity:', len(os.sched_getaffinity(0)))
print('FAISS build:', faiss.get_compile_options())

!cat /sys/fs/cgroup/cpu.max
!cat /sys/fs/cgroup/cpu.stat
```

With cgroup v2, `cpu.max` can reveal a CPU quota despite many visible CPUs.
Read `cpu.stat` before and after a benchmark to check whether throttling
counters increase. Missing files require inspecting the server's applicable
resource controls. Also compare the same benchmark when the server is
quieter to test for shared resource contention.

## Current evaluation setting

```python
faiss.omp_set_num_threads(1)
scores = ar.evaluate_index(index, top_k=10, batch_size=512)
```

The evaluation returns scores; the caller must save them. Yesterday's run
snippet saved completed layers in `cgn_lexicon_w2v_mean_results.json` and
`cgn_lexicon_hubert1_mean_results.json`. Reloading `analyse_results` only
updates code after the changed file has been pulled onto the server.
