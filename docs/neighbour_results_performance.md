# Cheerilee neighbour-results benchmark

Benchmark of the compact-results computation on `cheerilee.science.ru.nl`
(hostname `mlp05`), using the existing validation index
`hubert_cross_boundary_1572_6_mean`: 157,200 tokens, 768 dimensions,
100 tokens per type.

## Method

- Same first 512 query tokens against every candidate for each trial.
- Current local `_cosine_batches` and `_write_batches` source executed in a
  temporary remote process; the remote checkout was not changed.
- Includes cosine scoring, stable per-query sorting, extraction of 30
  non-self neighbours and all same-type ranks/similarities, and LZF HDF5
  writes/flush. Outputs were temporary and removed after each trial.
- Excludes interpreter startup, index/mapping identity checks, metadata
  loading, and FAISS index construction. Vectors were loaded/normalized
  once, outside timed trials. Loading vectors and metadata took 16.90 s.
- OpenBLAS 0.3.31.188.0 (Haswell), NumPy, 64 CPUs available. Thread counts
  controlled with `threadpoolctl.threadpool_limits(..., user_api="blas")`.
  The server normally defaults to 64 OpenBLAS threads.
- One warmup, then 28 thread/batch combinations in fixed random order.
  The fastest three combinations were repeated once. These are limited
  trials on one index and query subset, not a general performance guarantee.

## Full sweep

Total seconds for the same 512 queries, first measured trial:

| BLAS threads | Batch 64 | Batch 128 | Batch 256 | Batch 512 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.905 | 15.551 | 15.383 | 15.315 |
| 2 | 13.693 | 13.562 | 13.479 | 13.356 |
| 4 | 12.691 | 12.478 | 12.507 | 12.389 |
| 8 | 12.112 | 11.966 | 11.979 | 11.883 |
| 16 | 11.913 | 11.743 | 12.079 | 11.657 |
| 32 | 11.941 | 11.882 | 11.668 | 11.576 |
| 64 | 12.291 | 12.056 | 11.945 | 11.835 |

## Repeated fastest settings

| Threads | Batch | First trial (s) | Repeat (s) |
| ---: | ---: | ---: | ---: |
| 32 | 512 | 11.576 | 11.540 |
| 16 | 512 | 11.657 | 11.646 |
| 32 | 256 | 11.668 | 11.614 |

The fastest measured configuration was **32 BLAS threads, batch 512**.
It took 11.576 s and 11.540 s, versus 15.315 s for one thread/batch 512
in the sweep: approximately 25% less elapsed time. Sixteen threads/batch
512 and 32 threads/batch 256 were within about 1% of the fastest setting.

In the fastest repeat:

- Similarity scoring: 0.419 s.
- Rank sorting/extraction and other loop overhead: 11.120 s (about 96%).
- HDF5 dataset writes: 0.0012 s; remaining flush overhead is included above.

The main bottleneck is now sorting/rank extraction. The old FAISS
one-thread result does not transfer to this NumPy computation. More BLAS
parallelism helps scoring, but has diminishing effect on overall time.

## Using the measured setting

```python
from threadpoolctl import threadpool_limits
from lexicon_validation.retrieval_evaluation import export_neighbours

with threadpool_limits(limits=32, user_api='blas'):
    paths = export_neighbours('hubert', 6, 'mean',
        condition='cross_boundary', batch_size=512)
```

This temporarily changes BLAS limits and restores them afterwards.
`faiss.omp_set_num_threads(1)` does not control this computation.
No application defaults were changed as part of the benchmark.

At this token count, batch 512 needs about 322 MB for one temporary float32
score matrix, versus 40 MB at batch 64; vectors and sorting need additional
memory. The benchmark process peaked around 1.9 GiB, but did not construct
a FAISS index. A normal export also holds that backend in memory.

A linear extrapolation from the fastest trials is approximately 59 minutes
for this 157,200-query condition. A full export was not run, and other
models/layers/datasets, system load, and memory effects can change timing.
