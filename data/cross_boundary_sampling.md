# Adaptive cross-boundary lexicon sampling

```python
import cross_boundary_lexicon as cbl

ideal_counts, _ = cbl.inspect_lexicon_phonemes()
filtered_phrases = cbl.filter_phrases_for_cross_boundary_lexicon(phrases)
lexicon = cbl.CrossBoundaryLexicon(filtered_phrases)
plan = cbl.plan_sampling(lexicon, ideal_counts, tokens_per_type=100,
    tolerance=10)
print(plan)
sampled_tokens = lexicon.sample_lexicon(plan, min_types=2200,
    tries_per_round=3, max_attempts=100, random_seed=0)
result = lexicon.sampling_result
```

The plan owns the token count and tolerance. There is no capacity fraction.
It starts from all sufficiently frequent types, retaining original counts
where supported and moving deficits to nearby lengths with capacity. It
preserves the original total if possible. The frequency ceiling does not
account for phone overlap. A plan below `min_types` fails before sampling;
for small experiments, explicitly choose a smaller minimum.

Each current quota has bounds of ±max(tolerance, floor(10% of quota)), with
its lower bound clamped to zero. The current total is exact for each attempt.
Zero-target bins remain available up to their upper bound, so a long length
can disappear or contribute tokens depending on actual availability. There
is no ten-phone cutoff.

Sampling visits bins round-robin, one successful type per bin per cycle.
Bins with nonzero minimums go first, ordered by eligible types per requested
type (scarce first); optional bins follow with the same scarcity ordering.
Minimums are filled before the remaining total. Every attempt starts with
fresh bins and reservations; no failed attempt changes phrase selections.

Three consecutive seeds use identical quotas by default. After a failed
round, bins exhausted in every attempt are assigned a preference based on
the best count observed for that bin. Missing types are redistributed toward
other lengths with frequency capacity. These observations are heuristics,
not proofs of feasible capacity: different attempts compete for different
phones. If redistribution stalls or revisits prior quotas, the total falls
by 5% (rounded up), never below `min_types`. At that floor, remaining attempts
continue to explore seeds even if quotas cannot be relaxed further. The
100-attempt limit includes all quota rounds, including an incomplete final
round.

Every round prints its quotas and limits; failures show provisional counts
and minimum shortfalls. On success the sampler reports the actual length
distribution, total type count, successful seed, total variation distance,
and mean-phone-count shift. Distortion always refers to the original ideal
counts, not the most recent quota plan. The original plan is not modified.

To reproduce a success, retain both its seed and its final plan:

```python
lexicon.reset_selection()
sampled_tokens = lexicon.sample_lexicon(result.plan, random_seed=result.seed,
    max_attempts=1, min_types=2200)
```

To continue after the attempt limit, preserve the final constraints too:

```python
from cross_boundary_lexicon_sampling import SamplingFailure

try:
    sampled_tokens = lexicon.sample_lexicon(plan, min_types=2200)
except SamplingFailure as error:
    resume_plan = error.plan
    next_seed = error.next_random_seed
    print(error)

# If you decide to continue:
sampled_tokens = lexicon.sample_lexicon(resume_plan, min_types=2200,
    random_seed=next_seed, max_attempts=100)
```

The candidate pool and iteration order must remain the same for seed-based
reproduction. Failure does not prove impossibility. Sampling requires a
fresh selection; use `reset_selection()` before sampling an already selected
lexicon again.

Planning, reference lexicon inspection, sampling, and reporting live in
`cross_boundary_lexicon_sampling.py`; `cbl.plan_sampling` and
`cbl.inspect_lexicon_phonemes` remain convenient public entry points.
