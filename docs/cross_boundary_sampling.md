# Adaptive cross-boundary lexicon sampling

```python
from lexicon_validation import cross_boundary_lexicon as cbl
from lexicon_validation.pipeline import cross_boundary_type_bounds

ideal_counts, _ = cbl.inspect_lexicon_phonemes(selected_words)
filtered_phrases = cbl.filter_phrases_for_cross_boundary_lexicon(phrases)
lexicon = cbl.CrossBoundaryLexicon(filtered_phrases,
    min_duration_ms=100, max_duration_ms=1500)
plan = cbl.plan_sampling(lexicon, ideal_counts, n_tokens=100,
    tolerance=10)
print(plan)
min_types, _ = cross_boundary_type_bounds(sum(ideal_counts.values()))
sampled_tokens = lexicon.sample_lexicon(plan, min_types=min_types,
    tries_per_round=3, max_attempts=100, random_seed=0)
result = lexicon.sampling_result
```

The lexicon owns inclusive token duration limits (100–1500 ms by default).
Candidate collection filters on `sequence.end - sequence.start`, then
applies per-type 5 * IQR duration filtering before counting eligible
occurrences. Planning and sampling share these filters.
Extracted sequences and raw `sequence_counts` remain available for inspection.
Create a new plan after changing the limits.

The local material-store stage selects Netherlandic Dutch and excludes
components `acdhm` by default. Sampling does not repeat those filters.
Word selection lives in `word_lexicon.py`, allows any syllable
count, and excludes one-phone words. Its saved selection supplies the
phone-count targets above. `fiemrok_experiment` has independent rules.
See the [pipeline](lexicon_validation_pipeline.md) for both datasets.

The plan owns the token count and tolerance. It starts from all sufficiently
frequent types and balances deviations from the original length proportions
under the frequency limits. It preserves the original total if possible.
The frequency ceiling does not account for phone overlap. A plan below
`min_types` fails before sampling. The pipeline derives the minimum from
the selected word-type count rather than using an absolute 2,000-type floor.

Quota allocation minimizes weighted squared error between normalized length
proportions: `sum((actual_share - ideal_share)**2 / weight)`, where `weight`
is the ideal share, with a floor of one original type for zero-target bins.
This penalizes relative overrepresentation instead of sending an entire
shortage to the nearest available length. For a fixed total, quotas are an
integer optimum of this objective under the supplied capacity limits.
Smaller totals are compared using proportions against the same original
ideal; the reference never changes during adaptation. This objective can
prefer a different result from total variation distance or mean length.

Each current quota has bounds of ±max(tolerance, floor(10% of quota)), with
its lower bound clamped to zero. The current total is exact for each attempt.
Zero-target bins remain available up to their upper bound, so a long length
can disappear or contribute tokens depending on actual availability. There
is no ten-phone cutoff.

Sampling first fills bin minimums round-robin, one successful type per bin
per cycle. Bins with nonzero minimums go first, ordered by eligible types per
requested type (scarce first); optional bins follow with the same ordering.
Once minimums are met, each remaining type goes to the feasible bin with the
smallest marginal distribution error against the original ideal. This can
favor a long bin over a short bin even when both satisfy their minimums.
Every attempt starts with fresh bins and reservations; no failed attempt
changes phrase selections.

Three consecutive seeds use identical quotas by default. While no sample
has succeeded, failed rounds reduce the requested total by 5% (rounded up),
never below `min_types`. Once a sample succeeds, all remaining attempts
preserve its type count and optimize distribution fit. The pipeline sets
the floor to 90% of the word-type count and starts with the full word-type
goal, subject to frequency eligibility.

Capacity observations persist across rounds within a search. Bins exhausted
in every failed attempt in a round use the best count observed across all
attempts, including successes, as a learned capacity estimate. Learning a
limit for five phones preserves previously learned limits for six, seven,
and eight phones. Later observed counts can raise those estimates. These
are heuristic estimates under competition, not proven capacity limits.
If estimates cannot supply the requested total, only the missing slots are
allocated beyond them; other learned limits are not reset.

After success, two of every three subsequent rounds reuse the best
successful plan with new seeds. The third round tests
quotas calculated from the learned estimates, increasing one constrained
bin's estimate by 10% (rounded up, at least one type), capped by frequency
eligibility. Probes rotate through constrained positive-ideal bins; a failed
probe does not erase earlier limits. With no success yet, ordinary rounds
continue to use the learned estimates. The best plan can contain more than
`min_types`; reusing it does not restart the total reduction sequence.

The sampler uses the full `max_attempts` search budget, including attempts
after the first success. It prioritizes type count, then the lowest weighted
distribution error, then the earlier success. Only the retained result is
committed after the search.
Consequently a run can return an earlier success even if later attempts
fail. `max_attempts=1` remains available for a single attempt.

Every round prints quotas, learned estimates, and limits. Failure tables
include frequency-eligible counts, provisional selected counts, shortfalls
below each minimum, and overshoots above each target. Overshoot is allowed
within the quota tolerance; it does not mean the maximum was exceeded.
New best successes and the final retained sample report eligible and actual
length counts, quota overshoot, weighted distribution error, total variation
distance, and mean-phone-count shift. A second mean reports bins 2–10 only,
renormalizing the ideal and sampled counts within that range. Both means are
diagnostics and do not affect sample ranking. Distortion always refers to the
original ideal.
`result.seed` and `result.selected_attempt` identify the retained attempt;
`result.attempts` is the total search effort. The original plan is unchanged.

To reproduce a success, retain both its seed and its final plan:

```python
lexicon.reset_selection()
sampled_tokens = lexicon.sample_lexicon(result.plan, random_seed=result.seed,
    max_attempts=1, min_types=min_types)
```

To continue after the attempt limit, preserve the final constraints too:

```python
from lexicon_validation.cross_boundary_sampling import SamplingFailure

try:
    sampled_tokens = lexicon.sample_lexicon(plan, min_types=min_types)
except SamplingFailure as error:
    resume_plan = error.plan
    next_seed = error.next_random_seed
    print(error)

# If you decide to continue:
sampled_tokens = lexicon.sample_lexicon(resume_plan, min_types=min_types,
    random_seed=next_seed, max_attempts=100)
```

The candidate pool and iteration order must remain the same for seed-based
reproduction. Failure does not prove impossibility. Sampling requires a
fresh selection; use `reset_selection()` before sampling an already selected
lexicon again.

Planning, reference lexicon inspection, sampling, and reporting live in
`cross_boundary_sampling.py`; `cbl.plan_sampling` and
`cbl.inspect_lexicon_phonemes` remain convenient public entry points.
