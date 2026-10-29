# Cross-boundary lexicon refactor proposal

Status: reviewed proposal (2026-10-28), nothing implemented yet.
Covers `cross_boundary_lexicon.py` and `cross_boundary_lexicon_sampling.py`.
Goal: preserve selected occurrences, their order, successful seeds, quota
transitions, and sampling guarantees while making ownership clearer.
Equality changes and public API removals are separate decisions.

## Problems

1. **Private coupling between the sampler and data model.**
   - The sampler reads `ps._occupied` and traverses `lexicon.phrase_sequences`.
   - Calls to public lexicon methods are reasonable; the private mask access
     and unclear ownership of sampling state need attention.
   - The CBL re-exports (`SamplingPlan`, `plan_sampling`, and
     `inspect_lexicon_phonemes`) are useful public conveniences, not private
     dependencies. Retain these entry points.
2. **Overlap operations are duplicated.**
   - `PhraseSequences` checks and marks its committed occupancy mask.
   - The sampler repeats those operations on attempt reservations and
     per-type trial masks.
   - These states must remain separate: a failed attempt must not commit
     selections, and a failed type must not leak reservations.
3. **Candidates are collected twice.** `plan_sampling` and `sample_lexicon`
   each scan all phrases. Avoiding this duplicate pass is more valuable than
   avoiding the small amount of repeated length-bin construction.
4. **`sample_lexicon` mixes responsibilities.** The search loop handles
   attempts, next-plan selection, best-result tracking, and reporting.
   `relax_plan` duplicates the shrink step and is only used in tests.
5. **Plan descriptions and mutable state share classes.**
   - `LengthBin` combines quota bounds with per-attempt remaining/selected
     types.
   - `SamplingFailure` represents both failed attempts and failed searches.
   - `sample_types_from_bins(minimums_only=...)` hides two different traversal
     strategies behind a flag.
6. **Reference inspection is misplaced.** `inspect_lexicon_phonemes` analyses
   the CGN word lexicon and belongs with existing lexicon inspection code.
7. **Smaller cleanup opportunities.**
   - `successful_seed` duplicates `sampling_result.seed` as stored state.
   - Three counters share counting logic, but have different source iterables.
   - `sequence.n_phones < 2` is redundant for sequences produced by the normal
     cross-boundary extraction path.
   - `describe_distortion` repeats mean-phone calculations.
   - Blank lines and line lengths could follow the existing repository style
     more consistently.

## Target layout

Keep two cross-boundary modules; a separate planning module is unnecessary.
Keep public entry points near the top and implementation helpers below them.

```
cross_boundary_lexicon.py           data model + phrase filters + markers
  CrossBoundaryLexicon              phrases, durations, candidates, commit,
                                    markers; successful_seed as a property
  PhraseSequences                   public overlap operations; committed mask
  CrossBoundarySequence             one occurrence; retain current equality
                                    during the structural refactor
  filter_* / load_markers            public helpers
  existing sampling entry points    retained as conveniences

cross_boundary_lexicon_sampling.py  planning + sampling through public APIs
  plan:     LengthBin (bounds only), SamplingPlan, plan_sampling,
            allocate_targets, distribution_loss, describe_distortion
  attempt:  BinState, sample_attempt, _sample_minimums, _sample_rest,
            _sample_type
  search:   CapacityEstimates, sample_lexicon, _next_plan, _better
  AttemptFailure / SamplingFailure

validate_lexicon.py                 + inspect_lexicon_phonemes
```

## Implementation order

0. **Capture behavior before restructuring.** Add deterministic comparisons
   across several seeds and representative success/failure cases, using the
   checks below. Review each stage against that baseline.
1. **Collect candidates once.** Prefer an explicit candidate mapping shared
   by planning and search, keeping plans lightweight. If `plan.candidates`
   is chosen instead, every `with_targets()` plan must share the same pool;
   do not copy the pool for each quota round. Treat the pool as a snapshot:
   changing duration limits or tokens per type requires rebuilding it.
   Attempts must not mutate its candidate lists or iteration order.
2. **Expose shared overlap operations.** Add a public mask-creation API and
   common check/reserve primitives usable with different masks. An
   `empty_mask()` method alone removes private access but does not remove
   duplicated overlap logic. Keep committed selections, attempt reservations,
   and per-type trial masks independent. Retain final `select()` checks and
   use `id(ps)` keys initially, avoiding equality/hash changes in this stage.
3. **Extract search decisions.** Keep the attempt loop visible; extract
   `_next_plan(...)` and `_better(a, b)`. Preserve the existing round counters,
   reductions, persistent capacity estimates, probe rotation, best-plan reuse,
   score ordering, and tie-breaking. Delete `relax_plan` once its useful
   behavior has moved into the search helper; test that behavior there rather
   than keeping a test-only production function.
4. **Separate descriptions from attempt state.** `LengthBin` owns bounds;
   `BinState` owns remaining/selected types. Introduce `AttemptFailure` for
   per-attempt diagnostics while retaining `SamplingFailure` and its public
   `plan` / `next_random_seed` fields for exhausted searches. Split the
   minimum-fill and remaining-fill phases into named functions, sharing small
   primitives without obscuring their different ordering rules.
5. **Perform compatible lexicon cleanup.** Derive `successful_seed` from
   `sampling_result`, preserving its public name and `None` behavior before
   sampling and after reset. Share counter implementation while retaining
   the existing public properties/methods and their meanings. In particular,
   raw sequence counts must not silently become duration/frequency-filtered
   counts. Retain `words`, `has_vowel`, `max_phone_gap`, and
   `available_sequence_counts()` for inspection and possible future filters.
   Remove the redundant phone-count guard only under the normal extraction
   invariant. Share mean-phone calculations while preserving normalization
   separately for overall and bins-2-through-10 diagnostics.
6. **Move reference inspection.** Move `inspect_lexicon_phonemes` to
   `validate_lexicon.py`. Keep the CBL convenience entry point and use lazy
   imports where needed: importing CBL should not pull in the full evaluation
   dependency stack. Update the usage notes.
7. **Style pass.** Follow applicable repository instructions and existing
   style. Keep this separate from algorithmic changes where practical.

Internal signature changes must be reflected in tests and `data/*.md` usage
notes. Preserve the existing CBL entry points with thin adapters if needed;
removing or renaming public conveniences is outside this refactor.

## Separate equality decision

`CrossBoundarySequence.__eq__` currently compares pronunciation strings.
Two different occurrences can therefore compare equal; tests explicitly
expect that behavior. `PhraseSequences.__eq__` compares source-phrase
identity. Both classes are unhashable, as is normal for classes defining
custom equality without a corresponding hash.

Identity equality would make sense for occurrence objects, but it changes
public behavior and should be reviewed separately. Compare
`phoneme_sequence` explicitly when checking whether occurrences have the
same type. Do not simply add an identity hash while retaining the existing
value-based equality: equal objects must have equal hashes. Before using
`PhraseSequences` directly as mask-dictionary keys, decide its equality and
hashing together. Keeping `id(ps)` avoids that dependency for now.

## Regression checks

- Compare ordered token identities `(phrase.key, start_index, end_index)`
  before and after the refactor across several seeds. Use source keys/spans,
  not pronunciation labels or process-specific object IDs.
- Compare winning seed, attempt counts, final quotas, and intermediate quota
  transitions, including failures followed by recovery and later successes.
- Preserve candidate ordering, shuffle/pop order, RNG-call ordering, scarcity
  and marginal-cost tie-breaking, and the reduction/probe/reuse schedule.
- Verify exact tokens per type, duration limits, total floors, quota bounds,
  and no shared selected phones.
- Verify that failed types leak no reservations and failed attempts/searches
  leave committed selections unchanged. Reset must restore fresh sampling.
- Preserve original-ideal scoring, overall and bins-2-through-10 diagnostics,
  frequency eligibility, overshoot, and minimum-shortfall meanings.
- Keep marker export/loading and exact phone reconstruction checks passing.
- Verify that moving inspection does not add heavy import-time dependencies
  to the sampling/data-model modules.

The current tests use `unittest`; pytest is not required. Run them in an
appropriate environment with the project's dependencies, for example:

```bash
python -m unittest discover -s tests -p 'test_cross_boundary*.py'
```

## Open decisions

- Final candidate-pool ownership: an explicit shared mapping (preferred) or
  `plan.candidates` with shared snapshot ownership.
- Whether to make a later, separate equality/API change. It is not required
  for the structural refactor and does not block the stages above.
