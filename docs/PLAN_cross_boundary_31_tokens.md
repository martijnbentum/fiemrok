# Cross-boundary selection at 31 occurrences per type

Use the existing `cgn_abefghijklno_nl` material store and saved
`word_lexicon.pkl`. Keep the full word inventory unchanged.

## Candidate eligibility

- Cross exactly one boundary between consecutive words within a phrase.
- Keep partial words, excluding the left word's first phone and the right
  word's last phone as in the existing extractor.
- Require every included phone's duration assessment to be `within_range`.
- Keep inclusive 100-1500 ms candidate durations.
- Exclude positive time overlap with another speaker's non-silence words.
  Same-speaker speech, explicit silence and touching boundaries do not count.
- Replace the previous per-type 5 * IQR filter; do not recompute durations.
- Report sequential token and type losses before frequency eligibility.

## Sampling and outputs

- Reuse the saved Word Lexicon through `prepare_samples(word_inventory=...)`.
- Sample 31 word tokens per type using seed 0: 5,675 types, 175,925 tokens.
- Target the same number of cross-boundary types and match the sampled
  words' modal phone-count distribution.
- Permit reducing to 5,108 types only when the current sampler needs to.
  Never reduce the 31-occurrence quota to obtain more types.
- Selected cross-boundary occurrences cannot share source phones, including
  across different types. Candidate inventories can contain alternatives.
- Keep current adaptive search settings and retain the best successful
  result. Report shortages and distribution distortion if infeasible.
- Save ordered word keys and cross-boundary marker keys at workspace root.
  Create markers once after successful sampling; do not start embeddings.

## Verification

- Test short, long and unset constituent-phone rejection.
- Test span-level speech exclusion, silence and touching boundaries.
- Test that whole-word overlap outside the candidate does not exclude it.
- Test that IQR is not reapplied and absolute duration bounds are inclusive.
- Retain the existing sampling tests for quota, shared-phone exclusion and
  adaptive type-count matching.
- Test saved word inventory reuse and reject non-Word inventories.
- Verify the completed remote sample's counts, source phones, durations,
  simultaneous-speech exclusion and saved-key/marker round trips.
