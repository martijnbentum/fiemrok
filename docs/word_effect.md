# Word identity and the word effect

This note defines what the lexicon validation measures as a word effect and
how it is estimated. It belongs to `lexicon_validation` only. Open points
are listed at the end and will be settled one by one.

## Word identity

**Word.** A word is a lexical entry, studied through its phonological form:
a recurring string of phones, realised with variation. Semantics is not the
focus.

**Operationalisation.** A word type is an orthographic label, and all its
sampled tokens belong to the type. Tokens are not filtered or regrouped by
pronunciation, so a type holds all realisations of the word, variants
included. The orthographic label is a proxy for the lexical entry:

- Homographs are negligible. Labels with clearly different phone strings
  turned out to be pronunciation variants (voicing, schwa or -n deletion,
  reduction, assimilation).
- Homophones cover about 6% of the word types (361 of 5,675 labels share
  their most common phone string with another label), mostly spelling-only
  differences (*word/wordt*, *bedoeld/bedoelt*) and inflections merged by
  n-deletion (*alle/allen*, *bijdrage/bijdragen*). The most common phone
  string per label is only used to detect them. Homophones are not merged;
  they enter the model as a separate factor.

**Control.** A cross-boundary item is a phone string spanning the boundary
between two adjacent words. Its type is the exact phone string, and the
sample matches the words' phone-count distribution. These strings recur in
speech, but not as lexical units, and they carry the processes of a word
boundary, such as final lengthening, assimilation across the boundary and
resyllabification.

## Word effect

**Definition.** The word effect is the extra similarity between two tokens
of the same word, beyond that between two tokens of the same cross-boundary
phone string, with other factors held constant by modelling rather than by
matching.

**Estimate.** One pair model of cosine similarity over both datasets, with
pairs sampled on predictors only (`lexicon_validation/factor_models.py`):

```
similarity ~ same_type × is_word
           + same_type × same_speaker
           + same_type × shared_left_word + same_type × shared_right_word   (cb)
           + same_form_different_label                                       (words)
           + phone_edit_distance + log_duration_ratio + mean_n_phones
           + same_recording + same_component + same_gender + age_difference
           + rate_difference + shared_previous_word + shared_next_word
           + frequency_difference
           (× is_word for duration, length and phone distance)
```

- `same_type × is_word` is the word effect.
- `same_type` is phone-string identity: the cross-boundary baseline.
- `same_type × shared_left/right_word` measures how much of cross-boundary
  similarity is carried by the words a string is part of. Shared words are
  modelled, not excluded, to keep the control set large. A possible
  refinement is the share of the string's phones that come from a shared
  word.
- `same_form_different_label` is the homophone effect.
- `same_type × same_speaker` shows whether identity holds across speakers.

Confidence intervals come from resampling query types.

**Interpretation.**

- Word pairs include pronunciation variants, cross-boundary pairs are
  phone-identical by construction. With phone edit distance as a
  covariate, the estimate is conservative.
- HuBERT has no stored lexicon. A positive effect means word units are
  represented more consistently than recurring strings that cross a word
  boundary.

## Results so far

Per-dataset pair models (not yet the joint model), HuBERT layer 9:

| | words, mean | words, center | cb, mean | cb, center |
|---|---|---|---|---|
| same type | +0.192 | +0.213 | +0.135 | +0.142 |
| homophone | +0.173 | +0.192 | n/a | n/a |

Homophones get about 90% of the same-word effect, so the word effect is
mainly a form effect. Same-word pairs with identical phones remain about
0.10 (mean) and 0.06 (center pooling) more similar than homophone pairs,
a smaller label-specific part (possibly residual acoustic traces of deleted
segments, or different morphosyntactic contexts).

## Open points

1. Boundary cues: part of the effect, or a confound to control?
2. Frequency: mediator of the effect or nuisance covariate? Needs counts
   that are comparable for words and cross-boundary strings.
3. Dataset-specific slopes for duration, length and phone distance only,
   with a fully interacted model as a robustness check?
4. Word and cross-boundary tokens with the same phone string: an extra
   stratum in the joint model, or a separate contrast?
5. Mean pooling as the main analysis and center pooling as a check?
