# Word identity and the word effect

This note defines what the lexicon validation measures as a word effect and
how it is estimated. It belongs to `lexicon_validation` only.

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
similarity ~ is_word
           + same_type
           + same_type × is_word                        # word effect
           + same_type × same_speaker
           + same_speaker + same_recording
           + same_component + same_gender               # speaker and channel
           + shared_previous_word + shared_next_word    # context
           + cb_same_left_word + cb_same_right_word     # cb only, 0 for words
           + same_form_different_label                  # words only
           + other_type_distance                        # different types only
```

| term | meaning |
|---|---|
| `is_word` | baseline similarity difference between the datasets |
| `same_type` | gain for the same phone string (cross-boundary), across speakers |
| `same_type × is_word` | **the word effect**: extra gain when the string is a word |
| `same_type × same_speaker` | extra gain when a same-type pair also shares the speaker |
| speaker and channel | nuisance, one slope for both datasets |
| context | shared word just before or after the item; for cross-boundary items the words outside its two constituent words |
| `cb_same_left_word`, `cb_same_right_word` | gain when two cross-boundary tokens share a constituent word: the word effect inside the control |
| `same_form_different_label` | homophones: same most common phone string, different label |
| `other_type_distance` | phonetic distance of a different-type pair, 0 for same-type pairs |

`other_type_distance` is the phone Levenshtein distance divided by the
longer item's phone count, centred on the mean of random different-type
pairs with one centre for both datasets, and 0 for same-type pairs. It makes
the baseline a typical unrelated item, so oversampled near neighbours do not
shift it. Same-type pairs are not adjusted, so pronunciation variants count
against words, as the definition requires. The scale is linear: similarity
falls roughly linearly with normalised distance (random pairs from about
0.63 near 0.2 to 0.43 near 1.0, mean pooling). Normalising by length also
removes the datasets' difference in how close unrelated items are: random
pairs average 0.900 (words) and 0.907 (cross-boundary), against 6.4 and 5.8
raw phones.

The model is implemented in `lexicon_validation/word_effect.py`
(`run_word_effect_model`); summaries are saved as
`factor_models/word_effect_{model}_layer{layer}_{pooling}.json`.

**Pairs.** All strata except close phones: same type, same type with same
speaker, different type, different type with same speaker, and homophones
(words). Every oversampled feature has a model term, so the stratum sizes do
not bias the estimates.

**Dropped.** Age, duration, length, rate and the token-level phone distance
covariate.

Confidence intervals come from resampling query types. Covariates have one
slope for both datasets.

**Checks.**

- Every term with its own slope per dataset (`× is_word`); a shift of the
  word effect beyond its confidence interval is reported.
- `same_type × same_speaker × is_word`.
- Phone distance for all pairs, which estimates the word effect at
  identical phones; the difference with the main model is the cost of
  pronunciation variation.
- Boundary covariates (see below).
- Center pooling.
- The word effect by phone count.

The main model has no frequency terms. Recurrence as a unit may be the
mechanism behind the effect, so frequency is treated as a mediator: a
second model will add string frequency (all occurrences of the phone
string in running speech, within or across words, counted the same way
for both datasets), unit frequency and their interactions with
`same_type`, and report how much the word effect changes.

**Interpretation.**

- Word pairs include pronunciation variants, cross-boundary pairs are
  phone-identical by construction. Same-type pairs are not adjusted for
  phone distance, so the estimate is conservative.
- HuBERT has no stored lexicon. A positive effect means word units are
  represented more consistently than recurring strings that cross a word
  boundary.
- Boundary cues are part of the effect. A word is a unit without an
  internal boundary, and the acoustic realisation of a boundary (final
  lengthening, pauses, assimilation, coarticulation) is part of what makes
  a string not a word. The main model therefore has no boundary covariates.
  A sensitivity model may add them (relative duration of the boundary
  phones, a pause flag) to show how much of the effect runs through
  boundary acoustics.

## Results so far

Joint model, HuBERT layer 9 (cosine units, 95% type-bootstrap CI):

| term | mean pooling | center pooling |
|---|---|---|
| `same_type` (cross-boundary, no shared word) | +0.204 [0.201, 0.206] | +0.251 [0.246, 0.255] |
| `same_type_x_is_word` (**word effect**) | +0.043 [0.039, 0.046] | +0.037 [0.032, 0.043] |
| same-type gain for words | +0.246 | +0.288 |
| `same_type_x_same_speaker` | +0.062 | +0.089 |
| `cb_same_left_word` | +0.038 | +0.035 |
| `cb_same_right_word` | +0.056 | +0.033 |
| `same_form_different_label` | +0.045 | +0.173 |
| `other_type_distance` (per unit) | -0.163 | -0.112 |
| R2 | 0.65 | 0.42 |

The homophone term is now the gain beyond what phone closeness predicts:
homophones have normalised distance near 0, so the distance term already
adds about 0.15 (mean pooling) relative to a typical unrelated item.

Earlier per-dataset models:


| | words, mean | words, center | cb, mean | cb, center |
|---|---|---|---|---|
| same type | +0.192 | +0.213 | +0.135 | +0.142 |
| homophone | +0.173 | +0.192 | n/a | n/a |

Homophones get about 90% of the same-word effect, so the word effect is
mainly a form effect. Same-word pairs with identical phones remain about
0.10 (mean) and 0.06 (center pooling) more similar than homophone pairs,
a smaller label-specific part (possibly residual acoustic traces of deleted
segments, or different morphosyntactic contexts).

## Decisions

- Mean pooling is the main analysis; center pooling is a check, since a
  cross-boundary item's centre frame can sit on the word boundary.
- Cross-boundary strings identical to a word form are an edge case and get
  no model term: 49 cross-boundary types (0.9%) equal a sampled word type's
  most common form, 100 (1.9%) equal some sampled word token, and 261
  (4.8%) occur as a whole word somewhere in the store. They are mostly 2-3
  phones (*maar*, *naam*, *kop*, *moe*).
- Short cross-boundary strings often occur inside words
  (`frequencies/cross_boundary_string_counts.pkl`): 66% of the occurrences
  of 2-phone and 48% of 3-phone types lie within one word. For short items
  the control is therefore partly a familiar word part, which makes the word
  effect conservative there. This is not modelled; a check reports the word
  effect by phone count.
