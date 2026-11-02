# Word-specific information in HuBERT layer 9

Research note and proposed first analysis, 30 October 2026.
Background: [neighbour retrieval overview](neighbour_results_overview.md) and
[CGN retrieval for word-aligned and across-word segments](cgn_lexicon_plot.md).

## Summary

- **Question.** Does HuBERT layer 9 group speech around whole words, so that
  tokens of the same word cluster more consistently than tokens of the same
  phoneme sequence spanning a word boundary?
- **Current evidence.** With mean pooling, P@10 is 81.09% for word-aligned
  and 56.56% for across-word segments. The comparison is descriptive: the
  inventories, labels, and token samples differ.
- **Next step.** Check how many neighbours share the query's speaker or
  recording, using the existing top-30 neighbour files and the metadata
  lookup in `neighbour_metadata.py`.
- **Status.** The metadata lookup is implemented and cached per dataset.
  The remote data have not been enriched or reanalysed for this proposal.

## Hypotheses

Three claims need progressively stronger evidence:

1. **Accessible word identity.** Same-word tokens are each other's nearest
   neighbours across speakers and recordings.
2. **Word-level organisation.** Grouping follows whole words rather than
   the selected phone span alone.
3. **A separable word subspace.** A reproducible linear subspace carries
   the word signal. Word-level organisation does not require this.

Each prediction below has a result that would count against it:

| Prediction | Disconfirming result |
|---|---|
| The word advantage holds when neighbours must come from another speaker and recording | The advantage disappears under that restriction |
| The word advantage holds within strata of matched span length and phone count | The advantage disappears within those strata |
| Whole words retrieve better than word-internal spans that are not whole words | Word-internal spans retrieve as well as whole words |
| Near-homophone word pairs are separated above chance | Near-homophones are confused at chance level |

If the word-internal spans match whole words, the advantage comes from not
crossing a boundary rather than from aligning with a whole word.

## Controls

- **Nuisance variables: speaker and recording.** Same-speaker or
  same-recording neighbours can inflate retrieval without word-level
  organisation. Control for them by restricted retrieval, and report results
  with and without the restriction.
- **Possible carriers of word identity: phonetics, stress, and timing.**
  Word identity is realised through these cues. They are described and
  reported, not regressed out or matched away. Removing them could remove the
  phenomenon under study.
- **Design differences: span length and inventory composition.** Span
  duration, phone count, and the number of near-identical competitor types
  differ between the datasets because of how segments were selected. Match
  or stratify on these.

## First-stage analyses

All of these use the existing top-30 neighbour files plus metadata. None
needs a new export.

1. **Speaker and recording composition.** For each query, count neighbours
   from the same speaker and the same recording, and the availability of
   same-type tokens from those sources in the index. Compare the datasets.
2. **Near-homophones.** Word pairs such as *verandert*/*veranderd* (final
   devoicing) and *-n* inflection pairs (*maakte*/*maakten*) are
   pronounced the same or nearly so. Measure whether they are separated above
   chance, and whether sentence context explains any separation.
3. **Competitor density.** For each inventory, count types within one phone
   edit of another type, such as nested across-word sequences
   (`t x r oː` / `t x r oː t`) or inflection pairs.
4. **Span-matched comparison.** Compare the word advantage within strata of
   span duration and phone count, and report its distribution across types.

## Later analyses

5. **Word-internal spans.** Add a condition with phone sequences inside a
   word that are not whole words, matched in phone count to the across-word
   sequences. This needs a new index.
6. **Contributing-word structure.** Within a fixed across-word label such
   as `/n # ə/`, compare similarity between tokens that share both
   contributing words, only the left, only the right, or neither, across
   speakers and recordings. `#` fixes the boundary position. Shared
   contributing words also share the phones of those words outside the
   span; both belong to the containing word. This test cannot separate them
   and does not need to. It can show that an across-word label pools tokens
   whose embeddings reflect different containing words, which would explain
   lower across-word retrieval. A null result weakens this explanation only.
7. **Top-100 export.** Export the top 100 non-self neighbours for the four
   HuBERT layer 9 conditions (two datasets, two pooling methods). Each query
   has 99 same-label tokens, so this allows R-precision (P@99). Store
   enriched occurrence records in a companion table referenced from
   neighbour rows, not repeated on every edge.
8. **Temporal distribution.** Compare centre frames, quarter means,
   whole-span means, and expanded windows, with frame count controlled.
9. **Subspace interventions.** Estimate word, phone, and speaker centroid
   directions on training speakers. Evaluate retrieval inside and after
   removing a candidate subspace on held-out speakers, against random and
   variance-matched controls.
10. **Context and boundary interventions.** Shift pooling boundaries within
    an existing encoding, and separately re-encode with altered context.
    Compare boundary-crossing shifts with matched within-word shifts.

## Design constraints

- **Top 100 is not all neighbours.** Some same-type tokens can rank below
  100. Export their similarities separately if complete same-label coverage
  is needed.
- **Restricted retrieval needs re-querying.** Removing same-speaker or
  same-recording neighbours can exhaust a fixed list. Re-query under the
  restriction for valid restricted P@k.
- **Use sampled pairs for geometry.** Neighbour lists are selected
  high-similarity pairs. Geometric analyses need deliberately sampled
  pairs, including matched negatives.
- **Keep vector access explicit.** Saved search vectors are L2-normalised.
  Link the original Echoframe embeddings when magnitude or frame-level
  variation matters; do not duplicate vectors in neighbour records.
- **Check identifiability first.** Some across-word labels may occur with
  only one contributing word pair or one speaker. Such groups cannot
  estimate word-pair or speaker effects.
- **Never drop rows silently.** All metadata joins must succeed before
  distribution analysis; report failed joins and missing values explicitly.

## Metadata lookup

`neighbour_metadata.py` resolves Echoframe source keys to Phraser segments
and builds deduplicated dictionaries for keys, speakers, phones, and words,
with configurable phone and word context. `query_distributions` yields each
query's speaker and recording counts from an existing neighbour file. See
the module docstrings for details. The functions have synthetic tests; joins
on the remote data still need a sample audit.

## Paper background

These papers motivate tools and controls rather than define a required
result. Local summaries and relevant full-text passages were inspected.

- [Sanabria et al. (2023)](../../../overview/Papers/summaries/@sanabria_2023.md):
  pooled contextual word embeddings; neighbouring-word clustering motivates
  explicit context controls rather than assuming context-free lexical vectors.
- [Pasad et al. (2024)](../../../overview/Papers/summaries/@pasad_2024.md):
  compare frame and quarter pooling; distinguish encoded information from its
  accessibility through cosine retrieval.
- [Liu et al. (2023)](../../../overview/Papers/summaries/@liu_orthogonal_subspaces.md):
  centroid PCA and subspace removal provide a practical intervention template.
- [Mohamed et al. (2024)](../../../overview/Papers/summaries/@mohamed_orthogonality_isotropy.md):
  CRV provides variance-weighted subspace comparisons; geometry alone does not
  establish what linguistic information a subspace contains.
