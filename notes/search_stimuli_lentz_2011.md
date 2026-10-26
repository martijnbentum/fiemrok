# Search-stimuli results in relation to Lentz (2011)

The search-stimuli analysis asks whether a speech model's representation of
a target word changes with its phonotactic context. It connects to Tom
Lentz's human segmentation experiments through the aligned/misaligned
stimulus contrast. The current results measure target-word retrieval at an
annotated vowel location; they do not yet establish boundary detection or
the independence of segmentation from lexical recognition.

## Source and related results

Lentz, T. (2011). Speech segmentation (Chapter 3, pp. 95–149). In
*Phonotactic Illegality and Probability in Speech Perception: Evidence from
second language listeners*. Utrecht University dissertation, LOT 290.

- [Search-stimuli plot and workflow](search_stimuli_plot.md)
- [HuBERT layer 9 results](search_stimuli_hubert_layer_9_cf_nucleus.md)
- [wav2vec 2.0 layer 6 results](search_stimuli_wav2vec_layer6_cf_nucleus.md)

## What connects the stimuli and experiments

In Experiment 3, listeners spot words embedded in bisyllabic nonsense
words. For the final-position target `rok`, the aligned `fiemrok`
encourages a boundary between /m/ and /r/: /mr/ cannot start a Dutch word.
In misaligned `fiedrok`, the legal onset /dr/ supports a competing boundary
before /d/. Splitting /dr/ also leaves a voiced obstruent in the preceding
coda, which is illegal in Dutch. Alignment therefore combines more than
one phonotactic cue. Experiment 4a separates cluster wellformedness from
the legality of the residue left by a segmentation.

The local model analysis uses the 60 real-word targets in final position,
each in aligned and misaligned contexts.
The stimuli were generated with MBROLA-synthesis with controlled durations.

The human task uses eye movements to displayed letters to track possible
word onsets, including alternatives that do not yield words. Experiment 3
also includes initial-position targets and non-word conditions; the current
retrieval analysis selects neither of these. Experiments 4a and 4b test
additional cue contrasts and listener-language differences that the current
model comparison does not reproduce.

## What the model results show

Both query and CGN reference vectors use a single centre frame of the vowel
nucleus. Each stimulus retrieves ten CGN word tokens; precision at 10 is
the fraction carrying the intended target label. Scores below are means
over 60 stimuli per condition, calculated from `../all_results.json`.
The two detailed result notes match those stored scores in order.

| Model and layer | Aligned | Misaligned | Aligned minus misaligned |
| --- | ---: | ---: | ---: |
| HuBERT layer 9 | 44.17% | 34.50% | +9.67 percentage points |
| wav2vec 2.0 layer 6 | 17.00% | 17.67% | −0.67 percentage points |

HuBERT layer 9 shows an aligned retrieval advantage in the direction of
the human word-spotting result. Wav2vec layer 6 shows little separation,
with a small difference in the opposite direction. These are descriptive
comparisons, without a paired significance test. The plot includes other
layers; neither pattern should be generalised to every layer of a model.

Human final-position word-spotting accuracy in Experiment 3 was 79.7%
aligned versus 43.7% misaligned (Table 3.2, printed p. 108;
[chapter PDF page 14](/Users/martijn.bentum/Documents/papers/word_segmentation/Lentz_2011.pdf#page=14)).
These percentages describe a different task and cannot be directly compared
with retrieval precision. The correspondence is in the direction of the
condition difference, rather than its numerical size.

## Interpretation and limits

The chapter's central evidence is that phonotactics affects segmentation
attempts even without successful recognition of a real embedded word.
Word-only retrieval cannot test that claim: it asks how strongly a known
target is represented at a location supplied by annotations. An alignment
effect could also reflect acoustic context, coarticulation, or phonetic
similarity to the CGN reference tokens.

Centre-frame extraction avoids averaging the whole annotated word, but
selecting the nucleus still supplies its location. A transformer state at
that frame can also incorporate surrounding and later input; its timestamp
does not establish when a listener or model could recognise the word online.
The chapter likewise cautions that its data do not directly establish that
phonotactic cues operate before lexical influence (printed pp. 142–143).

The plotted confidence bands describe condition means separately. A claim
about alignment should instead assess the paired differences for the same
targets, retaining target-level variation. Layer selection and the mismatch
between synthetic stimuli and natural CGN speech also matter for interpretation.

## How to bring the model analysis closer to the chapter

- Evaluate non-word stimuli and competing onset hypotheses, so an effect
  does not require recovering a real target label.
- Follow target and competitor evidence across successive frames, and test
  predicted boundaries against annotations used only for evaluation. For
  claims about timing, control how much future audio the model can access.
- Separate cluster wellformedness and residue legality, following Experiment
  4a, to identify which cue drives a retrieval or boundary effect.
- Consider the [prototype approach](filler_prototypes.md) for stable reference
  vectors across spoken occurrences. Prototypes could use centre frames
  and be queried over time; keep evaluation recordings separate from their
  source recordings. The existing prototype results have a possible overlap
  between query tokens and prototype source tokens.

## Code references

- [search_stimuli_embeddings.py](../search_stimuli_embeddings.py):
  `make_all_compare_embeddings()` runs models/layers;
  `CompareEmbeddings._set_aligned_and_misaligned()` selects final-position
  word targets; `load_center_frame_nucleus_speech_vector()` extracts queries;
  `compute_accuracy()` calculates retrieval precision.
- [cgn_lexicon_vector_search.py](../cgn_lexicon_vector_search.py):
  `metadata_to_center_frame_nucleus_speech_vector()` extracts CGN vectors;
  `make_vector_search_index_for_layer()` builds the reference index.
- [plot_result_dict.py](../plot_result_dict.py): `plot_accuracy()` plots
  condition means and `_mean_ci()` calculates their confidence intervals.
