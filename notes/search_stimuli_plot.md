See [the connection to Lentz (2011)](search_stimuli_lentz_2011.md) for the
human experiments, interpretation of these results, and segmentation limits.

This plot summarises word retrieval for aligned and misaligned stimuli across
wav2vec 2.0 and HuBERT layers. The stimulus-level results for
[wav2vec layer 6](search_stimuli_wav2vec_layer6_cf_nucleus.md) and
[HuBERT layer 9](search_stimuli_hubert_layer_9_cf_nucleus.md) match the corresponding
scores in `../all_results.json`, which supplies the plot data.

The queries come from Tom's experimental audio stimuli. This analysis uses
real-word targets in final position, with aligned and misaligned versions of
each target in a preceding context. The stimulus annotations identify the
target vowel nucleus, whose centre frame supplies the query embedding.
For example, `fiemrok` and `fiedrok` are the two contexts for the target `rok`.
The model searches the CGN lexicon for neighbours of that embedding, and the
score measures how many of the ten retrieved tokens carry the target label.
The tables show individual stimulus results; the plot averages these scores
by model, layer, and alignment condition. This tests how recoverable the
target word is in each stimulus context, using the annotated nucleus location.

Embeddings use the centre frame of the target vowel nucleus. Relevant modules and functions:

- [search_stimuli_embeddings.py: make_all_compare_embeddings()](../search_stimuli_embeddings.py#L11)
  computes results across models and layers; [CompareEmbeddings.compute_results()](../search_stimuli_embeddings.py#L60)
  evaluates the aligned and misaligned stimuli.
- [load_center_frame_nucleus_speech_vector()](../search_stimuli_embeddings.py#L171)
  extracts the query vector; [compute_accuracy()](../search_stimuli_embeddings.py#L103)
  calculates precision at 10 as the fraction of retrieved labels matching the target.
- [cgn_lexicon_vector_search.py: metadata_to_center_frame_nucleus_speech_vector()](../cgn_lexicon_vector_search.py#L20)
  extracts nucleus centre-frame vectors for CGN words;
  [make_vector_search_index_for_layer()](../cgn_lexicon_vector_search.py#L77) builds the search index.
- [plot_result_dict.py: plot_accuracy()](../plot_result_dict.py#L50)
  reads `../all_results.json` and plots the condition means across layers;
  [_mean_ci()](../plot_result_dict.py#L24) calculates the 95% confidence intervals.

- Created a lexicon from the Spoken Dutch Corpus.
  - Excluded components c, d, and m (telephone conversations and sermons).
  - Excluded overlapping speech.
  - Excluded Flemish recordings.
  - Excluded words with frequencies outside the range 100–1500.
  - Included only monosyllabic words.
  - Included only words with a frequency of at least 100.
  - Resulting lexicon contains approximately 800 word types.

- Added each word from the target stimuli using the same restrictions, except for the frequency criterion.

  <img width="1392" height="496" alt="Screenshot 2026-06-29 at 14 31 53" src="https://github.com/user-attachments/assets/f82383af-2458-4a9f-bc09-a0ddbec3cac9" />



  

  

  

  
