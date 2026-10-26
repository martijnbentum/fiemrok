selected 2466 word types from cgn with minimally 100 word token (excluded components a,c,d,h,m,
excluded words shorter than .1 second and longer than 1.5 seconds)

This plot summarises the same CGN lexicon as [cgn_lexicon.md](cgn_lexicon.md),
evaluating all 100 tokens per word type rather than one token per word type.
Word embeddings used the centre frame of each word's embedding frames,
as implemented by `load_speech_vectors()` in [validate_lexicon.py](../validate_lexicon.py).
The exact generating run was not recorded, so the centre-frame method is inferred from the code.
The per-token and per-word precision calculations are in [analyse_results.py](../analyse_results.py).

search is based on each item in the lexicon (100 items per word type, requested 10 nearest neighbours) \
per word score is average precision at 10 (average over 100 items) \
bar plot shows average precision over all word types and 95 % CI

<img width="519" height="379" alt="Screenshot 2026-05-05 at 10 12 10" src="https://github.com/user-attachments/assets/025b5358-3c1e-4337-a0e7-4b839ef8b80b" />
