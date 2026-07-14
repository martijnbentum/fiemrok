selected phrases from cgn comp-k nl
selected first 30 phrases from each speaker in comp-k nl (approximately 1K phrases)

used https://github.com/martijnbentum/sonority_rsa to compute rsa values
sampled 1000 syllables 12 times from approximately 10K syllable pool (1000 syllables in subset are unique, subsets can overlap)

computed sonority values based on phone labels with https://github.com/martijnbentum/dutch-syllabifier (used in sonority_rsa)
sampling at the syllable level ensures balanced sampling from all syllable positions

Applied dutch Wav2vec 2 (200k checkpoint) and dutch HuBERT (200k checkpoint)
Extract the hidden state at later 1, 3, 6, 9 & 12 at the center frame of each phone.

<img width="1609" height="978" alt="sonority_rsa_w2v2_hubert_compk" src="https://github.com/user-attachments/assets/c3274c14-457b-4d2a-998b-9a71ed40561a" />


