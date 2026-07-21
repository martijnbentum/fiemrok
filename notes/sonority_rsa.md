selected phrases from cgn comp-k nl
selected first 30 phrases from each speaker in comp-k nl (approximately 1K phrases)

used https://github.com/martijnbentum/sonority_rsa to compute rsa values
sampled 1000 syllables 12 times from approximately 10K syllable pool (1000 syllables in subset are unique, subsets can overlap)

computed sonority values based on phone labels with https://github.com/martijnbentum/dutch-syllabifier (used in sonority_rsa)
sampling at the syllable level ensures balanced sampling from all syllable positions

Applied dutch Wav2vec 2 (200k checkpoint) and dutch HuBERT (200k checkpoint)
Extract the hidden state at later 1, 3, 6, 9 & 12 at the center frame of each phone.

A random baseline is computed based on shuffled sonority values
A intensity baseline is computed with sonority_rsa.intensity compute_intensity(signal) based on the phone interval (established by Webmaus FA)

A partial RSA is computed for both intensity and sonority.

<img width="1190" height="547" alt="Screenshot 2026-07-21 at 15 24 07" src="https://github.com/user-attachments/assets/7faee85d-d899-4fd7-aa83-fa6d93831c68" />



Partial RSA measures whether a model's representational geometry reflects one explanatory factor (e.g., sonority) after removing the variation it shares with another factor (e.g., intensity). It does this by regressing the control variable out of both the model RDM and the explanatory RDM and then correlating the resulting residuals. The resulting partial correlation indicates whether the model still aligns with the unique structure of sonority beyond what can be explained by intensity. Positive values indicate that the unique sonority structure is preserved in the model, values near zero indicate little unique relationship, and negative values indicate that the model is organized in opposition to the residual structure of the explanatory variable. Because the residual represents only the component not shared with the control variable, partial RSA quantifies unique association rather than explained variance or causal influence.


Old plot
<img width="700" height="500" alt="sonority_rsa_w2v2_hubert_compk" src="https://github.com/user-attachments/assets/26e16d14-5ded-4065-833a-26db8e55bb0c" />

