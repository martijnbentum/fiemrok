Used speech vector search to make word prototypes see repo: [Speech Vector Search (GitHub)](https://github.com/martijnbentum/speech-vector-search)

the prototype lexicons holds 10 tokens per word type. Each prototype token consists of 10 word tokens.
the embedding frames for each word token are aggregated (averaged, other option center frame) to single vector. 
The word token vectors are averaged to a word prototype.

For each word type (e.g. 'dak') there are 10 prototype tokens. 
New word token vectors are generated and matched with the 10 nearest neighbours in the prototype lexicon.
For each word 10 word token vectors are generated and matched below the accuracy and distribution of
matched prototype words.

| word  | acc  | predictions |
|-------|------|-------------|
| dak   | 0.86 | dak (86), vak (6), plak (4), laag (2), buis (1), map (1) |
| trap  | 1.00 | trap (100) |
| leek  | 0.86 | leek (86), leeg (8), beek (6) |
| rol   | 0.96 | rol (96), rook (3), knoop (1) |
| neer  | 0.92 | neer (92), dijk (3), meid (2), leeg (1), pech (1), lies (1) |
| lijn  | 0.85 | lijn (85), lui (4), tij (3), dijk (2), rijp (2), la (1), muis (1), vuil (1), link (1) |
| loop  | 0.96 | loop (96), knoop (2), lam (1), rook (1) |
| trek  | 0.73 | trek (73), vet (4), meid (3), bek (2), tij (2), tip (2), lust (2), prijs (1), la (1) |
| bal   | 1.00 | bal (100) |
| doos  | 0.72 | doos (72), doof (5), muis (4), boos (3), vos (3), beet (3), kip (2), bek (1), knoop (1) |
| munt  | 0.93 | munt (93), nut (3), buis (2), nest (1), knoop (1) |
| la    | 0.37 | la (37), laag (22), lam (13), lui (7), dijk (4), vaas (3), maag (3), tij (3), loop (1) |
| tas   | 0.64 | tas (64), nut (5), lust (4), vuil (3), lam (3), dak (2), bek (2), dijk (2), muis (2) |
| raak  | 0.78 | raak (78), laag (4), la (3), pech (2), vaag (2), vuist (1), muis (1), vet (1), dak (1) |
| kip   | 0.72 | kip (72), tip (8), vee (6), fee (4), lip (3), vet (2), lies (2), nicht (2), pech (1) |
| tij   | 0.63 | tij (63), dijk (13), muis (3), la (3), lijn (2), maag (2), dak (2), meid (2), nest (2) |
| dijk  | 0.67 | dijk (67), la (8), vuil (8), tij (7), muis (3), meid (2), lui (2), nek (1), bek (1) |
| muur  | 0.92 | muur (92), nieuws (2), lust (2), link (1), beet (1), lui (1), lies (1) |
| nek   | 0.80 | nek (80), lui (5), dijk (4), leeg (2), beest (2), bel (2), muur (2), tip (1), trek (1) |
| tel   | 0.64 | tel (64), nicht (6), bel (5), tas (4), lies (4), vuil (2), knoop (2), buis (2), vet (2) |
| pech  | 0.66 | pech (66), pen (7), lies (4), nut (3), vet (3), pet (2), trek (2), bel (2), muis (2) |
| vak   | 0.64 | vak (64), dak (7), pech (5), vet (4), laag (3), pad (2), raak (2), la (2), vaag (2) |
| prijs | 0.66 | prijs (66), meid (6), trek (4), la (3), buis (3), tij (3), beet (2), pech (2), vet (2) |
| laag  | 0.78 | laag (78), maag (9), vaag (7), vaas (2), pech (1), lust (1), lies (1), meid (1) |
| vuil  | 0.52 | vuil (52), buis (8), vet (7), bel (5), nauw (5), lui (3), tij (2), la (2), vaag (2) |
| beek  | 0.83 | beek (83), beet (10), dijk (3), beest (2), bek (1), tij (1) |
| bel   | 0.37 | bel (37), tel (7), vuil (6), nicht (5), rook (5), lies (4), lui (3), pech (3), bek (3) |
| nieuws| 0.51 | nieuws (51), lies (12), muur (7), leek (7), muis (4), vuist (3), vos (2), nicht (2), tas (2) |
| beest | 0.44 | beest (44), beet (8), scheef (6), lies (5), meid (5), prijs (5), beek (4), map (3), pech (2) |
