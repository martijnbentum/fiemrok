# Cross-boundary embedding refactor plan

Agreed design: keep setup, store opening, and computation explicit and small.
Keep public functions toward the top of their modules.

## Paths and store loading

- Define the two EchoFrame store paths directly under the repo root in
  `locations.py`, plus `svs_data/cross_boundary_marker_keys.pkl`.
- Add `stores.load_echoframe_cross_boundary_store(path=...)` to open the
  selected store and attach the shared CGN store.
- Use the marker-key path constant in the CBL save/load defaults.
- Test the default paths and CGN attachment.

## Setup

- `create_stores()` opens the wav2vec and HuBERT stores and returns both.
- Register version 1/checkpoint 200000 from `model_paths.py`, one model per
  store; skip registration if already present so setup is rerunnable.
- Remove the word-store configuration dependency and module settings dict.
- Test checkpoint lookup, model registrations, and repeated setup.

## Computation

- `compute_embeddings(embedding_store, ...)` accepts an open, configured store.
- Obtain CGN through that store, load the saved markers, infer the sole
  registered model, and call EchoFrame's batch function.
- Preserve full-frame extraction, layers 3/6/9/12, 500 ms collar per side,
  batch controls, and resume behavior. A 2000 ms collar remains configurable.
- Let the caller own the store's lifetime; omit redundant input branches.
- Keep real-store integration tests for exact frame matrices, marker links,
  reload, missing layers, and skipped inference on completed requests.
- Update examples to use explicit setup and open stores.

All design choices above were reviewed in the interview; no open questions.
