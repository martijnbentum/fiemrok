# Workspace layout

Research workflows come first. `locations.py` defines defaults anchored to the
parent of `repo/`; explicitly supplied relative paths retain their usual
meaning relative to the caller's working directory.

```text
fiemrok/
  repo/                              code, tests, docs, notes, versioned data/
    fiemrok_experiment/               experiment preparation and retrieval
    lexicon_validation/              lexicon construction and evaluation
    phone_analysis/                  legacy phone and codebook modules
  fiemrok_env/                        existing environment, unchanged
  fiemrok_experiment/
    inputs/                          spreadsheet, audio, annotations
    stores/                          Phraser and Echoframe experiment stores
    svs_fiemrok_experiment_data/      saved vector bundle
    audio_info.json
    results/                         retrieval summaries and plots
  lexicon_validation/
    cgn_befgijklno_nl/                local Phraser recordings and annotations
    word_keys.pkl                    rebuilt validation word selection
    cross_boundary_marker_keys.pkl   rebuilt cross-boundary selection
    echoframe_stores/                 {model}_{condition}_{n_types}
    stores/                          legacy stores from the migration
    samples/                         legacy selected and filtered word keys
    frame_analysis/                  frame counts and derived analyses
    svs_data/                        saved speech-vector indexes
    results/                         comparison summaries and plots
  sonority/
    stores/
    results/
  phone_analysis/
    stores/                          phone embeddings
    metadata/                        phone counts and derived metadata
    codebooks/                       codebook arrays, labels, PNMI outputs
  maintenance/
    config/                          inventories and migration backups
    restructure_moves.jsonl          original migration history
    workflow_moves.jsonl             workflow migration history
```

Only directories containing existing files are created. Small versioned
resources stay in `repo/data/`; publication figures stay with their reports in
`repo/notes/assets/`. Shared CGN data and model checkpoints stay outside this
workspace.

The [validation rebuild](build_speech_vector_index_pipeline.md) uses
`lexicon_validation/lv_locations.py`, the two top-level selection PKLs,
and `echoframe_stores/`. Old migrated stores and results are not moved or
deleted by the new pipeline. `fiemrok_experiment` remains separate.

The experiment's `echoframe_cgn_lexicon_store` belongs to
`fiemrok_experiment/stores/`. The `lexicon100_*` and cross-boundary stores
belong to `lexicon_validation/stores/`; existing dataset names are preserved.

Vector arrays, metadata, scores, and neighbours remain together in their saved
bundles because the current APIs expect one directory. The experiment bundle
is named `svs_fiemrok_experiment_data`; lexicon validation retains `svs_data`.

`repo/phone_analysis/` groups the two leaf modules `phone_model_outputs.py` and
`codebooks.py`. They still use shared utilities from the repository root.
Their older model APIs are unchanged; moving the code does not modernize it.

## Migration from the previous layout

Close stores and saved indexes being moved. The external shared CGN store can
remain open. From `repo/`, inspect the exact mapping before applying:

```bash
python3 scripts/restructure_workspace.py --workspace /path/to/fiemrok
python3 scripts/restructure_workspace.py --workspace /path/to/fiemrok --apply
```

This migrates the previous type-first `inputs/`, `stores/`, `artifacts/`, and
`results/` layout. It refuses destination conflicts and cross-filesystem moves.
Existing data directories are renamed, without copying their contents. Missing
sources are skipped. Empty containers from the old layout are removed; any
unexpected files are left in place. The script does not migrate older flat
layouts directly.

The migration journals intent before each rename in
`maintenance/workflow_moves.jsonl`. After interruption, inspect the source and
destination paths; rerunning resumes remaining moves and path updates.
Echoframe registrations are updated, preserving original configs beside them.
Saved-index metadata save destinations are updated without changing vector
arrays or metadata rows; original metadata remains in each bundle's
`metadata_before_restructure/` folder.

Root-level `tom-eye` and `stimuli_textgrids` symlinks point into
`fiemrok_experiment/inputs/` to preserve filenames persisted in stores.
The existing environment stays at `fiemrok_env/`; activate it from `repo/` with
`source ../fiemrok_env/bin/activate`. Reload modules or restart an interactive
session before using the new defaults.

Only remote data was migrated. Other checkouts need their own workspace
migration; older layouts may require a separate mapping.
