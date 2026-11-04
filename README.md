# fiemrok

This repository lives in a larger `fiemrok` workspace. Code, tests,
documentation, and small versioned resources stay in `repo/`. Inputs,
embedding stores, generated artifacts, and results live alongside it.
See [Workspace layout and migration](docs/workspace_layout.md).

## Setup

Keep the existing workspace environment if available. For a new setup,
run from `repo/`:

```bash
python3.12 -m venv ../fiemrok_env
source ../fiemrok_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code also calls `sox` via the `sox --i` command to inspect audio files, so
that system dependency needs to be available on your machine as well.

If you instantiate `fiemrok.Experiment()` without arguments, it will load the
spreadsheet and build or load `audio_info.json` from the surrounding workspace.
For isolated tests or callers that already have trial rows in memory, pass
`header=...`, `data=...`, and optionally `audio_info_dict=...` to avoid that
workspace-dependent setup step.

## Module layout

Experimental materials, the experimental CGN lexicon, and stimulus comparisons
live in `fiemrok_experiment/`. Word and cross-boundary lexicon construction
and evaluation live together in `lexicon_validation/`. Legacy phone embedding
and codebook modules live in `phone_analysis/`.

```python
from fiemrok_experiment import fiemrok
from fiemrok_experiment import select_lexicon
from lexicon_validation import vector_indices
from lexicon_validation import cross_boundary_lexicon
from phone_analysis import phone_model_outputs, codebooks
```

The previous top-level imports are no longer supported. Shared utilities
(`locations`, `stores`, `lexicon_helper`, `model_paths`, `audio`, and
`annotation`) remain at the repository root. Scripts and tests remain in
`scripts/` and `tests/`. Run from the repository root, or put that directory
on Python's import path.

Default data and output paths are anchored to the repository or its parent
workspace through `locations.py` and validation-specific `lv_locations.py`;
moving modules does not move stores or
results. Explicitly supplied relative paths remain relative to the working
directory. Previously pickled custom objects may require migration before
loading under the new module names.

## CGN lexicon evaluation

Use the [lexicon validation pipeline](docs/lexicon_validation_pipeline.md)
to select words and cross-boundary items, extract both models, and build
center/mean indexes. The experiments have independent selection rules and
outputs. The new validation pipeline rebuilds generated data from saved
selection keys; retrieval outputs and metrics are being redesigned separately.

## Cross-boundary lexicon sampling

See [Adaptive sampling](docs/cross_boundary_sampling.md) for planning quotas,
retrying with consecutive seeds, relaxing constraints, and resuming failures.
See [Marker storage](docs/cross_boundary_sampled_tokens_markers.md) for saving
and loading the sample, and [Marker embeddings](docs/cross_boundary_embeddings.md)
for full-frame wav2vec2 and HuBERT computation.
The [previous neighbour workflow](docs/cross_boundary_neighbours.md) documents
the historical sample and filenames.
