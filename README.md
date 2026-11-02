# fiemrok

This repository is not currently a standalone Python package. It is intended
to be used inside the larger local `fiemrok` data workspace that contains this
`repo/` directory and the sibling data files and folders the code reads
directly, including:

- `../trialsNewTryWithReps.ods`
- `../audio_info.json`
- `../tom-eye/`

If you copy or install only `repo/`, the code will not work without adapting
those paths.

## Setup

From inside `repo/`, create a Python 3.12 environment named `fiemrok_env` and
install the requirements:

```bash
python3.12 -m venv fiemrok_env
source fiemrok_env/bin/activate
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
and evaluation live together in `lexicon_validation/`.

```python
from fiemrok_experiment import fiemrok
from fiemrok_experiment import select_lexicon
from lexicon_validation import validate_lexicon
from lexicon_validation import cross_boundary_lexicon
```

The previous top-level imports are no longer supported. Shared utilities
(`locations`, `stores`, `lexicon_helper`, `model_paths`, `audio`, and
`annotation`) remain at the repository root. Scripts and tests remain in
`scripts/` and `tests/`. Run from the repository root, or put that directory
on Python's import path.

Default data and output paths are anchored to the repository or its parent
workspace through `locations.py`; moving modules does not move stores or
results. Explicitly supplied relative paths remain relative to the working
directory. Previously pickled custom objects may require migration before
loading under the new module names.

## CGN lexicon evaluation

See [Resume HuBERT mean pooling](docs/hubert_mean_pooling.md) for efficient
shard-grouped frame loading, saved-index reuse, and resumable evaluation.

## Cross-boundary lexicon sampling

See [Adaptive sampling](docs/cross_boundary_sampling.md) for planning quotas,
retrying with consecutive seeds, relaxing constraints, and resuming failures.
See [Marker storage](docs/cross_boundary_sampled_tokens_markers.md) for saving
and loading the sample, and [Marker embeddings](docs/cross_boundary_embeddings.md)
for full-frame wav2vec2 and HuBERT computation.
See [Cross-boundary neighbours](docs/cross_boundary_neighbours.md) to run both
models with mean and center pooling and save results after each layer.
