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
