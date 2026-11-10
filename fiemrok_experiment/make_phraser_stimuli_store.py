from phraser import textgrids

import locations
import stores


def experiment_to_db_objects(experiment, store=None):
    add_stimuli_to_db(experiment.stimuli, store=store)


def add_stimuli_to_db(stimuli, store=None):
    if store is None: store = stores.load_phraser_stimuli_store()
    for stimulus in stimuli:
        add_stimulus_to_db(stimulus, store=store)


def add_stimulus_to_db(stimulus, store=None):
    if store is None: store = stores.load_phraser_stimuli_store()
    audio = textgrids.audio_filename_to_db_object(stimulus.audio_path,
        store=store, save_to_db=True)
    f = locations.audio_filename_to_webmaus_textgrid_path(
        stimulus.audio_filename)
    textgrids.textgrid_filename_to_database_objects(str(f), store=store,
        audio=audio, save_to_db=True)
