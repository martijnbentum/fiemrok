import locations
from phraser import store
from phraser import textgrid_loader as tl
import stores

def experiment_to_db_objects(experiment, store = stores.stimuli):
    stimuli = experiment.stimuli
    add_stimuli_to_db(stimuli, store = store)


def add_stimuli_to_db(stimuli, store = stores.stimuli):
    for stimulus in stimuli:
        add_stimulus_to_db(stimulus, store = store)

def add_stimulus_to_db(stimulus, store = stores.stimuli):
    audio = tl.audio_filename_to_db_object(stimulus.audio_path, store = store,
        save_to_db = True)
    f = locations.audio_filename_to_webmaus_textgrid_path(stimulus.audio_filename)
    o = tl.textgrid_filename_to_database_objects(str(f), store = store,
        audio = audio, save_to_db = True)
