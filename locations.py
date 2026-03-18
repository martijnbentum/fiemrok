from pathlib import Path
data = Path(__file__).parent.parent


excel_filename = '../trialsNewTryWithReps.ods'

audio_info_dict = Path('../audio_info.json')

stimuli = data / 'tom-eye'

audio_fn= list(stimuli.glob('*.wav'))
textgrid_fn = list(stimuli.glob('*.tim'))

def textgrid_filename_to_path(textgrid_filename):
    f = stimuli / textgrid_filename
    if not f.exists():
        raise ValueError(f'{textgrid_filename} does not exist in {stimuli}')
    return f

def audio_filename_to_path(audio_filename):
    f = stimuli / audio_filename
    if not f.exists():
        raise ValueError(f'{audio_filename} does not exist in {stimuli}')
    return f
