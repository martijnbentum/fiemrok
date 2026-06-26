from pathlib import Path
data = Path(__file__).parent.parent


excel_filename = '../trialsNewTryWithReps.ods'

audio_info_dict = Path('../audio_info.json')


cgn_lmdb = Path('/vol/mlusers/mbentum/phraser/data/cgn_lmdb')
if not cgn_lmdb.exists(): cgn_lmdb = data / 'cgn_lmdb'


stimuli = data / 'tom-eye'
embedding_store_path = data / 'embeddings'
phone_embedding_store_path = data / 'phone_embeddings'

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

webmaus_textgrid_dir = data / 'stimuli_textgrids'
webmaus_textgrid_fn = list(webmaus_textgrid_dir.glob('*.TextGrid'))

def audio_filename_to_webmaus_textgrid_path(audio_filename):
    textgrid_filename = audio_filename.replace('.wav', '.TextGrid')
    f = webmaus_textgrid_dir / textgrid_filename
    if not f.exists():
        m = f'{textgrid_filename} does not exist in {webmaus_textgrid_dir}'
        raise ValueError(m)
    return f

phraser_stimuli_store_root = data / 'phraser_stimuli_store'
echoframe_stimuli_store_root = data / 'echoframe_stimuli_store'
echoframe_cgn_lexicon_store = data / 'echoframe_cgn_lexicon_store'
