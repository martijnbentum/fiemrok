from pathlib import Path
repo = Path(__file__).resolve().parent
data = repo.parent


excel_filename = data / 'trialsNewTryWithReps.ods'

audio_info_dict = data / 'audio_info.json'


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

svs_data = repo / 'svs_data'
svs_experiment_data = repo / 'svs_experiment_data'

filtered_word_keys = repo / 'data/filtered_word_keys.pkl'
selected_word_keys = repo / 'data/selected_word_keys.pkl'

cgn_word_wav2vec_store = repo / 'lexicon100_2466_cgn'
cgn_word_hubert_store = repo / 'lexicon100_2466_cgn_hubert'
cross_boundary_wav2vec_store = repo / 'cross_boundary_lexicon100_2200_cgn'
cross_boundary_hubert_store = repo / 'cross_boundary_lexicon100_2200_cgn_hubert'
cross_boundary_marker_keys = svs_data / 'cross_boundary_marker_keys.pkl'

echoframe_sonority_store_root = data / 'sonority'

all_results = data / 'all_results.json'
