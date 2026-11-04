from pathlib import Path
repo = Path(__file__).resolve().parent
# Small versioned resources remain in repo/data; generated files live outside it.
workspace = repo.parent
data = workspace
fiemrok_experiment = workspace / 'fiemrok_experiment'
lexicon_validation = workspace / 'lexicon_validation'
phone_analysis = workspace / 'phone_analysis'
sonority = workspace / 'sonority'
experiment_inputs = fiemrok_experiment / 'inputs'
experiment_stores = fiemrok_experiment / 'stores'
lexicon_stores = lexicon_validation / 'stores'
sonority_results = sonority / 'results'
phone_counts = phone_analysis / 'metadata/phone_counts.json'
phone_labels = repo / 'data/phone_labels'
frame_counts = lexicon_validation / 'frame_analysis/cgn_lexicon_frame_counts.json'
frame_count_analysis = (lexicon_validation
    / 'frame_analysis/cgn_lexicon_frame_count_analysis.json')
mean_pooling_comparison = (lexicon_validation
    / 'results/mean_pooling_dataset_comparison.json')


excel_filename = experiment_inputs / 'trialsNewTryWithReps.ods'

audio_info_dict = fiemrok_experiment / 'audio_info.json'


cgn_lmdb = Path('/vol/mlusers/mbentum/phraser/data/cgn_lmdb')
if not cgn_lmdb.exists(): cgn_lmdb = data / 'cgn_lmdb'


stimuli = experiment_inputs / 'tom-eye'
embedding_store_path = experiment_stores / 'embeddings'
phone_embedding_store_path = phone_analysis / 'stores/phone_embeddings'

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

webmaus_textgrid_dir = experiment_inputs / 'stimuli_textgrids'
webmaus_textgrid_fn = list(webmaus_textgrid_dir.glob('*.TextGrid'))

def audio_filename_to_webmaus_textgrid_path(audio_filename):
    textgrid_filename = audio_filename.replace('.wav', '.TextGrid')
    f = webmaus_textgrid_dir / textgrid_filename
    if not f.exists():
        m = f'{textgrid_filename} does not exist in {webmaus_textgrid_dir}'
        raise ValueError(m)
    return f

phraser_stimuli_store_root = experiment_stores / 'phraser_stimuli_store'
echoframe_stimuli_store_root = experiment_stores / 'echoframe_stimuli_store'
echoframe_cgn_lexicon_store = experiment_stores / 'echoframe_cgn_lexicon_store'

svs_data = lexicon_validation / 'svs_data'
svs_fiemrok_experiment_data = (fiemrok_experiment
    / 'svs_fiemrok_experiment_data')

filtered_word_keys = lexicon_validation / 'samples/filtered_word_keys.pkl'
selected_word_keys = lexicon_validation / 'samples/selected_word_keys.pkl'

cgn_word_wav2vec_store = lexicon_stores / 'words/lexicon100_2466_cgn'
cgn_word_hubert_store = lexicon_stores / 'words/lexicon100_2466_cgn_hubert'
cross_boundary_stores = lexicon_stores / 'cross_boundary'
cross_boundary_wav2vec_store = (cross_boundary_stores
    / 'cross_boundary_lexicon100_2200_cgn')
cross_boundary_hubert_store = (cross_boundary_stores
    / 'cross_boundary_lexicon100_2200_cgn_hubert')
cross_boundary_marker_keys = svs_data / 'cross_boundary_marker_keys.pkl'

echoframe_sonority_store_root = sonority / 'stores'

all_results = fiemrok_experiment / 'results/all_results.json'
