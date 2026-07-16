from echoframe import batch_segment_features
import stores
from sonority_rsa import display_analysis, run_analysis, save_analysis


def load_cgn():
    '''Load the CGN dataset from the stores module.
    '''
    cgn = stores.load_phraser_cgn_store()
    return cgn

def load_echoframe_sonority_store():
    '''Load the echoframe sonority store from the stores module.
    '''
    sonority_store = stores.load_echoframe_sonority_store()
    return sonority_store

def select_cgn_data(cgn):
    phrases = list(cgn.phrases)
    knl = [x for x in phrases if 'comp-k/nl' in x.audio.filename]
    sd = {}
    syllables, selection = [], []
    for x in knl:
        if x.speaker not in sd: sd[x.speaker] = []
        sd[x.speaker].append(x)
        selection.extend(sd[x.speaker][:30])
    for phrase in selection:
        syllables.extend(x.syllables)
    return syllables, selection, sd

def compute_and_store_vectors(model_name, batch_size= 32, selection = None,
    echoframe_store = None, layers = [1,3,6,9,12], gpu = False):
    if selection is None: _, selection, _ = select_cgn_data(load_cgn())
    if echoframe_store is None: echoframe_store = load_echoframe_sonority_store()
    batch_segment_features.compute_and_store_vectors(selection,
        model_name = model_name, batch_size = batch_size, tags = [model_name],
        store = echoframe_store, layers = layers, gpu = gpu)

def run_analysis(model_name, syllables = None, layers = [1, 3, 6, 9, 12],
    subset_size = 1000, n_subsets = 12, echoframe_store = echoframe_store,
    compute_intensity_baseline = True, compute_random_baseline = True):
    if syllables is None: syllables, _, _ = select_cgn_data(load_cgn())
    summary, scores, log = run_analysis(syllables, model_name = model_name, 
        layers = layers, subset_size = subset_size, n_subsets = n_subsets,
        echoframe_store = echoframe_store, 
        compute_intensity_baseline = compute_intensity_baseline,
        compute_random_baseline = compute_random_baseline)
    output_dir = f'../sonority_results/{model_name}/'
    print(f'Saving analysis results to {output_dir}')
    save_analysis(summary, scores, log, output_dir)
    
        

    

