import echoframe
from phraser import store
import locations

CGN_SOURCE_ID = 'cgn'
STIMULI_SOURCE_ID = 'stimuli'
cgn = None
stimuli = None

echoframe_cgn = None
echoframe_stimuli = None

def load_phraser_cgn_store(path = locations.cgn_lmdb):
    global cgn
    if cgn: return cgn
    cgn = store.Store(path)
    return cgn

def load_phraser_stimuli_store(path = locations.phraser_stimuli_store_root):
    global stimuli
    if stimuli: return stimuli
    stimuli = store.Store(path)
    return stimuli

def _attach_cgn(echoframe_store):
    '''Attach the cgn phraser store as a registered source on an echoframe
    store. Required before saving segment-linked metadata.
    '''
    cgn = load_phraser_cgn_store()
    echoframe_store.attach_phraser_store(CGN_SOURCE_ID, cgn)
    return echoframe_store

def _attach_stimuli(echoframe_store):
    '''Attach the phraser stimuli store as a registered source on an echoframe
    store. Required before saving stimulus segment-linked metadata.
    '''
    stimuli = load_phraser_stimuli_store()
    echoframe_store.attach_phraser_store(STIMULI_SOURCE_ID, stimuli)
    return echoframe_store

def load_echoframe_stimuli_store(path = locations.echoframe_stimuli_store_root):
    global echoframe_stimuli
    if echoframe_stimuli: return echoframe_stimuli
    echoframe_stimuli = echoframe.store.Store(path)
    _attach_stimuli(echoframe_stimuli)
    return echoframe_stimuli

def load_echoframe_cgn_store(path = locations.echoframe_cgn_lexicon_store):
    global echoframe_cgn
    if echoframe_cgn: 
        if str(echoframe_cgn.root) == str(path):
            return echoframe_cgn
        del echoframe_cgn
    echoframe_cgn = echoframe.store.Store(path)
    _attach_cgn(echoframe_cgn)
    return echoframe_cgn

