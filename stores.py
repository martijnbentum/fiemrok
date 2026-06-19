from phraser import store
import locations

CGN_SOURCE_ID = 'cgn'
STIMULI_SOURCE_ID = 'stimuli'
cgn = store.Store(locations.cgn_lmdb)
stimulus_store = store.Store(locations.phraser_stimuli_store_root)

def attach_cgn(echoframe_store):
    '''Attach the cgn phraser store as a registered source on an echoframe
    store. Required before saving segment-linked metadata.
    '''
    echoframe_store.attach_phraser_store(CGN_SOURCE_ID, cgn)
    return echoframe_store

def attach_stimuli(echoframe_store):
    '''Attach the phraser stimuli store as a registered source on an echoframe
    store. Required before saving stimulus segment-linked metadata.
    '''
    echoframe_store.attach_phraser_store(STIMULI_SOURCE_ID, stimulus_store)
    return echoframe_store
