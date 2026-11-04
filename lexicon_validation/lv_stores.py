'''Open Phraser materials and Echoframe stores for lexicon validation.'''

from pathlib import Path

from echoframe import Store as EchoframeStore
from phraser import Store as PhraserStore

from . import lv_locations


CGN_SOURCE_ID = 'cgn'


def load_phraser_store(components=lv_locations.default_components,
    language=lv_locations.default_language, workspace=lv_locations.workspace,
    path=None):
    '''Open existing local materials; the caller closes the returned Store.
    path: optional explicit path instead of the named local material store
    '''
    if path is None:
        path = lv_locations.phraser_store_path(components, language, workspace)
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f'build the local Phraser store first: {path}')
    return PhraserStore(path=path)


def load_echoframe_store(model, condition, *, materials=None,
    components=lv_locations.default_components,
    language=lv_locations.default_language, workspace=lv_locations.workspace):
    '''Open an existing embedding store with local Phraser materials attached.
    model:     w2v or hubert
    condition: word or cross_boundary
    materials: caller-owned Phraser Store; open local materials when omitted

    The caller closes the returned store. When materials is omitted, also
    call close_phraser_stores() on it to close its local Phraser handle.
    '''
    path = lv_locations.find_store_path(model, condition, workspace)
    owns_materials = materials is None
    if owns_materials:
        materials = load_phraser_store(components, language, workspace)
    embedding_store = None
    try:
        embedding_store = EchoframeStore(path)
        embedding_store.attach_phraser_store(CGN_SOURCE_ID, materials)
    except Exception:
        if embedding_store is not None: embedding_store.close()
        if owns_materials: materials.close()
        raise
    return embedding_store
