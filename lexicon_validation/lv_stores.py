'''Share local validation stores; close module-owned handles explicitly.'''

from pathlib import Path
from threading import RLock

from echoframe import Store as EchoframeStore
from phraser import Store as PhraserStore

from . import lv_locations


CGN_SOURCE_ID = 'cgn'
_materials = {}
_embeddings = {}
_store_lock = RLock()


def load_phraser_store(components=lv_locations.default_components,
    language=lv_locations.default_language, workspace=lv_locations.workspace,
    path=None):
    '''Return the process-shared local Phraser store for a resolved path.

    Open existing materials only. Use close_stores() for module-owned handles;
    do not close returned stores in individual pipeline stages or results.
    '''
    if path is None:
        path = lv_locations.phraser_store_path(components, language, workspace)
    path = Path(path).resolve()
    with _store_lock:
        cached = _cached_materials(path)
        if cached is not None: return cached['store']
        if not path.is_dir():
            raise FileNotFoundError(f'build the local Phraser store first: {path}')
        store = PhraserStore(path=path)
        _materials[path] = dict(store=store, owned=True)
        return store


def load_echoframe_store(model, condition, *, materials=None,
    components=lv_locations.default_components,
    language=lv_locations.default_language, workspace=lv_locations.workspace):
    '''Return the process-shared embedding store with local materials attached.

    Model and condition identify an existing store. Supplied materials remain
    caller-owned and are registered for reuse. Omitted materials use the named
    local store. Call close_stores() when all module-owned store users finish.
    '''
    path = lv_locations.find_store_path(model, condition, workspace).resolve()
    if materials is None:
        materials_path = lv_locations.phraser_store_path(components, language,
            workspace).resolve()
    else:
        materials_path = Path(materials.path).resolve()
    with _store_lock:
        entry = _embeddings.get(path)
        if entry is not None and entry['materials_path'] != materials_path:
            raise ValueError('embedding store already uses different Phraser materials')
        opened_materials = materials is None and _cached_materials(materials_path) is None
        if materials is None:
            materials = load_phraser_store(path=materials_path)
        else:
            _register_materials(materials)
        if entry is not None:
            if entry['materials'] is not materials:
                entry['store'].attach_phraser_store(CGN_SOURCE_ID, materials,
                    set_back_reference=False)
                entry['materials'] = materials
            return entry['store']
        store = None
        try:
            store = EchoframeStore(path)
            store.attach_phraser_store(CGN_SOURCE_ID, materials,
                set_back_reference=False)
        except BaseException:
            try:
                if store is not None: store.close()
            finally:
                if opened_materials:
                    del _materials[materials_path]
                    materials.close()
            raise
        _embeddings[path] = dict(store=store, materials_path=materials_path,
            materials=materials)
        return store


def _cached_materials(path):
    '''Return a live cached entry, forgetting externally closed materials.'''
    entry = _materials.get(path)
    if entry is not None and not entry['store'].is_open():
        del _materials[path]
        return None
    return entry


def _register_materials(materials):
    '''Reuse caller-owned materials without taking ownership of their handle.'''
    if not materials.is_open(): raise ValueError('supplied Phraser store is closed')
    path = Path(materials.path).resolve()
    entry = _cached_materials(path)
    if entry is not None:
        if entry['store'] is not materials:
            raise ValueError(f'a different Phraser handle is already registered for {path}')
        return
    _materials[path] = dict(store=materials, owned=False)


def close_stores():
    '''Close all module-owned handles, leaving caller-owned materials open.

    Call only after all users have finished. Subsequent loader calls open fresh
    module-owned handles. Supplied Echoframe stores bypassing these loaders are
    unaffected.
    '''
    with _store_lock:
        stores = [entry['store'] for entry in _embeddings.values()]
        stores.extend(entry['store'] for entry in _materials.values()
            if entry['owned'])
        _embeddings.clear()
        _materials.clear()
        errors = []
        for store in stores:
            try: store.close()
            except Exception as error: errors.append(error)
        if errors: raise errors[0]
