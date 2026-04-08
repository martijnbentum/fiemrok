import random

import echoframe
import frame
import numpy
import phraser
from progressbar import progressbar
import numpy as np
import to_vector
import speech_vector_search as svs

import locations
import model_paths


def test_prototypes(prototype_index, word_type_dict, store):
    random.seed = 0
    results = {}
    for word_label, metadata in word_type_dict.items():
        if len(metadata) < 100:continue
        mds = random.sample(metadata, 10)
        results[word_label] = {'items':[]}
        all_labels, accuracies = [], []
        for md in mds:
            embedding_array = load_embedding_array(store, md)
            embedding = aggregate_embedding_array(embedding_array, 
                method = 'mean')
            closest_prototypes = prototype_index.query(embedding, 10)
            labels = [x['label'] for x in closest_prototypes['metadata']]
            accuracy = labels.count(word_label) / len(labels)
            items = {'metadata':md, 'closest_prototypes': closest_prototypes,
                'labels': labels, 'accuracy': accuracy}
            results[word_label]['items'].append(items)
            accuracies.append(accuracy)
            all_labels.extend(labels)
        results[word_label]['avg_accuracy'] = np.mean(accuracies)
        results[word_label]['labels'] = all_labels
    return results


def make_prototypes(o = None, store = None, model_name = 'wav2vec2',
    tag = 'filler',layer = 6, subset_size = 10, n_subsets = 10, method = 'mean'):
    if o is None: 
        o = make_embeddings_for_word_types(store, model_name,  tag, 
            layer,  n_tokens = subset_size * n_subsets, method = method)
    X = np.empty((len(o)*n_subsets, 768))
    index = 0
    rows, configs = [], []
    for word_label in progressbar(o.keys()):
        embeddings = o[word_label]['embeddings']
        metadatas = o[word_label]['metadatas']
        vectors, metadata_rows, config = svs.build_subset_prototypes(word_label, 
            embeddings, metadatas, subset_size = subset_size, 
            n_subsets = n_subsets)
        X[index:index+n_subsets] = vectors
        index += n_subsets
        rows.extend(metadata_rows)
        configs.append(config)
    prototype_index = (X, rows)
    return prototype_index, X, rows, configs

        

def make_embeddings_for_word_types(store, model_name = 'wav2vec', 
    tag= 'filler', layer = 6, n_tokens = 100, method = 'mean'):
    word_type_dict = load_store_word_dict(store, model_name = model_name,
        tag = tag, layer = layer)
    o = {}
    for word_label in progressbar(word_type_dict):
        try:
            embeddings, metadatas = make_embeddings_for_word_tokens(word_label, 
                word_type_dict, store, n_tokens = n_tokens, method = method)
        except ValueError as e:
            print(f'Error making embeddings for word label {word_label}: {e}')
            continue
        o[word_label] = {'embeddings': embeddings, 'metadatas': metadatas}
    return o

def load_store_word_dict(store, model_name = 'wav2vec', tag= 'filler', 
    layer = 6):
    wd = {}
    for md in store.metadata:
        if model_name in md.model_name and tag in md.tags:
            key = md.phraser_key
            word_token = phraser.models.cache.load(bytes.fromhex(key))
            word_label = word_token.label
            if word_label not in wd: wd[word_label] = []
            md.label = word_label
            md.phraser_word_token = word_token
            wd[word_label].append(md)
    return wd

def make_embeddings_for_word_tokens(label, store_word_dict, store, 
    n_tokens = 100,method = 'mean'):
    random.seed(0)
    echoframe_metadatas = store_word_dict[label]
    if len(echoframe_metadatas) < n_tokens: 
        m = f'Not enough tokens for label {label}. Requested {n_tokens},'
        m += ' but only {len(echoframe_metadatas)} available.'
        raise ValueError(m)
    metadatas = random.sample(echoframe_metadatas, n_tokens)
    mds, embeddings = [], []
    for md in metadatas:
        embedding_array = load_embedding_array(store, md)
        embedding = aggregate_embedding_array(embedding_array, method = method)
        embeddings.append(embedding)
        mds.append(md.__dict__)
        mds[-1]['echoframe_key'] = md.entry_id
    return numpy.array(embeddings), mds

def load_embedding_array(store, echoframe_metadata):
    return store.metadata_to_payload(echoframe_metadata)

def aggregate_embedding_array(embedding_array, method = 'mean'):
    '''aggregate frames in embedding_array to a single vector using method.
    embedding_array     2D array of shape (n_frames, embedding_dim)
    method              method of aggregation. One of 'mean', 'center_frame',
                    'centroid'. 
    '''
    if method == 'mean': return numpy.mean(embedding_array, axis = 0)
    if method == 'center_frame': 
        return embedding_array[len(embedding_array) // 2]
    if method == 'centroid':
        centroid = svs.centroid(embedding_array)
        distances = numpy.linalg.norm(embedding_array - centroid, axis = 1)
        closest_index = numpy.argmin(distances)
        return embedding_array[closest_index]
    else: raise ValueError(f'Unknown aggregation method {method}')
   




def load_store():
    store = echoframe.Store(locations.embedding_store_path)
    return store

def load_models_final_checkpoint():
    wav2vec2 = load_model('wav2vec2', checkpoint = 200_000)
    hubert = load_model('hubert', checkpoint = 200_000)
    spidr = load_model('spidr', checkpoint = 400_000)
    return [wav2vec2, hubert, spidr]

def store_all_words(store, models = None, layers = [3,6,12], collar = 500):
    if models is None: models = load_models_final_checkpoint()
    store_fillers(store, models, layers = layers, collar = collar)
    store_targets(store, models, layers = layers, collar = collar)


def store_fillers(store, models, layers = [3,6,12], collar = 500):
    words = load_filler_words()
    tags = ['filler']
    for model in models:
        print(f'Storing fillers for model {model.model_name[0]}')
        store_words(store, words, model, tags, layers = layers, 
            collar = collar)

def store_targets(store, models, layers = [3,6,12], collar = 500):
    words = load_target_words()
    tags = ['target']
    for model in models:
        print(f'Storing targets for model {model.model_name[0]}')
        store_words(store, words, model, tags, layers = layers,
            collar = collar)

def store_words(store, words, model, tags, layers = [3,6,12], collar = 500):
    for word in progressbar(words):
        print(f'Processing word {word}, with model {model.model_name[0]}')
        phraser_word_tokens = load_phraser_word_token(word)
        for phraser_word_token in progressbar(phraser_word_tokens):
            for layer in layers:
                handle_phraser_word_token(phraser_word_token, model, store,
                    layers = layers, collar = collar, tags = tags)

def handle_phraser_word_token(phraser_word_token, model, store, layers = [6],
    collar = 500, tags = None):
    try:audio = load_audio(phraser_word_token, collar = collar)
    except ValueError as e:
        print(f'Error loading audio for {phraser_word_token.key.hex()}: {e}')
        return
    outputs = to_vector.to_embeddings.audio_to_vector(audio, model)
    data = select_transformer_frames(phraser_word_token, outputs, 
        collar = collar)
    phraser_key = phraser_word_token.key.hex()
    print(f'Putting {phraser_key} in store with tags {tags}')
    for layer in layers:
        store.put(phraser_key, collar, model.model_name[-1], 'hidden_state',
            layer = layer, data = data, tags = tags)
    

def load_phraser_word_token(word_label):
    return phraser.models.cache.label_to_instances(word_label, 'Word')

def load_audio(phraser_word, collar = 500, enforce_collar = True):
    filename = phraser_word.audio.filename
    audio_duration = phraser_word.audio.duration
    if phraser_word.start - collar < 0 and enforce_collar:
        m = f'Collar of {collar}ms is too large for word starting at '
        m += f'{phraser_word.start}ms'
        raise ValueError(m)
    if phraser_word.end + collar > audio_duration and enforce_collar:
        m = f'Collar of {collar}ms is too large for word ending at '
        m += f'{phraser_word.end}ms in audio of duration {audio_duration}ms'
        raise ValueError(m)
    start = phraser_word.start - collar
    end = phraser_word.end + collar
    audio = to_vector.audio.load_audio_milliseconds(filename, start, end)
    return audio

def load_model(model_name = 'wav2vec2', checkpoint = None, version = None, 
    finetuned = False):
    if finetuned: 
        raise not NotImplementedError('Finetuned models not yet implemented')
    model_path = model_paths.get_pretrained_model(model_name, checkpoint, version)
    print(f'Loading model from {model_path}')
    model = to_vector.load.load_model(model_path)
    model.model_name = [model_name,str(model_path)]
    return model

def select_transformer_frames(phraser_word, outputs, layer = 6, collar = 500):
    collar_seconds = collar / 1000
    start = phraser_word.start_seconds - collar_seconds
    n_frames = frame.frames.determine_n_frames_from_outputs(outputs)
    f = frame.Frames(n_frames, start_time = start, outputs = outputs)
    M = f.transformer(layer, phraser_word.start_seconds,
        phraser_word.end_seconds, average = False, percentage_overlap = 100)
    return M

def to_word_token_vector(phraser_word, outputs, layer = 6, collar = 500):
    collar_seconds = collar / 1000
    start = phraser_word.start_seconds - collar_seconds
    n_frames = frame.frames.determine_n_frames_from_outputs(outputs)
    f = frame.Frames(n_frames, start_time = start, outputs = outputs)
    embedding = f.transformer(layer, phraser_word.start_seconds,
        phraser_word.end_seconds, average = True, percentage_overlap = 100)
    return embedding

def load_filler_words():
    with open('data/filler_words') as f:
        t = f.read().split('\n')
    return [word.strip() for word in t]

def load_target_words():
    with open('data/target_words') as f:
        t = f.read().split('\n')
    return [word.strip() for word in t]
