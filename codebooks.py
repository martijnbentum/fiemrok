from echoframe.store import Store
from echoframe import segment_features as sf
import random
import model_paths
from phraser import models
from progressbar import progressbar
import to_vector 

def load_spider_model():
    p = model_paths.get_pretrained_spidr_model_path()
    sm = to_vector.load.load_model(p)
    return sm

def load_phone_labels():
    with open('data/phone_labels', 'r') as f:
        phone_labels = f.read().split('\n')
    return phone_labels

def compute_load_ci(phone_labels = None, model = None, store = None,
    n_tokens = 1000):
    if phone_labels is None: phone_labels = load_phone_labels()
    if model is None: model = load_spider_model()
    if store is None: store = Store('phone_store')
    random.seed(42)
    ci_dict = {}
    for phone_label in progressbar(phone_labels):
        print(phone_label)
        x = models.cache.label_to_instances(phone_label, 'Phone')
        if len(x) < n_tokens:
            print(phone_label, len(x), 'skipping')
            continue
        print(f'working on {phone_label}')
        if len(x) > n_tokens + n_tokens // 10: 
            sample_size = n_tokens + n_tokens // 10
            x = random.sample(x, sample_size)
        print(f'{phone_label} sampled {len(x)} tokens')
        ci_dict[phone_label] = []
        n = 0
        for token in x:
            try:ci = se.get_codebook_indices(token, model = sm, store = store )
            except: continue
            ci_dict[phone_label].append(ci)
            n += 1
            if n >= n_tokens: break
    return ci_dict

