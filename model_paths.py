from pathlib import Path

speech_training_root = Path('/vol/mlusers/mbentum/speech_training/models')
sr = speech_training_root
wav2vec2_nl1_root = sr / 'pretrained/wav2vec2_the_first'
wav2vec2_nl2_root = sr / 'pretrained/wav2vec2_the_second'
hubert_nl1_root = sr / 'pretrained/huibert_the_first/second_iteration'
hubert_nl2_root = sr / 'pretrained/huibert_the_second/second_iteration'

wav2vec_nl1_checkpoints = list(wav2vec2_nl1_root.glob('checkpoint_*'))
wav2vec_nl2_checkpoints = list(wav2vec2_nl2_root.glob('checkpoint_*'))
hubert_nl1_checkpoints = list(hubert_nl1_root.glob('checkpoint_*'))
hubert_nl2_checkpoints = list(hubert_nl2_root.glob('checkpoint_*'))

spidr_root = Path('/vol/mlusers/mbentum/spidr_models')
spinnetje = spidr_root / 'spinnetje-train'
anansie = spidr_root / 'anansie-nl_training'

spidr_checkpoints = list(spinnetje.glob('*.pt'))
anansie_checkpoints = list(anansie.glob('*.pt'))

finetuned_root = Path('/vol/mlusers/mbentum/speech_training/finetuned')
finetuned_models = list(finetuned_root.glob('*'))

def get_pretrained_model(model_name = 'wav2vec2', checkpoint = None, 
    version = 1):
    if model_name == 'wav2vec2': 
        return get_pretrained_wav2vec2_model_path(checkpoint, version)
    if model_name == 'hubert': 
        return get_pretrained_hubert_model_path(checkpoint, version)
    if model_name == 'spidr' or model_name == 'spider':
        return get_pretrained_spidr_model_path(version, checkpoint)

def get_pretrained_wav2vec2_model_path(checkpoint = 200_000, version= 1):
    version = _handle_version(version)
    checkpoint = _handle_checkpoint(checkpoint, 'wav2vec2')
    cp = wav2vec_nl1_checkpoints if version == 1 else wav2vec_nl2_checkpoints
    for c in cp:
        if f'_{checkpoint}' in c.name:
            m = f'Found checkpoint {checkpoint} for wav2vec2 {version}: \n{c}'
            print(m)
            return c
    print(f'Checkpoint {checkpoint} not found for wav2vec2 version {version}')
    
def get_pretrained_hubert_model_path(checkpoint = 200_000, version=1):
    version = _handle_version(version)
    checkpoint = _handle_checkpoint(checkpoint, 'hubert')
    cp = hubert_nl1_checkpoints if version == 1 else hubert_nl2_checkpoints
    for c in cp:
        if f'_{checkpoint}' in c.name:
            m = f'Found checkpoint {checkpoint} for hubert {version}: \n{c}'
            print(m)
            return c
    print(f'Checkpoint {checkpoint} not found for hubert version {version}')

def get_pretrained_spidr_model_path(version = 1, checkpoint = '400_000'):
    version = _handle_version(version)
    checkpoint = _handle_checkpoint(checkpoint, 'spidr')
    cp_path = spinnetje if version == 1 else anansie
    cp = cp_path / f'step_{int(checkpoint)}.pt'
    if cp.exists():
        m = f'Found checkpoint {checkpoint} for spidr {version}: \n{cp}'
        print(m)
        return cp
    print(f'Checkpoint {checkpoint} not found for spidr version {version}')
    
    
def _handle_version(version = None):
    if version is None: 
        version = 1
    if version == 1 or version == 'the_first' or version == '1':
        version = 1
    elif version == 2 or version == 'the_second' or version == '2':
        version = 2
    else:
        raise ValueError(f'Invalid version: {version}')
    return version
    
def _handle_checkpoint(checkpoint = None, model_name = 'wav2vec'):
    model_name = model_name.lower()
    if checkpoint is None:
        if model_name in ['wav2vec2', 'hubert']: checkpoint = 200_000
        if model_name in ['spidr', 'spider']: checkpoint = '400_000'
    return checkpoint
