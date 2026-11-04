'''Move existing workspace files without overwriting; dry run by default.'''

import argparse
import json
import os
from pathlib import Path
import re


def update_vector_directories(workspace):
    '''Update saved-index destinations without changing vector metadata rows.'''
    pattern = rb'(\A\s*\{\s*"directory"\s*:\s*)("(?:[^"\\]|\\.)*")'
    bundles = ('fiemrok_experiment/svs_fiemrok_experiment_data',
        'lexicon_validation/svs_data')
    for bundle in bundles:
        directory = workspace / bundle
        for vector_path in directory.glob('*.npy'):
            path = vector_path.with_suffix('.json')
            if not path.exists(): continue
            with path.open('rb') as source: header = source.read(4096)
            match = re.match(pattern, header)
            if not match:
                raise ValueError(f'Unrecognized saved-index header: {path}')
            desired = json.dumps(str(directory)).encode()
            if json.loads(match[2]) == str(directory): continue
            backup = directory / 'metadata_before_restructure' / path.name
            backup.parent.mkdir(exist_ok=True)
            if not backup.exists(): os.link(path, backup)
            temporary = path.with_suffix('.json.restructure.tmp')
            with path.open('rb') as source, temporary.open('wb') as target:
                target.write(source.read(match.start(2)))
                source.seek(match.end(2))
                target.write(desired)
                while chunk := source.read(1024 * 1024): target.write(chunk)
            temporary.chmod(path.stat().st_mode)
            temporary.replace(path)
            print(f'Updated vector save directory: {path.name}')


def update_registered_paths(workspace):
    '''Update store registries, preserving each original config beside it.'''
    roots = [(str(workspace / src), str(workspace / dst))
        for src, dst in migration_map()]

    def replace(value):
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list): return [replace(v) for v in value]
        if isinstance(value, str):
            for old, new in roots:
                if value == old or value.startswith(old + '/'):
                    return new + value[len(old):]
        return value

    configs = []
    for workflow in ('fiemrok_experiment', 'lexicon_validation',
        'sonority', 'phone_analysis'):
        configs.extend((workspace / workflow / 'stores').rglob('config.json'))
    for path in configs:
        original = path.read_text()
        config = json.loads(original)
        updated = replace(config)
        if config == updated: continue
        backup = path.with_name('config.before_restructure.json')
        if not backup.exists(): backup.write_text(original)
        temporary = path.with_name('config.restructure.tmp')
        temporary.write_text(json.dumps(updated, indent=2) + '\n')
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
        print(f'Updated registered paths: {path}')


def migration_map():
    return [
        ('inputs/fiemrok_experiment', 'fiemrok_experiment/inputs'),
        ('stores/fiemrok_experiment', 'fiemrok_experiment/stores'),
        ('artifacts/fiemrok_experiment/svs_experiment_data',
            'fiemrok_experiment/svs_fiemrok_experiment_data'),
        ('artifacts/fiemrok_experiment/audio_info.json',
            'fiemrok_experiment/audio_info.json'),
        ('results/fiemrok_experiment', 'fiemrok_experiment/results'),
        ('stores/lexicon_validation', 'lexicon_validation/stores'),
        ('artifacts/lexicon_validation/svs_data',
            'lexicon_validation/svs_data'),
        ('artifacts/lexicon_validation/selected_word_keys.pkl',
            'lexicon_validation/samples/selected_word_keys.pkl'),
        ('artifacts/lexicon_validation/filtered_word_keys.pkl',
            'lexicon_validation/samples/filtered_word_keys.pkl'),
        ('artifacts/lexicon_validation/cgn_lexicon_frame_counts.json',
            'lexicon_validation/frame_analysis/cgn_lexicon_frame_counts.json'),
        ('artifacts/lexicon_validation/cgn_lexicon_frame_count_analysis.json',
            'lexicon_validation/frame_analysis/'
            'cgn_lexicon_frame_count_analysis.json'),
        ('results/lexicon_validation', 'lexicon_validation/results'),
        ('stores/sonority', 'sonority/stores'),
        ('results/sonority', 'sonority/results'),
        ('stores/phone_analysis', 'phone_analysis/stores'),
        ('artifacts/phone_analysis', 'phone_analysis/metadata'),
        ('artifacts/codebooks', 'phone_analysis/codebooks'),
        ('artifacts/config', 'maintenance/config'),
        ('restructure_moves.jsonl', 'maintenance/restructure_moves.jsonl')]


def remove_empty_parents(workspace):
    candidates = set()
    for source, _ in migration_map():
        parent = (workspace / source).parent
        while parent != workspace:
            candidates.add(parent)
            parent = parent.parent
    for path in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
        if not path.exists(): continue
        # Remove only empty containers from the former layout.
        try: path.rmdir()
        except OSError: pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path,
        default=Path(__file__).resolve().parents[2])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    pending = []
    destinations = set()
    for source, destination in migration_map():
        src, dst = workspace / source, workspace / destination
        if not src.exists(): continue
        if src.is_symlink() or dst.exists() or dst.is_symlink():
            raise SystemExit(f'Refusing ambiguous migration: {src} -> {dst}')
        if dst in destinations:
            raise SystemExit(f'Multiple sources for {dst}; resolve before moving')
        destinations.add(dst)
        # Rename must remain on the same filesystem; never silently copy stores.
        ancestor = dst.parent
        while not ancestor.exists(): ancestor = ancestor.parent
        if src.stat().st_dev != ancestor.stat().st_dev:
            raise SystemExit(f'Cross-filesystem move requires separate planning: {src}')
        pending.append((src, dst))
        print(f'{source} -> {destination}')
    if not args.apply:
        print(f'Dry run: {len(pending)} moves. Use --apply after closing stores.')
        return
    journal = workspace / 'maintenance/workflow_moves.jsonl'
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('a') as log:
        for src, dst in pending:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Record intent before rename, so interrupted operations are traceable.
            log.write(json.dumps(dict(source=str(src), destination=str(dst))) + '\n')
            log.flush()
            os.fsync(log.fileno())
            src.rename(dst)
    update_registered_paths(workspace)
    update_vector_directories(workspace)
    for name in ('tom-eye', 'stimuli_textgrids'):
        alias = workspace / name
        relative_target = Path('fiemrok_experiment/inputs') / name
        old_targets = (Path('inputs/experiment') / name,
            Path('inputs/fiemrok_experiment') / name)
        if alias.is_symlink() and alias.readlink() in old_targets:
            alias.unlink()
        if (workspace / relative_target).exists() and not alias.exists():
            alias.symlink_to(relative_target, target_is_directory=True)
    remove_empty_parents(workspace)
    print(f'Moved {len(pending)} paths. Journal: {journal}')


if __name__ == '__main__':
    main()
