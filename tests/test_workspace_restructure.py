import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/restructure_workspace.py'


class MigrationTests(unittest.TestCase):
    def test_workflow_move_preserves_payloads_and_updates_saved_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = root / 'inputs/fiemrok_experiment'
            (inputs / 'tom-eye').mkdir(parents=True)
            (inputs / 'tom-eye/example.wav').write_bytes(b'fixture')
            alias = root / 'tom-eye'
            alias.symlink_to('inputs/fiemrok_experiment/tom-eye')
            store = root / 'stores/fiemrok_experiment/echoframe_stimuli_store'
            store.mkdir(parents=True)
            config = {'phraser_sources': {'stimuli':
                str(root / 'stores/fiemrok_experiment/phraser_stimuli_store')}}
            (store / 'config.json').write_text(json.dumps(config))
            bundle = root / 'artifacts/fiemrok_experiment/svs_experiment_data'
            bundle.mkdir(parents=True)
            (bundle / 'sample.npy').write_bytes(b'vectors unchanged')
            metadata = {'directory': str(bundle), 'name': 'sample',
                'metadatas': [{'label': 'fixture'}]}
            original = json.dumps(metadata).encode()
            (bundle / 'sample.json').write_bytes(original)
            inode = store.stat().st_ino
            command = [sys.executable, str(SCRIPT), '--workspace', str(root)]
            subprocess.run(command, check=True, capture_output=True)
            self.assertFalse((root / 'fiemrok_experiment').exists())
            subprocess.run(command + ['--apply'], check=True,
                capture_output=True)
            moved = root / 'fiemrok_experiment/stores/echoframe_stimuli_store'
            self.assertEqual(moved.stat().st_ino, inode)
            updated = json.loads((moved / 'config.json').read_text())
            self.assertEqual(updated['phraser_sources']['stimuli'],
                str(root / 'fiemrok_experiment/stores/phraser_stimuli_store'))
            self.assertEqual(alias.readlink(),
                Path('fiemrok_experiment/inputs/tom-eye'))
            self.assertEqual((alias / 'example.wav').read_bytes(), b'fixture')
            moved_bundle = (root / 'fiemrok_experiment'
                / 'svs_fiemrok_experiment_data')
            saved = json.loads((moved_bundle / 'sample.json').read_text())
            metadata['directory'] = str(moved_bundle)
            self.assertEqual(saved, metadata)
            self.assertEqual((moved_bundle / 'metadata_before_restructure'
                / 'sample.json').read_bytes(), original)
            self.assertEqual((moved_bundle / 'sample.npy').read_bytes(),
                b'vectors unchanged')
            for parent in ('inputs', 'stores', 'artifacts'):
                self.assertFalse((root / parent).exists())
            repeated = subprocess.run(command + ['--apply'], check=True,
                capture_output=True, text=True)
            self.assertIn('Moved 0 paths', repeated.stdout)

    def test_collision_aborts_before_any_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'inputs/fiemrok_experiment'
            source.mkdir(parents=True)
            (root / 'stores/sonority').mkdir(parents=True)
            (root / 'sonority/stores').mkdir(parents=True)
            result = subprocess.run([sys.executable, str(SCRIPT),
                '--workspace', str(root), '--apply'], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(source.exists())
            self.assertFalse((root / 'fiemrok_experiment').exists())

    def test_unknown_files_in_old_containers_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'artifacts/phone_analysis'
            source.mkdir(parents=True)
            (source / 'phone_counts.json').write_text('{}')
            unknown = root / 'artifacts/unknown.txt'
            unknown.write_text('keep')
            subprocess.run([sys.executable, str(SCRIPT), '--workspace',
                str(root), '--apply'], check=True, capture_output=True)
            self.assertEqual(unknown.read_text(), 'keep')
            self.assertTrue((root / 'phone_analysis/metadata/phone_counts.json')
                .exists())


if __name__ == '__main__':
    unittest.main()
