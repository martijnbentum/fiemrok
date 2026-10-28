import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import echoframe
from echoframe import lmdb_helper
from echoframe import batch_segment_features
import numpy as np
import phraser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_cross_boundary_embeddings as cross_embeddings


class CrossBoundaryEmbeddingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cgn = phraser.Store(path=self.root / 'cgn')
        self.addCleanup(self.cgn.close)
        filename = self.root / 'audio.wav'
        filename.touch()
        audio = self.cgn.create(phraser.Audio, filename=str(filename),
            duration=4000, save=True)
        speaker = self.cgn.create(phraser.Speaker, name='speaker',
            dataset='test', save=True)
        self.phrase = self.cgn.create(phraser.Phrase, label='source phrase',
            start=1000, end=2000, audio_id=audio.identifier,
            speaker_id=speaker.identifier)
        self.cgn.save(self.phrase)
        self.markers = phraser.make_markers([
            dict(label='a n', start=1100, end=1300, audio=audio,
                phrase=self.phrase),
            dict(label='f d ə', start=1600, end=1850, audio=audio,
                phrase=self.phrase),
        ], self.cgn, save=True)
        self.keys_path = self.root / 'markers.pkl'
        with self.keys_path.open('wb') as f:
            pickle.dump([m.key for m in self.markers], f)
        for model, (name, reference, _) in cross_embeddings.MODEL_SETTINGS.items():
            path = self.root / reference
            path.mkdir()
            record = dict(model_name=name, model_id=14 if model == 'wav2vec2'
                else 261, local_path=f'/unused/{model}/checkpoint_200000',
                huggingface_id=None, language='Dutch', size='base',
                architecture=model)
            (path / 'config.json').write_text(json.dumps({'models': {
                name: record, 'unrelated_checkpoint': dict(record,
                    model_name='unrelated_checkpoint', model_id=999)}}))

    def reference(self, model):
        return self.root / cross_embeddings.MODEL_SETTINGS[model][1]

    def open_store(self, model='wav2vec2', **kwargs):
        store = cross_embeddings.load_store(model,
            root=self.root / f'output_{model}',
            reference_root=self.reference(model), phraser_store=self.cgn, **kwargs)
        self.addCleanup(store.close)
        return store

    def compute(self, model='wav2vec2', **kwargs):
        store = cross_embeddings.compute_embeddings(model=model,
            root=self.root / f'output_{model}',
            reference_root=self.reference(model), phraser_store=self.cgn,
            keys_path=self.keys_path, verbose=False, **kwargs)
        self.addCleanup(store.close)
        return store

    def outputs(self):
        base = np.arange(200 * 768, dtype=np.float32).reshape(1, 200, 768)
        return [SimpleNamespace(hidden_states=[base + layer * 100000
            + index * 1000000 for layer in range(13)]) for index in range(2)]

    def test_loader_copies_only_selected_checkpoint_and_attaches_cgn(self):
        for model, (name, _, _) in cross_embeddings.MODEL_SETTINGS.items():
            with self.subTest(model=model):
                store = self.open_store(model)
                record = store.load_model_metadata(name)
                self.assertEqual(record.local_path,
                    f'/unused/{model}/checkpoint_200000')
                self.assertEqual(record.architecture, model)
                self.assertEqual(record.language, 'Dutch')
                self.assertEqual(record.size, 'base')
                self.assertEqual(len(store.model_registry.model_metadatas), 1)
                self.assertNotIn(record.model_id, (14, 261))
                self.assertIs(store.load_phraser_store('cgn'), self.cgn)

    def test_existing_store_needs_no_reference_config(self):
        original = self.open_store()
        root = original.root
        original.close()
        store = cross_embeddings.load_store(root=root,
            reference_root=self.root / 'missing', phraser_store=self.cgn)
        self.addCleanup(store.close)
        self.assertIsNotNone(store.load_model_metadata(
            cross_embeddings.MODEL_SETTINGS['wav2vec2'][0]))

    def test_model_specific_default_paths(self):
        with mock.patch.object(cross_embeddings, 'ROOT', self.root):
            for model, (_, _, destination) in cross_embeddings.MODEL_SETTINGS.items():
                store = cross_embeddings.load_store(model, phraser_store=self.cgn)
                self.addCleanup(store.close)
                self.assertEqual(store.root, self.root / destination)

    def test_missing_checkpoint_registration_closes_store(self):
        reference = self.reference('wav2vec2')
        (reference / 'config.json').write_text('{"models": {}}')
        with self.assertRaisesRegex(ValueError, 'is not registered'):
            self.open_store()
        self.assertFalse(lmdb_helper.env_is_open(
            self.root / 'output_wav2vec2' / 'index.lmdb'))

    def test_rejects_unknown_model_empty_sample_and_words_before_inference(self):
        with self.assertRaisesRegex(ValueError, 'model must be'):
            cross_embeddings.load_store('unknown', root=self.root / 'invalid')
        self.assertFalse((self.root / 'invalid').exists())
        for segments in ([], [self.phrase]):
            with self.subTest(segments=segments):
                with mock.patch.object(cross_embeddings, 'load_store') as loader:
                    with self.assertRaises(ValueError):
                        self.compute(markers=segments)
                    loader.assert_not_called()

    def test_full_frame_matrices_link_to_markers_and_repeat_skips_inference(self):
        for model, (name, _, _) in cross_embeddings.MODEL_SETTINGS.items():
            with self.subTest(model=model):
                outputs = self.outputs()
                with (mock.patch.object(echoframe.Store, 'load_model',
                    return_value=object()) as load_model,
                    mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector', return_value=iter(outputs))
                    as inference):
                    store = self.compute(model)
                load_model.assert_called_once_with(name, gpu=False)
                inference.assert_called_once()
                np.testing.assert_allclose(inference.call_args.kwargs['starts'],
                    [0.6, 1.1])
                np.testing.assert_allclose(inference.call_args.kwargs['ends'],
                    [1.8, 2.35])
                self.assertEqual(len(store.metadatas), 8)
                store.close()
                store = self.open_store(model)
                for index, marker in enumerate(self.markers):
                    stop = 34 if index == 0 else 37
                    for layer in (3, 6, 9, 12):
                        embedding = store.phraser_key_to_embedding(marker.key,
                            name, layer, collar=500)
                        np.testing.assert_array_equal(embedding.data,
                            outputs[index].hidden_states[layer][0, 25:stop, :])
                        self.assertEqual(embedding.shape[1], 768)
                        self.assertEqual(embedding.phraser_object.key, marker.key)
                        self.assertEqual(embedding.object_class, 'Marker')
                        self.assertEqual(embedding.metadata.phraser_source_id, 'cgn')
                        self.assertEqual(set(embedding.metadata.tags),
                            {'cgn', 'cross_boundary'})
                store.close()
                with (mock.patch.object(echoframe.Store, 'load_model',
                    side_effect=AssertionError('model should not load')),
                    mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector',
                    side_effect=AssertionError('inference should not run'))):
                    resumed = self.compute(model)
                self.assertEqual(len(resumed.metadatas), 8)

    def test_collar_and_missing_layers_are_independent_storage_requests(self):
        with (mock.patch.object(echoframe.Store, 'load_model', return_value=object()),
            mock.patch.object(batch_segment_features.to_vector,
            'iter_filename_batch_to_vector', side_effect=lambda *a, **k:
                iter(self.outputs())) as inference):
            first = self.compute(markers=self.markers, layers=(3,))
            first.close()
            second = self.compute(markers=self.markers, layers=(3, 6))
            self.assertEqual(len(second.metadatas), 4)
            second.close()
            third = self.compute(markers=self.markers, layers=(3,), collar=2000)
        self.assertEqual(inference.call_count, 3)
        self.assertEqual(len(third.metadatas), 6)
        np.testing.assert_allclose(inference.call_args.kwargs['starts'], [0, 0])
        np.testing.assert_allclose(inference.call_args.kwargs['ends'], [3.3, 3.85])

    def test_failed_inference_closes_destination_for_resume(self):
        with mock.patch.object(echoframe.Store, 'load_model',
            side_effect=RuntimeError('inference unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'inference unavailable'):
                self.compute()
        self.assertFalse(lmdb_helper.env_is_open(
            self.root / 'output_wav2vec2' / 'index.lmdb'))


if __name__ == '__main__':
    unittest.main()
