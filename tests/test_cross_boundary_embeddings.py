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

import locations
import make_cross_boundary_embeddings as cross_embeddings
import stores


class LocationTests(unittest.TestCase):
    def test_cross_boundary_paths_are_anchored_to_repo(self):
        self.assertEqual(locations.cross_boundary_wav2vec_store,
            locations.repo / 'cross_boundary_lexicon100_2200_cgn')
        self.assertEqual(locations.cross_boundary_hubert_store,
            locations.repo / 'cross_boundary_lexicon100_2200_cgn_hubert')
        self.assertEqual(locations.cross_boundary_marker_keys,
            locations.repo / 'svs_data/cross_boundary_marker_keys.pkl')


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
        phrase = self.cgn.create(phraser.Phrase, label='source phrase',
            start=1000, end=2000, audio_id=audio.identifier,
            speaker_id=speaker.identifier)
        self.cgn.save(phrase)
        self.markers = phraser.make_markers([
            dict(label='a n', start=1100, end=1300, audio=audio, phrase=phrase),
            dict(label='f d ə', start=1600, end=1850, audio=audio, phrase=phrase),
        ], self.cgn, save=True)
        keys_path = self.root / 'markers.pkl'
        with keys_path.open('wb') as f:
            pickle.dump([m.key for m in self.markers], f)
        self.enterContext(mock.patch.multiple(locations,
            cross_boundary_wav2vec_store=self.root / 'wav2vec',
            cross_boundary_hubert_store=self.root / 'hubert',
            cross_boundary_marker_keys=keys_path))
        self.enterContext(mock.patch.object(stores, 'cgn', self.cgn))
        self.wav2vec_path = self.enterContext(mock.patch.object(
            cross_embeddings.model_paths, 'get_pretrained_wav2vec2_model_path',
            return_value=Path('/unused/wav2vec2/checkpoint_200000')))
        self.hubert_path = self.enterContext(mock.patch.object(
            cross_embeddings.model_paths, 'get_pretrained_hubert_model_path',
            return_value=Path('/unused/hubert/checkpoint_200000')))
        self.wav2vec_store, self.hubert_store = self.create_stores()

    def test_setup_registers_one_checkpoint_per_store_from_model_paths(self):
        self.wav2vec_path.assert_called_once_with()
        self.hubert_path.assert_called_once_with()
        for store, name, architecture in (
            (self.wav2vec_store, 'wav2vec2_nl1_checkpoint-200000', 'wav2vec2'),
            (self.hubert_store, 'huibert_nl1_checkpoint-200000', 'hubert')):
            with self.subTest(model=name):
                models = store.model_registry.model_metadatas
                self.assertEqual(len(models), 1)
                self.assertEqual(models[0].model_name, name)
                self.assertEqual(models[0].local_path,
                    f'/unused/{architecture}/checkpoint_200000')
                self.assertEqual(models[0].architecture, architecture)
                self.assertEqual(models[0].language, 'Dutch')
                self.assertEqual(models[0].size, 'base')
                self.assertIs(store.load_phraser_store('cgn'), self.cgn)

    def test_setup_can_be_repeated_without_checkpoint_lookup_or_reregistration(self):
        self.wav2vec_path.side_effect = AssertionError('already registered')
        self.hubert_path.side_effect = AssertionError('already registered')
        wav2vec, hubert = self.create_stores()
        self.assertEqual(len(wav2vec.model_registry.model_metadatas), 1)
        self.assertEqual(len(hubert.model_registry.model_metadatas), 1)

    def test_loader_attaches_shared_cgn_without_registering_models(self):
        store = stores.load_echoframe_cross_boundary_store(self.root / 'another')
        self.addCleanup(store.close)
        self.assertIs(store.load_phraser_store('cgn'), self.cgn)
        self.assertEqual(store.model_registry.model_metadatas, [])

    def test_full_frame_matrices_link_to_markers_and_repeat_skips_inference(self):
        for store, name in (
            (self.wav2vec_store, 'wav2vec2_nl1_checkpoint-200000'),
            (self.hubert_store, 'huibert_nl1_checkpoint-200000')):
            with self.subTest(model=name):
                outputs = self.outputs()
                with (mock.patch.object(store, 'load_model', return_value=object())
                    as load_model, mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector', return_value=iter(outputs))
                    as inference):
                    cross_embeddings.compute_embeddings(store, verbose=False)
                load_model.assert_called_once_with(name, gpu=False)
                inference.assert_called_once()
                np.testing.assert_allclose(inference.call_args.kwargs['starts'],
                    [0.6, 1.1])
                np.testing.assert_allclose(inference.call_args.kwargs['ends'],
                    [1.8, 2.35])
                self.assertEqual(len(store.metadatas), 8)
                store = self.reopen(store)
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
                with (mock.patch.object(store, 'load_model',
                    side_effect=AssertionError('model should not load')),
                    mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector',
                    side_effect=AssertionError('inference should not run'))):
                    cross_embeddings.compute_embeddings(store, verbose=False)
                self.assertEqual(len(store.metadatas), 8)

    def test_collar_and_missing_layers_are_independent_storage_requests(self):
        store = self.wav2vec_store
        with (mock.patch.object(store, 'load_model', return_value=object()),
            mock.patch.object(batch_segment_features.to_vector,
            'iter_filename_batch_to_vector', side_effect=lambda *a, **k:
                iter(self.outputs())) as inference):
            cross_embeddings.compute_embeddings(store, layers=(3,), verbose=False)
            cross_embeddings.compute_embeddings(store, layers=(3, 6), verbose=False)
            self.assertEqual(len(store.metadatas), 4)
            cross_embeddings.compute_embeddings(store, layers=(3,), collar=2000,
                verbose=False)
        self.assertEqual(inference.call_count, 3)
        self.assertEqual(len(store.metadatas), 6)
        np.testing.assert_allclose(inference.call_args.kwargs['starts'], [0, 0])
        np.testing.assert_allclose(inference.call_args.kwargs['ends'], [3.3, 3.85])

    def test_failed_inference_leaves_caller_owned_store_open(self):
        with mock.patch.object(self.wav2vec_store, 'load_model',
            side_effect=RuntimeError('inference unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'inference unavailable'):
                cross_embeddings.compute_embeddings(self.wav2vec_store,
                    verbose=False)
        self.assertTrue(lmdb_helper.env_is_open(self.wav2vec_store.root / 'index.lmdb'))
        self.assertEqual(self.wav2vec_store.metadatas, [])

    def create_stores(self):
        stores_pair = cross_embeddings.create_stores()
        for store in stores_pair:
            self.addCleanup(store.close)
        return stores_pair

    def reopen(self, store):
        root = store.root
        store.close()
        restored = stores.load_echoframe_cross_boundary_store(root)
        self.addCleanup(restored.close)
        return restored

    def outputs(self):
        base = np.arange(200 * 768, dtype=np.float32).reshape(1, 200, 768)
        return [SimpleNamespace(hidden_states=[base + layer * 100000
            + index * 1000000 for layer in range(13)]) for index in range(2)]


if __name__ == '__main__':
    unittest.main()
