from pathlib import Path
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

from lexicon_validation import lv_locations
from lexicon_validation import embeddings as cross_embeddings
import stores


class LocationTests(unittest.TestCase):
    def test_store_names_use_actual_count_and_flat_store_container(self):
        for model in ('w2v', 'hubert'):
            for condition in ('word', 'cross_boundary'):
                path = lv_locations.store_path(model, condition, 2103)
                self.assertEqual(path,
                    lv_locations.workspace / 'echoframe_stores'
                    / f'{model}_{condition}_2103')


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
            dict(label='f d ə', start=1600, end=1850, audio=audio,
                phrase=phrase),
        ], self.cgn, save=True)
        self.enterContext(mock.patch.object(stores, 'cgn', self.cgn))
        self.wav2vec_path = self.enterContext(mock.patch.object(
            cross_embeddings.model_paths, 'get_pretrained_wav2vec2_model_path',
            return_value=Path('/unused/wav2vec2/checkpoint_200000')))
        self.hubert_path = self.enterContext(mock.patch.object(
            cross_embeddings.model_paths, 'get_pretrained_hubert_model_path',
            return_value=Path('/unused/hubert/checkpoint_200000')))
        self.wav2vec_store, self.hubert_store = self.create_stores()

    def test_setup_registers_one_checkpoint_per_store_from_model_paths(self):
        self.wav2vec_path.assert_called_once_with(200_000, 1)
        self.hubert_path.assert_called_once_with(200_000, 1)
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

    def test_repeated_setup_skips_checkpoint_lookup_and_registration(self):
        self.wav2vec_path.side_effect = AssertionError('already registered')
        self.hubert_path.side_effect = AssertionError('already registered')
        wav2vec, hubert = self.create_stores()
        self.assertEqual(len(wav2vec.model_registry.model_metadatas), 1)
        self.assertEqual(len(hubert.model_registry.model_metadatas), 1)

    def test_loader_attaches_shared_cgn_without_registering_models(self):
        path = self.root / 'another'
        store = stores.load_echoframe_cross_boundary_store(path)
        self.addCleanup(store.close)
        self.assertIs(store.load_phraser_store('cgn'), self.cgn)
        self.assertEqual(store.model_registry.model_metadatas, [])

    def test_wrappers_register_their_model_in_a_fresh_store(self):
        for compute, alias, name in (
            (cross_embeddings.compute_embeddings_for_wv2,
                'w2v', 'wav2vec2_nl1_checkpoint-200000'),
            (cross_embeddings.compute_embeddings_for_hubert,
                'hubert', 'huibert_nl1_checkpoint-200000')):
            with self.subTest(model=name):
                workspace = self.root / 'fresh'
                path = (workspace / 'echoframe_stores'
                    / f'{alias}_cross_boundary_2')
                self.assertFalse(path.exists())
                with (mock.patch.object(echoframe.Store, 'load_model',
                    return_value=object()) as load_model,
                    mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector',
                    return_value=iter(self.outputs()))):
                    store = compute(self.markers, condition='cross_boundary',
                        workspace=workspace, verbose=False)
                self.addCleanup(store.close)
                load_model.assert_called_once_with(name, gpu=False)
                self.assertEqual(store.root, path)
                self.assertEqual([model.model_name for model in
                    store.model_registry.model_metadatas], [name])
                self.assertEqual(len(store.metadatas), 8)

    def test_frames_link_to_markers_and_repeat_skips_inference(self):
        for compute, name in (
            (cross_embeddings.compute_embeddings_for_wv2,
                'wav2vec2_nl1_checkpoint-200000'),
            (cross_embeddings.compute_embeddings_for_hubert,
                'huibert_nl1_checkpoint-200000')):
            with self.subTest(model=name):
                outputs = self.outputs()
                with (mock.patch.object(echoframe.Store, 'load_model',
                    return_value=object()) as load_model,
                    mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector', return_value=iter(outputs))
                    as inference):
                    store = compute(self.markers, condition='cross_boundary',
                        workspace=self.root, verbose=False, gpu=True,
                        batch_size=2)
                    self.addCleanup(store.close)
                load_model.assert_called_once_with(name, gpu=True)
                inference.assert_called_once()
                self.assertEqual(inference.call_args.kwargs['batch_size'], 2)
                self.assertTrue(inference.call_args.kwargs['gpu'])
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
                        self.assertEqual(embedding.phraser_object.key,
                            marker.key)
                        self.assertEqual(embedding.object_class, 'Marker')
                        self.assertEqual(embedding.metadata.phraser_source_id,
                            'cgn')
                        self.assertEqual(set(embedding.metadata.tags),
                            {'cgn', 'cross_boundary'})
                with (mock.patch.object(echoframe.Store, 'load_model',
                    side_effect=AssertionError('model should not load')),
                    mock.patch.object(batch_segment_features.to_vector,
                    'iter_filename_batch_to_vector',
                    side_effect=AssertionError('inference should not run'))):
                    resumed = compute(self.markers, condition='cross_boundary',
                        workspace=self.root, verbose=False)
                    self.addCleanup(resumed.close)
                self.assertEqual(len(resumed.metadatas), 8)

    def test_collar_and_missing_layers_are_independent_storage_requests(self):
        store = self.wav2vec_store
        with (mock.patch.object(store, 'load_model', return_value=object()),
            mock.patch.object(batch_segment_features.to_vector,
            'iter_filename_batch_to_vector', side_effect=lambda *a, **k:
                iter(self.outputs())) as inference):
            cross_embeddings.compute_embeddings(store, self.markers,
                condition='cross_boundary', layers=(3,), verbose=False)
            cross_embeddings.compute_embeddings(store, self.markers,
                condition='cross_boundary', layers=(3, 6), verbose=False)
            self.assertEqual(len(store.metadatas), 4)
            cross_embeddings.compute_embeddings(store, self.markers,
                condition='cross_boundary', layers=(3,), collar=2000,
                verbose=False)
        self.assertEqual(inference.call_count, 3)
        self.assertEqual(len(store.metadatas), 6)
        np.testing.assert_allclose(inference.call_args.kwargs['starts'], [0, 0])
        np.testing.assert_allclose(inference.call_args.kwargs['ends'],
            [3.3, 3.85])

    def test_failed_inference_leaves_caller_owned_store_open(self):
        with mock.patch.object(self.wav2vec_store, 'load_model',
            side_effect=RuntimeError('inference unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'inference unavailable'):
                cross_embeddings.compute_embeddings(self.wav2vec_store,
                    self.markers, condition='cross_boundary', verbose=False)
        path = self.wav2vec_store.root / 'index.lmdb'
        self.assertTrue(lmdb_helper.env_is_open(path))
        self.assertEqual(self.wav2vec_store.metadatas, [])

    def test_word_extraction_keeps_supplied_order_and_links_to_words(self):
        words = [self.cgn.create(phraser.Word, label=f'word_{i}',
            start=marker.start, end=marker.end,
            audio_id=marker.audio.identifier, speaker_id=marker.speaker_id,
            save=True) for i, marker in enumerate(self.markers)]
        words.reverse()
        with (mock.patch.object(echoframe.Store, 'load_model',
            return_value=object()), mock.patch.object(
            batch_segment_features.to_vector, 'iter_filename_batch_to_vector',
            return_value=iter(self.outputs())) as inference):
            store = cross_embeddings.compute_embeddings_for_wv2(words,
                workspace=self.root, layers=(6,), verbose=False)
        self.addCleanup(store.close)
        self.assertEqual(store.root.name, 'w2v_word_2')
        np.testing.assert_allclose(inference.call_args.kwargs['starts'],
            [1.1, 0.6])
        for word in words:
            embedding = store.phraser_key_to_embedding(word.key,
                'wav2vec2_nl1_checkpoint-200000', 6, collar=500)
            self.assertEqual(embedding.object_class, 'Word')
            self.assertEqual(embedding.phraser_object.key, word.key)
            self.assertEqual(set(embedding.metadata.tags), {'cgn', 'word'})

    def test_wrapper_closes_its_store_when_extraction_fails(self):
        store = mock.Mock()
        with (mock.patch.object(cross_embeddings, 'create_store',
            return_value=store), mock.patch.object(cross_embeddings,
            'compute_embeddings', side_effect=RuntimeError('failed'))):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                cross_embeddings.compute_embeddings_for_wv2(self.markers,
                    condition='cross_boundary', workspace=self.root)
        store.close.assert_called_once_with()

    def test_model_identity_is_required_for_multiple_registered_models(self):
        store = mock.Mock()
        store.model_registry.model_metadatas = [mock.Mock(), mock.Mock()]
        with self.assertRaisesRegex(ValueError, 'provide model_name'):
            cross_embeddings.compute_embeddings(store, self.markers)

    def test_explicit_model_identity_is_passed_to_batch_extractor(self):
        store = mock.Mock()
        with mock.patch.object(cross_embeddings,
            'compute_embeddings_batch') as extract:
            cross_embeddings.compute_embeddings(store, self.markers,
                condition='cross_boundary', model_name='requested', layers=(6,))
        self.assertIs(extract.call_args.args[0], self.markers)
        self.assertEqual(extract.call_args.kwargs['model_name'], 'requested')
        self.assertEqual(extract.call_args.kwargs['layers'], (6,))

    def create_stores(self):
        stores_pair = cross_embeddings.create_wv2_and_hubert('cross_boundary',
            2, self.cgn, workspace=self.root)
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
