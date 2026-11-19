import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import pickle
import tempfile

import numpy as np
import frame
from echoframe.key_helper import pack_hidden_state_key

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import frame_offset_evaluation as validation
from lexicon_validation import lv_locations


class IndexNameTests(unittest.TestCase):
    def test_pipeline_names_parse_including_cross_boundary(self):
        for model in ('hubert', 'w2v'):
            for condition in ('word', 'cross_boundary'):
                for pooling in ('center', 'mean'):
                    for layer in (3, 6, 9, 12):
                        name = f'{model}_{condition}_5000_{layer}_{pooling}'
                        with self.subTest(name=name):
                            self.assertEqual(lv_locations.parse_index_name(name),
                                dict(model=model, condition=condition, n_types=5000,
                                    layer=layer, pooling=pooling))

    def test_invalid_names_explain_the_expected_pattern(self):
        for name in ('missing', 'hubert_word_many_9_mean',
            'hubert_word_0_9_mean', 'hubert_word_-2_9_mean',
            'hubert_word_2_nine_mean', 'hubert_word_2_8_mean',
            'unknown_word_2_9_mean', 'hubert_unknown_2_9_mean',
            'hubert_word_2_9_max'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'invalid index name.*expected'):
                    lv_locations.parse_index_name(name)


class FrameOffsetValidationTests(unittest.TestCase):
    def setUp(self):
        self.word = SimpleNamespace(start_seconds=1.0,
            end_seconds=2.0, audio=SimpleNamespace(duration=4000,
                filename='audio.wav'), key=b'w' * 22, label='word')
        self.nucleus = SimpleNamespace(start_seconds=1.4,
            end_seconds=1.5)

    def test_window_adds_offset_context_and_preserves_frame_grid(self):
        start, end = validation.collared_window(self.word, 50,
            base_collar_ms=500)
        self.assertAlmostEqual(start, -0.5)
        self.assertAlmostEqual(end, 3.5)
        self.assertAlmostEqual((0.5 - start) / 0.02,
            round((0.5 - start) / 0.02))

    def test_context_embedding_pads_before_recording_start(self):
        from unittest import mock

        audio = np.ones(8)
        with (mock.patch.object(validation.to_vector, 'load_audio',
            return_value=audio) as load_audio,
            mock.patch.object(validation.to_vector, 'audio_to_vector',
            return_value='outputs') as audio_to_vector):
            result = validation.compute_contextual_embedding(self.word,
                -0.0005, 1.0, model='model')
        load_audio.assert_called_once_with('audio.wav', start=0.0, end=1.0)
        np.testing.assert_array_equal(audio_to_vector.call_args.args[0][:8],
            np.zeros(8))
        np.testing.assert_array_equal(audio_to_vector.call_args.args[0][8:],
            audio)
        self.assertEqual(result, 'outputs')

    def test_offsets_are_relative_to_center_pooling_frame(self):
        outputs = SimpleNamespace(hidden_states=(np.arange(500 * 2).reshape(
            1, 500, 2),))
        rows = validation.select_query_frames(outputs, self.word,
            self.nucleus, 0.0, 2, layer=0)
        center = rows[2]
        self.assertEqual(center['offset'], 0)
        self.assertEqual(center['center_index'], 72)
        self.assertEqual(center['query_index'], 72)
        self.assertEqual(rows[0]['query_index'], 70)
        self.assertEqual(rows[-1]['query_index'], 74)
        self.assertTrue(all(row['valid'] for row in rows))

    def test_wider_context_preserves_center_frame_time(self):
        reference = frame.Frames(150, start_time=0.5)
        contextual = frame.Frames(250, start_time=-0.5)
        reference_frames = reference.select_frames(1.4, 1.5,
            percentage_overlap=100)
        contextual_frames = contextual.select_frames(1.4, 1.5,
            percentage_overlap=100)
        reference_center = reference_frames[len(reference_frames) // 2]
        contextual_center = contextual_frames[len(contextual_frames) // 2]
        self.assertEqual(reference_center.start_time,
            contextual_center.start_time)
        self.assertEqual(reference_center.end_time,
            contextual_center.end_time)

    def test_out_of_range_offsets_are_marked_invalid(self):
        outputs = SimpleNamespace(hidden_states=(np.zeros((1, 100, 2)),))
        rows = validation.select_query_frames(outputs, self.word,
            self.nucleus, 0.0, 50, layer=0)
        self.assertFalse(rows[-1]['valid'])
        self.assertIsNone(rows[-1]['query_index'])

    def test_negative_max_offset_is_rejected(self):
        with self.assertRaises(ValueError):
            validation.collared_window(self.word, -1)

    def test_word_center_is_used_without_nucleus(self):
        outputs = SimpleNamespace(hidden_states=(np.zeros((1, 200, 2)),))
        rows = validation.select_query_frames(outputs, self.word, None,
            0.0, 0, layer=0)
        self.assertEqual(rows[0]['center_index'], 74)

    def test_padding_and_frames_crossing_recording_end_are_invalid(self):
        outputs = SimpleNamespace(hidden_states=(np.zeros((1, 250, 2)),))
        rows = validation.select_query_frames(outputs, self.word, None,
            -0.5, 130, layer=0)
        outside = [row for row in rows if row['invalid_reason'] ==
            'outside_recording']
        self.assertTrue(outside)
        self.assertTrue(all(row['vector'] is None for row in outside))
        self.assertFalse(rows[0]['valid'])
        self.assertFalse(rows[-1]['valid'])

    def test_evaluate_word_defaults_to_201_offsets_and_word_center(self):
        outputs = SimpleNamespace(hidden_states=tuple(
            np.zeros((1, 400, 2)) for _ in range(7)))
        with mock.patch.object(validation, 'compute_contextual_embedding',
            return_value=outputs):
            query = validation.evaluate_word(self.word, 'model')
        self.assertEqual(query['key'], self.word.key)
        self.assertEqual(query['reference_kind'], 'word')
        self.assertEqual([r['offset'] for r in query['records']],
            list(range(-100, 101)))
        self.assertEqual(query['records'][100]['query_index'],
            query['records'][100]['center_index'])
        self.assertIsNone(query['records'][0]['vector'])

    def test_unequal_supplied_nuclei_are_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, 'equal lengths'):
            validation.evaluate_words([self.word], None, None, 'unused.h5',
                nuclei=[])

    def test_validation_index_loader_uses_explicit_condition(self):
        with (mock.patch.object(validation.lv_locations, 'find_index_path',
            return_value=Path('/tmp/lv/hubert_word_2_6_mean.npy')) as find,
            mock.patch.object(validation.svs.VectorIndex, 'load') as load):
            validation.load_index('hubert', 6, 'mean', 'word', workspace='/tmp/lv')
        find.assert_called_once_with('hubert', 'word', 6, 'mean',
            Path('/tmp/lv/svs_data'))
        load.assert_called_once_with('/tmp/lv', 'hubert_word_2_6_mean')

    def test_selection_uses_shared_materials_in_saved_order_without_closing(self):
        keys = [self.word.key, b'x' * 22]
        index = SimpleNamespace(metadatas=[SimpleNamespace(
            source_echoframe_keys=[pack_hidden_state_key(1, 6, key, 500)])
            for key in keys])
        materials = mock.Mock()
        items = [self.word, SimpleNamespace(key=keys[1])]
        materials.load_many.return_value = items
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'word_keys.pkl').write_bytes(pickle.dumps(keys))
            with (mock.patch.object(validation, 'load_index', return_value=index),
                mock.patch.object(validation.lv_stores, 'load_phraser_store',
                    return_value=materials) as local,
                mock.patch.object(validation, 'load_model', return_value='model'),
                mock.patch.object(validation, 'evaluate_words') as evaluate):
                validation.evaluate_selection('hubert', 6, 'center', 'word',
                    'results.h5', workspace=directory)
                local.assert_called_once_with(workspace=directory)
                materials.load_many.assert_called_once_with(keys)
                evaluate.assert_called_once_with(items, index, 'model',
                    'results.h5', workspace=directory, gpu=False)
                materials.close.assert_not_called()
                validation.evaluate_selection('hubert', 6, 'center', 'word',
                    'results.h5', workspace=directory, gpu=True)
                validation.load_model.assert_called_with(mock.ANY, gpu=True)
                self.assertIs(evaluate.call_args.kwargs['gpu'], True)
                index.metadatas.reverse()
                with self.assertRaisesRegex(ValueError, 'selection order'):
                    validation.evaluate_selection('hubert', 6, 'center', 'word',
                        'results.h5', workspace=directory)


if __name__ == '__main__':
    unittest.main()
