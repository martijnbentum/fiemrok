import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import frame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import frame_offset_validation as validation


class FrameOffsetValidationTests(unittest.TestCase):
    def setUp(self):
        self.word = SimpleNamespace(start_seconds=1.0,
            end_seconds=2.0, audio=SimpleNamespace(duration=4000,
                filename='audio.wav'))
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


if __name__ == '__main__':
    unittest.main()
