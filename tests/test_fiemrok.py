import pathlib
import sys
import unittest
from unittest import mock


REPO_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import fiemrok


def make_header():
    return [f'column_{i}' for i in range(14)]


def make_line(position='Initial', condition='C'):
    return [
        'dw',
        1,
        condition,
        'woord',
        'ctx',
        'constraint_1',
        'alt',
        'constraint_2',
        'nonwoord',
        'phoneme',
        None,
        None,
        None,
        position,
    ]


class ExperimentTests(unittest.TestCase):
    def test_default_constructor_loads_data_and_audio_info(self):
        header = make_header()
        data = [make_line()]
        audio_info = {}

        with mock.patch.object(fiemrok, 'get_experiment_data',
            return_value=(header, data)) as get_data:
            with mock.patch.object(fiemrok.audio,
                'make_or_load_audio_info_dict',
                return_value=audio_info) as get_audio:
                experiment = fiemrok.Experiment()

        get_data.assert_called_once_with()
        get_audio.assert_called_once_with()
        self.assertIs(experiment.audio_info_dict, audio_info)
        self.assertEqual(experiment.header, header)
        self.assertEqual(experiment.data, data)

    def test_in_memory_data_skips_audio_info_generation(self):
        with mock.patch.object(fiemrok.audio, 'make_or_load_audio_info_dict',
            side_effect=AssertionError(
                'audio info generation should not be called')):
            experiment = fiemrok.Experiment(
                header=make_header(),
                data=[make_line()],
            )

        self.assertEqual(experiment.audio_info_dict, {})
        self.assertEqual(len(experiment.trials), 1)

    def test_in_memory_data_uses_provided_audio_info_dict(self):
        audio_info = {
            'woordWA.wav': {
                'filename': '/tmp/woordWA.wav',
                'duration': 0.5,
            },
            'woordNA.wav': {
                'filename': '/tmp/woordNA.wav',
                'duration': 0.5,
            },
            'woordWM.wav': {
                'filename': '/tmp/woordWM.wav',
                'duration': 0.5,
            },
            'woordNM.wav': {
                'filename': '/tmp/woordNM.wav',
                'duration': 0.5,
            },
        }
        with mock.patch.object(fiemrok.locations, 'textgrid_filename_to_path',
            return_value='/tmp/woordWA.tim'):
            with mock.patch.object(fiemrok.annotation, 'load_textgrid',
                return_value='textgrid'):
                with mock.patch.object(fiemrok.annotation,
                    'extract_time_points', return_value=[0.1, 0.2, 0.3]):
                    experiment = fiemrok.Experiment(
                        header=make_header(),
                        data=[make_line()],
                        audio_info_dict=audio_info,
                    )

        self.assertIs(experiment.audio_info_dict, audio_info)
        self.assertEqual(len(experiment.stimuli), 4)
        self.assertEqual(len(experiment.bad_stimuli), 0)

    def test_missing_audio_entries_land_in_bad_stimuli(self):
        experiment = fiemrok.Experiment(
            header=make_header(),
            data=[make_line()],
            audio_info_dict={},
        )

        self.assertEqual(len(experiment.stimuli), 0)
        self.assertEqual(len(experiment.bad_stimuli), 4)
        self.assertTrue(all(not stimulus.ok for stimulus in
            experiment.bad_stimuli))


class StimulusValidationTests(unittest.TestCase):
    def setUp(self):
        self.experiment = fiemrok.Experiment(
            header=make_header(),
            data=[make_line()],
        )
        self.trial = self.experiment.trials[0]

    def test_invalid_word_type_raises_value_error(self):
        with self.assertRaisesRegex(
            ValueError,
            'word_type should be word or non-word, not invalid',
        ):
            fiemrok.Stimulus(self.trial, word_type='invalid')

    def test_invalid_aligned_type_raises_value_error(self):
        with self.assertRaisesRegex(
            ValueError,
            'aligned_type should be aligned / misaligned, not invalid',
        ):
            fiemrok.Stimulus(self.trial, aligned_type='invalid')


if __name__ == '__main__':
    unittest.main()
