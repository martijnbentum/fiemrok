import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fiemrok_experiment import evaluate_retrieval as er
from fiemrok_experiment import fiemrok
from fiemrok_experiment import make_lexicon_vector_index as mlvi
from fiemrok_experiment import select_lexicon
from tests.test_fiemrok import make_header, make_line


def make_experiment():
    '''Initial target, final target and final filler trial, all stimuli ok.'''
    lines = [make_line('Initial'), make_line('Final'),
        make_line('Final', condition='F')]
    lines[0][3], lines[1][3], lines[2][3] = 'begin', 'eind', 'vul'
    audio_info = {}
    for word in ['begin', 'eind', 'vul']:
        for code in ['WA', 'WM', 'NA', 'NM']:
            name = f'{word}{code}.wav'
            audio_info[name] = {'filename': f'/tmp/{name}', 'duration': 0.5}
    with mock.patch.object(fiemrok.locations, 'textgrid_filename_to_path',
        return_value='/tmp/x.tim'):
        with mock.patch.object(fiemrok.annotation, 'load_textgrid'):
            with mock.patch.object(fiemrok.annotation, 'extract_time_points',
                return_value=[0.1, 0.2, 0.3]):
                return fiemrok.Experiment(header=make_header(), data=lines,
                    audio_info_dict=audio_info)


def make_result(labels):
    metadatas = [SimpleNamespace(label=x) for x in labels]
    return {'metadatas': metadatas, 'scores': [0.5] * len(labels)}


class ExperimentSelectTests(unittest.TestCase):
    def setUp(self):
        self.experiment = make_experiment()

    def test_select_final_aligned_target_words(self):
        selection = self.experiment.select(position='Final',
            word_type='word', aligned_type='aligned')
        self.assertEqual([x.audio_filename for x in selection],
            ['eindWA.wav'])

    def test_select_fillers(self):
        selection = self.experiment.select(word_type='word', target=False)
        self.assertEqual([x.audio_filename for x in selection],
            ['vulWA.wav', 'vulWM.wav'])

    def test_word_lists(self):
        self.assertEqual(self.experiment.target_words, ['begin', 'eind'])
        self.assertEqual(self.experiment.final_target_words, ['eind'])
        self.assertEqual(self.experiment.initial_target_words, ['begin'])
        self.assertEqual(self.experiment.filler_words, ['vul'])

    def test_echoframe_store_skips_stimulus_without_word(self):
        store = SimpleNamespace(metadatas=[SimpleNamespace(label='eind')])
        for stimulus in self.experiment.targets:
            stimulus.word = SimpleNamespace(label=stimulus.trial.word)
        with mock.patch('builtins.print'):
            self.experiment.add_echoframe_stimuli_store(store)
        final = self.experiment.final_targets[0]
        self.assertEqual(final.metadatas, store.metadatas)
        self.assertFalse(hasattr(self.experiment.fillers[0], 'metadatas'))


class RetrievalTests(unittest.TestCase):
    def test_compute_accuracy(self):
        result = make_result(['eind', 'end', 'eind', 'ijn'])
        self.assertEqual(er.compute_accuracy(result, 'eind'), 0.5)
        self.assertEqual(er.compute_accuracy(make_result([]), 'eind'), 0.0)

    def test_compare_embeddings_accuracies_per_aligned_type(self):
        experiment = make_experiment()
        search_vector = mock.Mock()
        search_vector.query.side_effect = [make_result(['eind', 'x']),
            make_result(['x', 'x'])]
        ce = er.CompareEmbeddings(experiment=experiment,
            search_vector=search_vector, search_depth=2)
        with mock.patch.object(er, 'load_center_frame_nucleus_speech_vector'):
            ce.compute_results()
        self.assertEqual(ce.accuracies('aligned'), [0.5])
        self.assertEqual(ce.accuracies('misaligned'), [0.0])
        self.assertEqual(ce.mean_accuracy('aligned'), 0.5)

    def test_check_stimulus_requires_two_syllables(self):
        stimulus = SimpleNamespace(word=SimpleNamespace(syllables=['s']))
        with self.assertRaisesRegex(ValueError, '2 syllables'):
            er.check_stimulus_ok(stimulus)

    def test_center_frame_nucleus_vector_uses_syllable_nucleus(self):
        nucleus = object()
        word = SimpleNamespace(syllables=[SimpleNamespace(nucleus=['a']),
            SimpleNamespace(nucleus=[nucleus])])
        data = np.arange(6, dtype=float).reshape(3, 2)
        embedding = mock.Mock()
        embedding.sub_embedding.return_value = SimpleNamespace(data=data)
        vector = mlvi.center_frame_nucleus_vector(embedding, word, 1)
        embedding.sub_embedding.assert_called_once_with(nucleus)
        np.testing.assert_array_equal(vector, data[1])


class SelectLexiconTests(unittest.TestCase):
    def test_find_targets_to_add_skips_words_in_lexicon(self):
        experiment = SimpleNamespace(final_target_words=['eind', 'kat'])
        selected_words = [SimpleNamespace(label='kat')]
        tokens = [SimpleNamespace(label='eind')]
        cgn = mock.Mock()
        cgn.label_to_instances.return_value = tokens
        with mock.patch.object(select_lexicon,
            'filter_word_tokens_for_lexicon', side_effect=lambda x,
            min_freq: x):
            with mock.patch('builtins.print'):
                addition = select_lexicon.find_targets_to_add(experiment,
                    selected_words, cgn=cgn)
        cgn.label_to_instances.assert_called_once_with('eind', 'Word')
        self.assertEqual(addition, tokens)


if __name__ == '__main__':
    unittest.main()
