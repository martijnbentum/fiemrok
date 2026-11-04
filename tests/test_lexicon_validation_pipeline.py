from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import echoframe
from echoframe import batch_segment_features, key_helper
import numpy as np
import phraser
import speech_vector_search as svs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import build_speech_vector_index_pipeline as pipeline
from lexicon_validation import lv_locations
from lexicon_validation import make_embeddings


class PipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = self.root / 'validation'
        self.cgn = phraser.Store(path=self.root / 'cgn')
        self.addCleanup(self.cgn.close)
        filename = self.root / 'comp-b/nl/sample.wav'
        filename.parent.mkdir(parents=True)
        filename.touch()
        audio = self.cgn.create(phraser.Audio, filename=str(filename),
            duration=4000, save=True)
        speaker = self.cgn.create(phraser.Speaker, name='speaker',
            dataset='test', save=True)
        identity = dict(audio_id=audio.identifier,
            speaker_id=speaker.identifier)
        phrase = phraser.Phrase('phrase', 1000, 1400, **identity)
        for start, label, phones in ((1000, 'left', ('x', 'a')),
            (1200, 'right', ('n', 'x'))):
            word = phraser.Word(label, start, start + 200, **identity)
            syllable = phraser.Syllable(label, start, start + 200, **identity)
            phrase.add_child(word)
            word.add_child(syllable)
            for index, phone in enumerate(phones):
                phone_start = start + index * 100
                syllable.add_child(phraser.Phone(phone, phone_start,
                    phone_start + 100, **identity))
        self.cgn.save_many(phrase.items)
        self.phrase = phrase

    def prepare(self, min_types=1):
        return pipeline.prepare_samples(n_tokens=1, min_types=min_types,
            max_attempts=1, tolerance=0, cgn=self.cgn,
            words=self.phrase.words, phrases=[self.phrase],
            workspace=self.workspace)

    def test_selection_files_are_ordered_keys_at_workspace_root(self):
        selected = self.prepare()
        self.assertEqual(len(selected['word']), 2)
        self.assertEqual(len(selected['cross_boundary']), 1)
        for condition in pipeline.CONDITIONS:
            path = lv_locations.selection_path(condition, self.workspace)
            self.assertEqual(path.parent, self.workspace)
            with path.open('rb') as handle:
                self.assertEqual(pickle.load(handle), selected[condition])
        markers = self.cgn.load_many(selected['cross_boundary'])
        self.assertEqual(markers[0].label, 'a n')

    def test_infeasible_floor_does_not_publish_partial_selection_files(self):
        with self.assertRaisesRegex(ValueError, 'below min_types'):
            self.prepare(min_types=2)
        self.assertFalse(lv_locations.selection_path('word',
            self.workspace).exists())
        self.assertFalse(lv_locations.selection_path('cross_boundary',
            self.workspace).exists())

    def test_saved_samples_build_four_stores_and_all_32_indexes(self):
        selected = self.prepare()

        def inference(filenames, **kwargs):
            base = np.arange(100 * 8, dtype=np.float32).reshape(1, 100, 8)
            return iter(SimpleNamespace(hidden_states=[base + layer + i * 10
                for layer in range(13)]) for i, _ in enumerate(filenames))

        with (mock.patch.object(make_embeddings.model_paths,
            'get_pretrained_model', return_value=Path('/unused/checkpoint')),
            mock.patch.object(echoframe.Store, 'load_model',
            return_value=object()), mock.patch.object(
            batch_segment_features.to_vector, 'iter_filename_batch_to_vector',
            side_effect=inference) as extract):
            stores = pipeline.build_embeddings(cgn=self.cgn,
                workspace=self.workspace, verbose=False)
        self.assertEqual(extract.call_count, 4)
        self.assertEqual({p.name for p in stores.values()},
            {'w2v_word_2', 'hubert_word_2',
            'w2v_cross_boundary_1', 'hubert_cross_boundary_1'})
        paths = pipeline.build_indexes(cgn=self.cgn, workspace=self.workspace)
        self.assertEqual(len(paths), 32)
        for (model, condition, layer, pooling), path in paths.items():
            with self.subTest(condition=(model, condition, layer, pooling)):
                index = svs.VectorIndex.load(str(path.parent), path.stem)
                keys = [key_helper.unpack_echoframe_key(
                    md.source_echoframe_keys[0])['phraser_key']
                    for md in index.metadatas]
                self.assertEqual(keys, selected[condition])
                self.assertEqual(index.metadatas.unit_type,
                    'word' if condition == 'word' else 'marker')
                self.assertTrue(path.exists())
        self.assertEqual(len(list((self.workspace / 'svs_data').glob('*'))), 64)
        self.assertEqual({p.name for p in self.workspace.iterdir()},
            {'word_keys.pkl', 'cross_boundary_marker_keys.pkl',
            'echoframe_stores', 'svs_data'})


if __name__ == '__main__':
    unittest.main()
