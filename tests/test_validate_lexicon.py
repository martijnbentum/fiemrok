import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import analyse_results
import validate_lexicon


def metadata(label, key):
    return SimpleNamespace(label=label, source_echoframe_keys=[key])


class PoolingTests(unittest.TestCase):
    def test_mean_and_center_use_different_frames(self):
        payload = np.array([[0., 0.], [9., 3.], [0., 0.]])
        token = SimpleNamespace(load_payload=lambda: payload)
        mean = validate_lexicon.load_speech_vectors([token], pooling='mean')
        center = validate_lexicon.load_speech_vectors([token])
        np.testing.assert_array_equal(mean, [[3., 1.]])
        np.testing.assert_array_equal(center, [[9., 3.]])

    def test_invalid_pooling_is_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            validate_lexicon.load_speech_vectors([], pooling='median')

    def test_mean_index_has_separate_name_and_honours_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            vector_metadata = SimpleNamespace(
                path=pathlib.Path(directory) / 'w2v_6_mean.json')
            token = SimpleNamespace(echoframe_key=b'a')
            with (mock.patch.object(validate_lexicon, 'load_metadata_layer',
                return_value=[token]), mock.patch.object(
                validate_lexicon, 'make_vector_metadata',
                return_value=vector_metadata) as make_metadata, mock.patch.object(
                validate_lexicon, 'load_speech_vectors') as load_vectors,
                mock.patch.object(validate_lexicon,
                'make_vector_search_index') as make_index):
                validate_lexicon.make_vector_search_index_for_layer(
                    object(), 6, 'w2v_6', pooling='mean',
                    directory=directory, overwrite=True)
                make_metadata.assert_called_once_with([token],
                    name='w2v_6_mean',
                    directory=directory)
                load_vectors.assert_called_once_with([token], pooling='mean')
                make_index.return_value.save.assert_called_once_with(
                    overwrite=True)

    def test_existing_index_reorders_source_metadata_by_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'w2v_6_mean.json'
            path.touch()
            path.with_suffix('.npy').touch()
            first = SimpleNamespace(echoframe_key=b'a')
            second = SimpleNamespace(echoframe_key=b'b')
            saved = SimpleNamespace(metadatas=[metadata('b', b'b'),
                metadata('a', b'a')])
            with (mock.patch.object(validate_lexicon, 'load_metadata_layer',
                return_value=[first, second]), mock.patch.object(
                validate_lexicon, 'make_vector_metadata',
                return_value=SimpleNamespace(path=path)), mock.patch.object(
                validate_lexicon.svs.VectorIndex, 'load', return_value=saved),
                mock.patch.object(validate_lexicon,
                'load_speech_vectors') as load_vectors):
                index, tokens = (
                    validate_lexicon.make_vector_search_index_for_layer(
                    object(), 6, 'w2v_6', pooling='mean', directory=directory))
                self.assertIs(index, saved)
                self.assertEqual(tokens, [second, first])
                load_vectors.assert_not_called()


class PrecisionTests(unittest.TestCase):
    def test_self_match_in_second_position_is_excluded(self):
        query = metadata('rok', b'query')
        result = {'metadatas': [metadata('rok', b'other'), query,
            metadata('dak', b'wrong')]}
        self.assertEqual(analyse_results.precision_at_k(result, query, 2), .5)

    def test_ties_can_omit_self_without_dropping_a_real_neighbour(self):
        query = metadata('rok', b'query')
        result = {'metadatas': [metadata('rok', b'first'),
            metadata('rok', b'second'), metadata('dak', b'third')]}
        self.assertEqual(analyse_results.precision_at_k(result, query, 2), 1.)

    def test_streamed_evaluation_groups_scores_by_word(self):
        tokens = [metadata('rok', b'a'), metadata('rok', b'b'),
            metadata('dak', b'c')]
        index = SimpleNamespace(metadatas=tokens,
            query_by_index=lambda i, top_k: {'metadatas': tokens})
        output = analyse_results.evaluate_index(index, top_k=2)
        self.assertEqual(output, {'rok': [.5, .5], 'dak': [0.]})

    def test_legacy_analysis_removes_self_by_key(self):
        query = metadata('rok', b'query')
        result = {'metadatas': [metadata('rok', b'other'), query,
            metadata('dak', b'wrong')]}
        index = SimpleNamespace(metadatas=[query])
        output = analyse_results.analyse_results({6: [result]},
            {6: [index, None]}, 6)
        self.assertEqual(output, {'rok': [.5]})


if __name__ == '__main__':
    unittest.main()
