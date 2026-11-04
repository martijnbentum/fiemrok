import pathlib
import sys
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import echoframe
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lexicon_validation import analyse_results
from lexicon_validation import build_speech_vector_index as validate_lexicon


def metadata(label, key):
    return SimpleNamespace(label=label, source_echoframe_keys=[key])


class SelectionTests(unittest.TestCase):
    def token(self, key, layer=6, model='hubert', collar=500,
        output_type='hidden_state'):
        return SimpleNamespace(phraser_key=key, echoframe_key=b'e' + key,
            label=key.hex(), layer=layer, model_name=model, collar=collar,
            output_type=output_type)

    def test_selected_order_and_extraction_condition_are_respected(self):
        first = self.token(b'a')
        second = self.token(b'b')
        records = [second, self.token(b'a', layer=9),
            self.token(b'a', model='w2v'), self.token(b'a', collar=0),
            self.token(b'a', output_type='attention'), first,
            self.token(b'not-selected')]
        store = SimpleNamespace(metadatas=records)
        actual = validate_lexicon.load_metadata_layer(store, 6,
            [b'a', b'b'], 'hubert')
        self.assertEqual(actual, [first, second])

    def test_missing_selected_token_is_not_silently_dropped(self):
        store = SimpleNamespace(metadatas=[self.token(b'a')])
        with self.assertRaisesRegex(ValueError, 'missing embedding'):
            validate_lexicon.load_metadata_layer(store, 6,
                [b'a', b'b'], 'hubert')

    def test_ambiguous_embeddings_are_reported(self):
        store = SimpleNamespace(metadatas=[self.token(b'a')] * 2)
        with self.assertRaisesRegex(ValueError, 'multiple embeddings'):
            validate_lexicon.load_metadata_layer(store, 6, [b'a'], 'hubert')


class FreshIndexTests(unittest.TestCase):
    def make_store(self):
        store = mock.Mock()
        store.metadatas = [SimpleNamespace(phraser_key=k,
            echoframe_key=b'e' + k, label=k.hex(), layer=6,
            model_name='hubert', collar=500, output_type='hidden_state')
            for k in (b'a', b'b')]
        payloads = {b'ea': np.array([3., 4.]),
            b'eb': np.array([4., 3.])}
        store.load_many_frames.side_effect = lambda keys, **kw: [
            payloads[k] for k in keys]
        return store

    def test_saved_index_preserves_selection_order_keys_units_and_name(self):
        store = self.make_store()
        with tempfile.TemporaryDirectory() as directory:
            index, tokens = validate_lexicon.make_vector_search_index_for_layer(
                store, 6, 'hubert_cross_boundary_2_6_mean', [b'b', b'a'],
                'hubert', directory=pathlib.Path(directory), pooling='mean',
                unit_type='marker')
            loaded = validate_lexicon.svs.VectorIndex.load(directory,
                'hubert_cross_boundary_2_6_mean')
        self.assertEqual([x.phraser_key for x in tokens], [b'b', b'a'])
        self.assertEqual(loaded.metadatas.unit_type, 'marker')
        self.assertEqual([md.source_echoframe_keys for md in loaded.metadatas],
            [[b'eb'], [b'ea']])
        self.assertEqual([md.label for md in loaded.metadatas], ['62', '61'])
        np.testing.assert_allclose(loaded.vectors, index.vectors)
        store.load_many_frames.assert_called_once_with([b'eb', b'ea'],
            frame='mean', keep_missing=True)

    def test_existing_files_are_rebuilt_only_with_explicit_overwrite(self):
        store = self.make_store()
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(directory=directory)
            validate_lexicon.make_vector_search_index_for_layer(store, 6,
                'hubert_word_2_6_center', [b'a', b'b'], 'hubert', **kwargs)
            store.load_many_frames.reset_mock()
            with self.assertRaises(FileExistsError):
                validate_lexicon.make_vector_search_index_for_layer(store, 6,
                    'hubert_word_2_6_center', [b'a', b'b'], 'hubert', **kwargs)
            validate_lexicon.make_vector_search_index_for_layer(store, 6,
                'hubert_word_2_6_center', [b'b', b'a'], 'hubert',
                overwrite=True, **kwargs)
            self.assertEqual(store.load_many_frames.call_count, 1)
            loaded = validate_lexicon.svs.VectorIndex.load(directory,
                'hubert_word_2_6_center')
            self.assertEqual(loaded.metadatas[0].source_echoframe_keys, [b'eb'])

    def test_layer_runner_uses_explicit_names_without_duplicating_pooling(self):
        store = self.make_store()
        with mock.patch.object(validate_lexicon,
            'make_vector_search_index_for_layer') as build:
            validate_lexicon.make_vsi_for_layers([3, 6], store,
                [b'a', b'b'], 'hubert', name_prefix='hubert_word_2_',
                pooling='mean', save=False)
        self.assertEqual([call.args[2] for call in build.call_args_list],
            ['hubert_word_2_3_mean', 'hubert_word_2_6_mean'])
        self.assertTrue(all(call.args[3] == [b'a', b'b']
            for call in build.call_args_list))

    def test_empty_selection_fails_without_frame_loading(self):
        store = self.make_store()
        with self.assertRaisesRegex(ValueError, 'no embeddings selected'):
            validate_lexicon.make_vector_search_index_for_layer(store, 6,
                'empty', [], 'hubert', save=False)
        store.load_many_frames.assert_not_called()


class BatchLoadingTests(unittest.TestCase):
    def test_one_batch_preserves_key_order_and_duplicates(self):
        store = mock.Mock()
        store.load_many_frames.return_value = [np.array([4, 5]),
            np.array([1, 2]), np.array([4, 5])]
        tokens = [SimpleNamespace(echoframe_key=key)
            for key in (b'b', b'a', b'b')]
        vectors = validate_lexicon.load_speech_vectors(store, tokens,
            pooling='mean')
        store.load_many_frames.assert_called_once_with([b'b', b'a', b'b'],
            frame='mean', keep_missing=True)
        np.testing.assert_array_equal(vectors, [[4, 5], [1, 2], [4, 5]])
        self.assertEqual(vectors.dtype, np.dtype(float))

    def test_missing_or_misaligned_results_fail(self):
        tokens = [SimpleNamespace(echoframe_key=b'a'),
            SimpleNamespace(echoframe_key=b'b')]
        for result in ([np.ones(2), None], [np.ones(2)],
            [np.ones(2)] * 3):
            with self.subTest(result=result):
                store = mock.Mock()
                store.load_many_frames.return_value = result
                with self.assertRaisesRegex(ValueError, 'missing vectors'):
                    validate_lexicon.load_speech_vectors(store, tokens)

    def test_empty_requests_do_not_access_store(self):
        store = mock.Mock()
        vectors = validate_lexicon.load_speech_vectors(store, [])
        self.assertEqual(vectors.shape, (0,))
        self.assertEqual(vectors.dtype, np.dtype(float))
        store.load_many_frames.assert_not_called()

    def test_real_shards_are_opened_once_and_use_echoframe_pooling(self):
        payloads = [np.array([[1e8, 2], [1, 4], [-1e8, 6]], dtype='float32'),
            np.array([[2, 1], [4, 3]], dtype='float32')]
        with tempfile.TemporaryDirectory() as directory:
            store = echoframe.Store(directory, max_shard_items=1)
            try:
                store.register_model('hubert')
                tokens = []
                for i, payload in enumerate(payloads):
                    key = store.make_echoframe_key('hidden_state',
                        model_name='hubert', layer=6,
                        phraser_key=i.to_bytes(22, 'big'), collar=0)
                    token = echoframe.EchoframeMetadata(key,
                        model_name='hubert', phraser_source_id='cgn')
                    tokens.append(store.save(key, token, payload))
                self.assertNotEqual(tokens[0].shard_id, tokens[1].shard_id)
                original_file = store.storage.h5.File
                opened = []

                def counting_file(filename, mode):
                    self.assertEqual(mode, 'r')
                    opened.append(str(filename))
                    return original_file(filename, mode)

                for pooling in ('mean', 'center'):
                    with self.subTest(pooling=pooling):
                        opened.clear()
                        h5 = SimpleNamespace(File=counting_file)
                        with mock.patch.object(store.storage, 'h5', h5):
                            vectors = validate_lexicon.load_speech_vectors(
                                store, [tokens[1], tokens[0], tokens[1]],
                                pooling=pooling)
                        self.assertEqual(sorted(Counter(opened).values()),
                            [1, 1])
                        if pooling == 'mean':
                            expected = [p.mean(axis=0) for p in payloads]
                        else:
                            expected = [p[len(p) // 2] for p in payloads]
                        np.testing.assert_array_equal(vectors,
                            [expected[1], expected[0], expected[1]])
                        self.assertEqual(vectors.dtype, np.dtype(float))
            finally:
                store.close()


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
        output = analyse_results.evaluate_index(index, top_k=2, batch_size=1)
        self.assertEqual(output, {'rok': [.5, .5], 'dak': [0.]})

    def test_batched_evaluation_matches_individual_queries(self):
        svs = validate_lexicon.svs
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(37, 16))
        tokens = [svs.metadata.VectorMetadata(str(i % 3), 'word',
            [i.to_bytes(2, 'big')], 0) for i in range(len(vectors))]
        metadatas = svs.metadata.VectorMetadatas(tokens, 'batch_test', '.')
        index = svs.VectorIndex(vectors, metadatas)
        individual = analyse_results.evaluate_index(index, batch_size=1)
        batched = analyse_results.evaluate_index(index, batch_size=8)
        oversized = analyse_results.evaluate_index(index, batch_size=100)
        self.assertEqual(batched, individual)
        self.assertEqual(oversized, individual)
        self.assertEqual(sum(map(len, batched.values())), 37)

    def test_legacy_analysis_removes_self_by_key(self):
        query = metadata('rok', b'query')
        neighbours = [metadata('rok', bytes([i])) for i in range(4)]
        neighbours += [metadata('dak', bytes([i])) for i in range(4, 10)]
        neighbours.insert(1, query)
        result = {'metadatas': neighbours}
        index = SimpleNamespace(metadatas=[query])
        output = analyse_results.analyse_results({6: [result]},
            {6: [index, None]}, 6)
        self.assertEqual(output, {'rok': [.4]})

    def test_saved_results_without_self_score_all_ten_neighbours(self):
        query = metadata('rok', b'query')
        neighbours = [metadata('dak', bytes([i])) for i in range(9)]
        neighbours.append(metadata('rok', b'tenth'))
        index = SimpleNamespace(metadatas=[query])
        output = analyse_results.analyse_results(
            {6: [{'metadatas': neighbours}]}, {6: [index, None]}, 6)
        self.assertEqual(output, {'rok': [.1]})

    def test_saved_results_with_only_nine_non_self_neighbours_fail(self):
        query = metadata('rok', b'query')
        neighbours = [query]
        neighbours += [metadata('rok', bytes([i])) for i in range(9)]
        index = SimpleNamespace(metadatas=[query])
        with self.assertRaisesRegex(ValueError, '10 non-self neighbours'):
            analyse_results.analyse_results(
                {6: [{'metadatas': neighbours}]}, {6: [index, None]}, 6)


if __name__ == '__main__':
    unittest.main()
