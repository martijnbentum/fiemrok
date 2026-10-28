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

import analyse_results
import validate_lexicon


def metadata(label, key):
    return SimpleNamespace(label=label, source_echoframe_keys=[key])


class StoreLoadingTests(unittest.TestCase):
    def test_loaders_attach_current_shared_cgn_store(self):
        for loader in (validate_lexicon.load_store_w2v,
            validate_lexicon.load_store_hubert1):
            with self.subTest(loader=loader.__name__):
                with (mock.patch.object(validate_lexicon.store, 'Store')
                    as constructor, mock.patch.object(validate_lexicon.stores,
                    'cgn', None)):
                    shared = object()
                    validate_lexicon.stores.cgn = shared
                    loaded = loader(root='test-embeddings')
                    constructor.assert_called_once_with('test-embeddings')
                    self.assertIs(loaded, constructor.return_value)
                    loaded.attach_phraser_store.assert_called_once_with(
                        validate_lexicon.stores.CGN_SOURCE_ID, shared)

    def test_loaders_leave_phraser_unattached_when_cache_is_empty(self):
        for loader in (validate_lexicon.load_store_w2v,
            validate_lexicon.load_store_hubert1):
            with self.subTest(loader=loader.__name__):
                with (mock.patch.object(validate_lexicon.store, 'Store')
                    as constructor, mock.patch.object(validate_lexicon.stores,
                    'cgn', None), mock.patch.object(validate_lexicon.stores,
                    'load_phraser_cgn_store') as load_phraser):
                    loaded = loader()
                    self.assertIs(loaded, constructor.return_value)
                    loaded.attach_phraser_store.assert_not_called()
                    load_phraser.assert_not_called()


class PoolingTests(unittest.TestCase):
    def test_layer_runner_defaults_keep_word_stores_names_and_units(self):
        tokens = [SimpleNamespace(label='word', echoframe_key=b'word')]
        for model, loader in (('w2v', 'load_store_w2v'),
            ('hubert1', 'load_store_hubert1')):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                embedding_store = mock.Mock()
                embedding_store.load_many_frames.return_value = [np.array([1., 2.])]
                with (mock.patch.object(validate_lexicon, loader,
                    return_value=embedding_store) as load_store,
                    mock.patch.object(validate_lexicon, 'load_metadata_layer',
                    return_value=tokens)):
                    result = validate_lexicon.make_vsi_for_layers([6],
                        model=model, directory=directory)
                load_store.assert_called_once_with()
                index, returned = result[6]
                self.assertEqual(index.metadatas.name, f'{model}_6')
                self.assertEqual(index.metadatas.unit_type, 'word')
                self.assertEqual(returned, tokens)

    def test_marker_indexes_use_supplied_store_and_separate_pooling_names(self):
        tokens = [SimpleNamespace(label='ŋ r', echoframe_key=b'first'),
            SimpleNamespace(label='f d ə', echoframe_key=b'second')]
        embedding_store = mock.Mock()
        embedding_store.load_many_frames.return_value = [np.array([1., 2.]),
            np.array([2., 1.])]
        with tempfile.TemporaryDirectory() as directory:
            with (mock.patch.object(validate_lexicon, 'load_store_w2v',
                side_effect=AssertionError('must not open word store')),
                mock.patch.object(validate_lexicon, 'load_store_hubert1',
                side_effect=AssertionError('must not open word store')),
                mock.patch.object(validate_lexicon, 'load_metadata_layer',
                return_value=tokens)):
                for pooling, suffix in (('center', ''), ('mean', '_mean')):
                    result = validate_lexicon.make_vsi_for_layers([6],
                        model='hubert1', pooling=pooling, directory=directory,
                        embedding_store=embedding_store,
                        name_prefix='cross_boundary_', unit_type='marker')
                    index, returned = result[6]
                    name = f'cross_boundary_hubert1_6{suffix}'
                    self.assertEqual(index.metadatas.name, name)
                    self.assertEqual(index.metadatas.unit_type, 'marker')
                    self.assertEqual(returned, tokens)
                    loaded = validate_lexicon.svs.VectorIndex.load(directory, name)
                    self.assertEqual(loaded.metadatas.unit_type, 'marker')
                    self.assertEqual([md.label for md in loaded.metadatas],
                        ['ŋ r', 'f d ə'])
                    self.assertEqual([md.source_echoframe_keys
                        for md in loaded.metadatas], [[b'first'], [b'second']])
                    self.assertEqual(analyse_results.evaluate_index(loaded,
                        top_k=1), {'ŋ r': [0.], 'f d ə': [0.]})
        self.assertEqual([call.kwargs['frame'] for call in
            embedding_store.load_many_frames.call_args_list], ['center', 'mean'])

    def test_mean_and_center_use_different_frames(self):
        payload = np.array([[0., 0.], [9., 3.], [0., 0.]])
        token = SimpleNamespace(echoframe_key=b'token')
        embedding_store = mock.Mock()
        embedding_store.load_many_frames.side_effect = (
            lambda keys, frame, keep_missing:
                [validate_lexicon.svs.util_math.pool_frames(payload, frame)])
        mean = validate_lexicon.load_speech_vectors(embedding_store,
            [token], pooling='mean')
        center = validate_lexicon.load_speech_vectors(embedding_store, [token])
        np.testing.assert_array_equal(mean, [[3., 1.]])
        np.testing.assert_array_equal(center, [[9., 3.]])

    def test_invalid_pooling_is_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            validate_lexicon.load_speech_vectors(object(), [], pooling='median')

    def test_mean_index_has_separate_name_and_honours_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            vector_metadata = SimpleNamespace(
                path=pathlib.Path(directory) / 'w2v_6_mean.json')
            token = SimpleNamespace(echoframe_key=b'a')
            embedding_store = object()
            with (mock.patch.object(validate_lexicon, 'load_metadata_layer',
                return_value=[token]), mock.patch.object(
                validate_lexicon, 'make_vector_metadata',
                return_value=vector_metadata) as make_metadata, mock.patch.object(
                validate_lexicon, 'load_speech_vectors') as load_vectors,
                mock.patch.object(validate_lexicon,
                'make_vector_search_index') as make_index):
                validate_lexicon.make_vector_search_index_for_layer(
                    embedding_store, 6, 'w2v_6', pooling='mean',
                    directory=directory, overwrite=True)
                make_metadata.assert_called_once_with([token],
                    name='w2v_6_mean',
                    directory=directory, unit_type='word')
                load_vectors.assert_called_once_with(embedding_store,
                    [token], pooling='mean')
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


class IndexReuseTests(unittest.TestCase):
    def test_fast_reuse_does_not_load_echoframe_metadata_or_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.json', '.npy'):
                (pathlib.Path(directory) / f'hubert1_6_mean{suffix}').touch()
            with (mock.patch.object(validate_lexicon.svs.VectorIndex, 'load')
                as load_index, mock.patch.object(validate_lexicon,
                'load_metadata_layer', side_effect=AssertionError(
                    'metadata should not be read')), mock.patch.object(
                validate_lexicon, 'make_vector_metadata',
                side_effect=AssertionError('labels should not be resolved'))):
                index, tokens = (
                    validate_lexicon.make_vector_search_index_for_layer(
                    object(), 6, 'hubert1_6', pooling='mean',
                    directory=directory, return_metadata=False))
                self.assertIs(index, load_index.return_value)
                self.assertIsNone(tokens)
                load_index.assert_called_once_with(directory, 'hubert1_6_mean')

    def test_partial_index_fails_before_metadata_loading(self):
        for suffix in ('.json', '.npy'):
            with self.subTest(suffix=suffix):
                with tempfile.TemporaryDirectory() as directory:
                    filename = f'hubert1_6_mean{suffix}'
                    (pathlib.Path(directory) / filename).touch()
                    with mock.patch.object(validate_lexicon,
                        'load_metadata_layer', side_effect=AssertionError(
                            'metadata should not be read')):
                        with self.assertRaisesRegex(ValueError,
                            'incomplete index'):
                            validate_lexicon.make_vector_search_index_for_layer(
                                object(), 6, 'hubert1_6', pooling='mean',
                                directory=directory, return_metadata=False)

    def test_new_index_can_omit_metadata_and_skip_saving(self):
        tokens = [SimpleNamespace(echoframe_key=b'a', label='word')]
        store = mock.Mock()
        store.load_many_frames.return_value = [np.array([1., 2.])]
        with tempfile.TemporaryDirectory() as directory:
            with (mock.patch.object(validate_lexicon, 'load_metadata_layer',
                return_value=tokens), mock.patch.object(validate_lexicon,
                'make_vector_search_index') as make_index):
                index, returned_tokens = (
                    validate_lexicon.make_vector_search_index_for_layer(
                    store, 6, 'hubert1_6', pooling='mean', directory=directory,
                    save=False, return_metadata=False))
                self.assertIs(index, make_index.return_value)
                self.assertIsNone(returned_tokens)
                index.save.assert_not_called()
                np.testing.assert_array_equal(
                    make_index.call_args.args[0], [[1., 2.]])
                vector_metadata = make_index.call_args.args[1]
                self.assertEqual(vector_metadata.name, 'hubert1_6_mean')
                self.assertEqual(vector_metadata.directory, directory)
                self.assertEqual(vector_metadata[0].source_echoframe_keys,
                    [b'a'])

    def test_missing_layer_is_reported_without_loading_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(validate_lexicon, 'load_metadata_layer',
                return_value=[]):
                store = mock.Mock()
                with self.assertRaisesRegex(ValueError, 'no embeddings'):
                    validate_lexicon.make_vector_search_index_for_layer(
                        store, 9, 'hubert1_9', directory=directory)
                store.load_many_frames.assert_not_called()

    def test_saved_index_round_trip_keeps_vector_and_metadata_alignment(self):
        tokens = [SimpleNamespace(echoframe_key=b'a', label='first'),
            SimpleNamespace(echoframe_key=b'b', label='second')]
        store = mock.Mock()
        store.load_many_frames.return_value = [np.array([3., 4.]),
            np.array([4., 3.])]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(validate_lexicon, 'load_metadata_layer',
                return_value=tokens):
                built, returned = (
                    validate_lexicon.make_vector_search_index_for_layer(
                    store, 6, 'hubert1_6', pooling='mean', directory=directory,
                    return_metadata=False))
                self.assertIsNone(returned)
            store.load_many_frames.assert_called_once_with([b'a', b'b'],
                frame='mean', keep_missing=True)
            store.load_many_frames.reset_mock()
            with (mock.patch.object(validate_lexicon, 'load_metadata_layer',
                return_value=list(reversed(tokens))), mock.patch.object(
                validate_lexicon, 'make_vector_metadata',
                side_effect=AssertionError('saved labels should be reused'))):
                loaded, returned = (
                    validate_lexicon.make_vector_search_index_for_layer(
                    store, 6, 'hubert1_6', pooling='mean', directory=directory))
            np.testing.assert_allclose(loaded.vectors, built.vectors)
            self.assertEqual([md.label for md in loaded.metadatas],
                ['first', 'second'])
            self.assertEqual(returned, tokens)
            store.load_many_frames.assert_not_called()


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
