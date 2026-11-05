from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from echoframe import key_helper
import h5py
import numpy as np
import speech_vector_search as svs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import neighbour_results as nr


def make_index(directory, model_id=1, layer=3, order=None, labels=None):
    '''Use non-contiguous types, identical vectors, ties and negative scores.'''
    vectors = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.],
        [0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    labels = labels or ['huis', 'boom', 'huis', 'boom', 'huis', 'boom']
    selected = [bytes([i + 1]) * 22 for i in range(len(vectors))]
    order = list(range(len(vectors))) if order is None else order
    records = [svs.metadata.VectorMetadata(labels[i], 'word',
        [key_helper.pack_hidden_state_key(model_id, layer, selected[i], 500)],
        0) for i in order]
    name = f'model{model_id}_word_2_{layer}_center'
    metadata = svs.metadata.VectorMetadatas(records, name, str(directory))
    index = svs.VectorIndex(vectors[order], metadata)
    index.save(overwrite=True)
    return index, [selected[i] for i in order]


def reference(index, token_index, top_k):
    '''Exhaustive cosine reference with an explicit deterministic tie rule.'''
    vectors = index.vectors.astype(np.float32)
    query = index.vectors[token_index]
    norm = np.linalg.norm(query)
    if norm: query = query / norm
    scores = vectors @ query.astype(np.float32)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    neighbours = [i for i in order if i != token_index][:top_k]
    label = index.metadatas[token_index].label
    same_type = [i for i, md in enumerate(index.metadatas) if md.label == label]
    ranks = [order.index(i) for i in same_type]
    return neighbours, scores[neighbours], same_type, ranks, scores[same_type]


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.index_directory = self.workspace / 'svs_data'
        self.filename = (self.workspace / 'neighbours' / 'word' / 'hubert' /
            'layer03_center.h5')
        self.mapping = self.filename.parent.parent / 'token_mapping.npz'
        self.index, self.selected = make_index(self.index_directory)

    def write(self, **kwargs):
        options = dict(top_k=3, batch_size=2, workspace=self.workspace)
        options.update(kwargs)
        return nr.write_neighbours(self.index, self.filename, self.mapping,
            self.selected, **options)

    def test_round_trip_matches_exhaustive_ranking_including_ties_and_self(self):
        self.assertEqual(self.write(), self.filename)
        with nr.Result(self.filename, index=self.index) as results:
            self.assertEqual(len(results), len(self.selected))
            for i in range(len(results)):
                expected = reference(self.index, i, 3)
                query = results[i]
                self.assertEqual(query.token_index, i)
                self.assertEqual(query.token.label,
                    self.index.metadatas[i].label)
                for actual, wanted in zip((query.neighbour_indices,
                    query.neighbour_similarities, query.same_type_indices,
                    query.same_type_ranks, query.same_type_similarities),
                    expected):
                    np.testing.assert_allclose(actual, wanted, atol=1e-7)
                self.assertNotIn(i, query.neighbour_indices)
                self.assertEqual(list(query.same_type_indices).count(i), 1)
                label = query.token.label
                same_label = [self.index.metadatas[int(j)].label == label
                    for j in query.neighbour_indices[:2]]
                self.assertEqual(query.precision(2), np.mean(same_label))
            # Query 2 ties with token 0: self is included, not forced to rank 0.
            self.assertEqual(results[2].same_type_ranks[1], 1)

    def test_numeric_arrays_are_uint32_and_float32_and_support_batch_reads(self):
        self.write()
        with nr.Result(self.filename, index=self.index) as results:
            batch = results.read(1, 4)
            self.assertEqual(set(batch), {'neighbour_indices',
                'neighbour_similarities', 'same_type_ranks',
                'same_type_similarities'})
            for name, values in batch.items():
                self.assertEqual(values.shape, (3, 3))
                expected_dtype = np.float32 if 'similarities' in name else np.uint32
                self.assertEqual(values.dtype, expected_dtype)
            np.testing.assert_array_equal(batch['same_type_ranks'][0],
                results[1].same_type_ranks)
            with self.assertRaises((ValueError, IndexError)):
                results[0].precision(4)

    def test_batch_size_does_not_change_results(self):
        self.write(batch_size=1)
        with h5py.File(self.filename) as handle:
            original = {name: handle[name][:] for name in (
                'neighbour_indices', 'neighbour_similarities',
                'same_type_ranks', 'same_type_similarities')}
        self.write(batch_size=100, overwrite=True)
        with h5py.File(self.filename) as handle:
            for name, values in original.items():
                np.testing.assert_array_equal(handle[name][:], values)

    def test_object_graph_shares_token_and_type_identities(self):
        self.write()
        with nr.Result(self.filename, index=self.index) as result:
            self.assertEqual(result._tokens, {})
            self.assertEqual(result._types, {})
            self.assertEqual([item.label for item in result.types],
                ['huis', 'boom'])
            query = result[2]
            token = query.token
            self.assertIsInstance(query, nr.Query)
            self.assertIsInstance(token, nr.Token)
            self.assertIs(token, result.token(2))
            self.assertIs(token, result[2].token)
            self.assertIs(token.type, result.type('huis'))
            self.assertIs(token.type, result.types[0])
            self.assertIsInstance(token.type, nr.Type)
            self.assertEqual(len(token.type), 3)
            self.assertIsInstance(token.type.tokens, tuple)
            self.assertEqual([item.index for item in token.type.tokens],
                [0, 2, 4])
            self.assertIs(token.type.tokens[1], token)
            self.assertEqual(token.phraser_key, self.selected[2])
            self.assertEqual(token.source_key,
                self.index.metadatas[2].source_echoframe_keys[0])
            self.assertEqual([q.token.index for q in result], list(range(6)))
            with self.assertRaisesRegex(ValueError,
                "label not in lexicon: 'missing'; expected: 'huis', 'boom'"):
                result.type('missing')
            with self.assertRaises(IndexError):
                result.token(-1)

    def test_query_access_is_lazy_and_shares_cached_queries(self):
        self.write()
        with nr.Result(self.filename, index=self.index) as result:
            self.assertEqual(len(result), 6)
            self.assertEqual(result._queries, {})
            query = result[2]
            self.assertIs(query, result[2])
            self.assertEqual(list(result._queries), [2])
            self.assertIsNone(query._same)
            self.assertIsNone(query._neighbours)
            self.assertIs(result[2].same, result[2].same)
            self.assertIs(result[2].neighbours, result[2].neighbours)
            queries = tuple(result)
            self.assertEqual([q.token_index for q in queries], list(range(6)))
            self.assertIs(queries[2], query)
            for index in [-1, 6, True, 1.5]:
                with self.assertRaises(IndexError):
                    result[index]
        self.assertEqual(result._queries, {})
        for lookup in [lambda: result[0], lambda: next(iter(result)),
            lambda: result.collect_queries()]:
            with self.assertRaisesRegex(ValueError, 'closed'):
                lookup()

    def test_collect_queries_loads_matches_once_without_annotations(self):
        self.write()
        store = mock.Mock()
        with nr.Result(self.filename, index=self.index, store=store) as result:
            query = result[2]
            self.assertIsNone(result.collect_queries())
            queries = tuple(result)
            self.assertEqual([q.token_index for q in queries], list(range(6)))
            self.assertIs(queries[2], query)
            for q in queries:
                self.assertIsNotNone(q._same)
                self.assertIsNotNone(q._neighbours)
            self.assertEqual(result._tokens, {})
            with mock.patch.object(nr.Query, '_array',
                side_effect=AssertionError('matches should be cached')):
                self.assertIsNone(result.collect_queries())
                for original, cached in zip(queries, result):
                    self.assertIs(original, cached)
                    self.assertIs(original.same, cached.same)
                    self.assertIs(original.neighbours, cached.neighbours)
            store.load_metadata.assert_not_called()
        self.assertIsNone(query._same)
        self.assertIsNone(query._neighbours)
        result.close()

    def test_reprs_are_lazy_and_handle_closed_results(self):
        self.write()
        with nr.Result(self.filename, index=self.index) as result:
            query = result[0]
            token = query.token
            type_ = token.type
            with mock.patch.object(h5py.Dataset, '__getitem__',
                side_effect=AssertionError('repr must not read arrays')):
                self.assertEqual(repr(result), "<Result model='hubert' "
                    "condition='word' layer=3 pooling='center' tokens=6>")
                self.assertEqual(repr(query), "<Query index=0 label='huis'>")
                self.assertEqual(repr(token), "<Token index=0 label='huis'>")
                self.assertEqual(repr(type_), "<Type label='huis' tokens=3>")
            self.assertIsNone(query._same)
            self.assertIsNone(query._neighbours)
            self.assertIsNone(type_._tokens)
            self.assertEqual(list(result._tokens), [0])
            self.assertIsNone(result._store)
            match = query.neighbours[0]
            tokens_before = dict(result._tokens)
            with mock.patch.object(h5py.Dataset, '__getitem__',
                side_effect=AssertionError('repr must not read arrays')):
                self.assertEqual(repr(query), "<Query index=0 label='huis'>")
                self.assertEqual(repr(match), "<Match index=2 label='huis' "
                    'rank=1 similarity=1.0000>')
            self.assertEqual(result._tokens, tokens_before)
        for obj, expected in [(result, '<Result closed>'),
            (query, '<Query index=0 closed>'),
            (token, '<Token index=0 closed>'), (type_, '<Type index=0 closed>'),
            (match, '<Match index=2 closed>')]:
            self.assertEqual(repr(obj), expected)

    def test_matches_have_global_ranks_including_tied_self(self):
        self.write()
        with nr.Result(self.filename, index=self.index) as result:
            for query in result:
                expected = reference(self.index, query.token_index, 3)
                neighbours, scores, indices, ranks, similarities = expected
                self.assertIsInstance(query.neighbours, tuple)
                self.assertIsInstance(query.same, tuple)
                self.assertEqual([m.token.index for m in query.neighbours],
                    list(neighbours))
                np.testing.assert_allclose(
                    [m.similarity for m in query.neighbours], scores)
                self.assertEqual([m.token.index for m in query.same],
                    [indices[i] for i in np.argsort(ranks)])
                self.assertEqual([m.rank for m in query.same], sorted(ranks))
                np.testing.assert_allclose([m.similarity for m in query.same],
                    similarities[np.argsort(ranks)])
                vector = self.index.vectors[query.token_index]
                all_scores = self.index.vectors @ vector
                order = sorted(range(6), key=lambda i: (-all_scores[i], i))
                self.assertEqual([m.rank for m in query.neighbours],
                    [order.index(int(i)) for i in neighbours])
                for match in query.neighbours + query.same:
                    self.assertIsInstance(match, nr.Match)
                    self.assertIs(match.query, query)
                    self.assertIs(match.token, result.token(match.token.index))
                for match in query.neighbours:
                    same = [m for m in query.same if m.token is match.token]
                    if same: self.assertIs(match, same[0])
            # Query 2 ties with token 0, so its first neighbour has rank zero.
            self.assertEqual(result[2].neighbours[0].rank, 0)
            self.assertEqual(result[0].neighbours[0].rank, 1)

    def test_self_can_rank_beyond_all_saved_neighbours(self):
        self.index.vectors[5] = 0
        self.write()
        with nr.Result(self.filename, index=self.index) as result:
            query = result[5]
            self.assertEqual([m.rank for m in query.neighbours], [0, 1, 2])
            self.assertEqual([m.rank for m in query.same], [1, 3, 5])
            self.assertEqual(query.same[-1].token.index, 5)

    def test_general_metadata_and_cached_views_respect_result_lifetime(self):
        self.write()
        result = nr.Result(self.filename, index=self.index)
        with result:
            self.assertEqual(result.model, 'hubert')
            self.assertEqual(result.condition, 'word')
            self.assertEqual(result.layer, 3)
            self.assertEqual(result.pooling, 'center')
            self.assertEqual(result.attributes['index_name'],
                self.index.metadatas.name)
            query = result[0]
            token = query.token
            type_ = token.type
            type_.tokens
            match = query.same[0]
            query.neighbours
        for lookup in [lambda: query.same, lambda: query.neighbours,
            lambda: query.token, lambda: token.label, lambda: token.source_key,
            lambda: type_.tokens, lambda: type_.label, lambda: len(type_),
            lambda: result.type('huis'), lambda: query.same_type_indices,
            lambda: query.neighbour_indices, lambda: token.metadata,
            lambda: match.token,
            lambda: match.rank, lambda: match.similarity,
            lambda: result.types, lambda: result.attributes]:
            with self.subTest(lookup=lookup):
                with self.assertRaisesRegex(ValueError, 'closed'):
                    lookup()

    def test_tokens_use_lazy_annotation_lookup_and_context(self):
        self.write()
        store = mock.Mock()
        with nr.Result(self.filename, index=self.index, store=store) as result:
            query = result[0]
            query.same
            query.neighbours
            query.token.phraser_key
            store.load_metadata.assert_not_called()
            self.assertIs(query.token.metadata, store.load_metadata.return_value)
            store.load_metadata.assert_called_once_with(query.token.source_key)
            with mock.patch.object(result, 'context', return_value={}) as context:
                self.assertEqual(query.token.context(4, 1), {})
                context.assert_called_once_with(0, 4, 1)
        store.close.assert_not_called()

    def test_reader_loads_metadata_without_loading_vectors_or_annotation_store(self):
        self.write()
        original_load = np.load

        def load_mapping_only(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                self.assertNotEqual(Path(path).suffix, '.npy')
            return original_load(path, *args, **kwargs)

        with mock.patch.object(np, 'load', side_effect=load_mapping_only):
            with nr.Result(self.filename) as results:
                self.assertEqual(results[2].token.label, 'huis')
                np.testing.assert_array_equal(results[2].same_type_indices,
                    [0, 2, 4])
                self.assertEqual(results[0].precision(1), 1.)

    def test_shared_mapping_is_reused_across_model_and_layer_keys(self):
        self.write()
        before = self.mapping.read_bytes()
        other, keys = make_index(self.index_directory, model_id=2, layer=6)
        other_path = self.filename.with_name('layer06_center.h5')
        nr.write_neighbours(other, other_path, self.mapping, keys, top_k=3,
            batch_size=2, workspace=self.workspace)
        self.assertEqual(before, self.mapping.read_bytes())
        with nr.Result(other_path, index=other) as results:
            np.testing.assert_array_equal(results[1].same_type_indices,
                [1, 3, 5])
        with self.assertRaises(ValueError):
            nr.Result(other_path, index=self.index)

    def test_same_source_keys_in_a_different_pooling_index_are_rejected(self):
        self.write()
        metadata = svs.metadata.VectorMetadatas(list(self.index.metadatas),
            self.index.metadatas.name.replace('_center', '_mean'),
            str(self.index_directory))
        other = svs.VectorIndex(self.index.vectors, metadata)
        with self.assertRaisesRegex(ValueError, 'different vector index'):
            nr.Result(self.filename, index=other)

    def test_duplicate_metadata_token_identities_are_rejected(self):
        self.index.metadatas.metadatas[1] = self.index.metadatas[0]
        self.selected[1] = self.selected[0]
        with self.assertRaisesRegex(ValueError, 'duplicate Phraser'):
            self.write()
        self.assertFalse(self.filename.exists())

    def test_changed_order_or_labels_cannot_reuse_mapping(self):
        self.write()
        for options in [dict(order=[1, 0, 2, 3, 4, 5]),
            dict(labels=['boom', 'huis', 'huis', 'boom', 'huis', 'boom'])]:
            with self.subTest(options=options):
                other, keys = make_index(self.index_directory, model_id=2,
                    **options)
                other_path = self.filename.with_name('other.h5')
                with self.assertRaises(ValueError):
                    nr.write_neighbours(other, other_path, self.mapping, keys,
                        top_k=3, workspace=self.workspace)
                self.assertFalse(other_path.exists())

    def test_selection_must_match_index_order_without_duplicates(self):
        for keys in [list(reversed(self.selected)), self.selected[:-1],
            [self.selected[0]] * len(self.selected)]:
            with self.subTest(keys=keys):
                with self.assertRaises(ValueError):
                    nr.write_neighbours(self.index, self.filename, self.mapping,
                        keys, top_k=3, workspace=self.workspace)
                self.assertFalse(self.filename.exists())

    def test_nonuniform_type_counts_are_rejected(self):
        index, keys = make_index(self.index_directory,
            labels=['huis', 'boom', 'huis', 'huis', 'huis', 'boom'])
        with self.assertRaises(ValueError):
            nr.write_neighbours(index, self.filename, self.mapping, keys,
                top_k=3, workspace=self.workspace)
        self.assertFalse(self.filename.exists())

    def test_invalid_search_sizes_fail_without_publishing_results(self):
        for options in [dict(top_k=0), dict(top_k=6), dict(top_k=True),
            dict(batch_size=0), dict(batch_size=True)]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.write(**options)
                self.assertFalse(self.filename.exists())

    def test_invalid_index_name_fails_before_creating_output_files(self):
        for name in ['idx', 'idx_custom', 'idx_center_extra', 'idxcenter']:
            with self.subTest(name=name):
                self.index.metadatas.name = name
                with self.assertRaisesRegex(ValueError,
                    'index name must end in _center or _mean'):
                    self.write()
                self.assertFalse(self.filename.parent.exists())
                self.assertFalse(self.mapping.exists())

    def test_recognized_pooling_suffixes_round_trip(self):
        for pooling in ['center', 'mean']:
            with self.subTest(pooling=pooling):
                self.index.metadatas.name = f'idx_{pooling}'
                self.write(overwrite=True)
                with nr.Result(self.filename, index=self.index) as result:
                    self.assertEqual(result.pooling, pooling)
                    self.assertEqual(len(result[0].neighbours), 3)

    def test_existing_output_is_preserved_without_explicit_overwrite(self):
        self.write()
        original = self.filename.read_bytes()
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(self.filename.read_bytes(), original)
        self.write(overwrite=True)
        with nr.Result(self.filename, index=self.index) as results:
            self.assertEqual(len(results), 6)

    def test_nonfinite_vectors_do_not_publish_or_replace_results(self):
        self.write()
        original = self.filename.read_bytes()
        self.index.vectors[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.write(overwrite=True)
        self.assertEqual(self.filename.read_bytes(), original)
        self.assertEqual(list(self.filename.parent.glob('*.tmp')), [])

    def test_reader_rejects_inconsistent_numeric_array_shapes(self):
        self.write()
        with h5py.File(self.filename, 'r+') as handle:
            del handle['same_type_ranks']
            handle.create_dataset('same_type_ranks', (6, 2), dtype='uint32')
        with self.assertRaises(ValueError):
            nr.Result(self.filename, index=self.index)

    def test_metadata_is_lazy_and_caller_store_remains_open(self):
        self.write()
        store = mock.Mock()
        metadata = SimpleNamespace(phraser_object=object())
        store.load_metadata.return_value = metadata
        with nr.Result(self.filename, index=self.index,
            store=store) as results:
            store.load_metadata.assert_not_called()
            self.assertIs(results[4].token.metadata, metadata)
            store.load_metadata.assert_called_once_with(
                self.index.metadatas[4].source_echoframe_keys[0])
        store.close.assert_not_called()
        store.close_phraser_stores.assert_not_called()

    def test_owned_store_opens_lazily_and_closes_with_attached_materials(self):
        self.write()
        store = mock.Mock()
        store.load_metadata.return_value = SimpleNamespace(phraser_object=None)
        with mock.patch('lexicon_validation.lv_stores.load_echoframe_store',
            return_value=store) as loader:
            with nr.Result(self.filename, index=self.index) as results:
                loader.assert_not_called()
                results.metadata(0)
                loader.assert_called_once()
                results.metadata(1)
                loader.assert_called_once()
            store.close.assert_called_once()
            store.close_phraser_stores.assert_called_once()

    def test_missing_metadata_raises_and_context_reuses_existing_helper(self):
        self.write()
        store = mock.Mock()
        store.load_metadata.return_value = None
        with nr.Result(self.filename, index=self.index,
            store=store) as results:
            with self.assertRaises(ValueError):
                results.metadata(0)
            with mock.patch(
                'lexicon_validation.neighbour_metadata.echoframe_key_to_metadata',
                return_value={'context': True}) as context:
                self.assertEqual(results.context(2, n_phone_context=4,
                    n_word_context=1), {'context': True})
                self.assertEqual(context.call_args.args[:2],
                    (self.index.metadatas[2].source_echoframe_keys[0], store))


if __name__ == '__main__':
    unittest.main()
