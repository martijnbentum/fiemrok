from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from echoframe import key_helper
import numpy as np
import speech_vector_search as svs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import lv_locations
from lexicon_validation import neighbour_results
from lexicon_validation import retrieval_evaluation


def save_index(workspace, model='hubert', condition='word', layer=3,
    pooling='center', directory=None, n_types=3):
    '''Save the current pipeline's index name and ordered dataset definition.'''
    directory = workspace / 'svs_data' if directory is None else directory
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(36, 16))
    keys = [i.to_bytes(22, 'big') for i in range(1, len(vectors) + 1)]
    model_id = 1 if model == 'hubert' else 2
    unit_type = 'word' if condition == 'word' else 'marker'
    records = [svs.metadata.VectorMetadata(f'type-{i % 3}', unit_type,
        [key_helper.pack_hidden_state_key(model_id, layer, key, 500)], 0)
        for i, key in enumerate(keys)]
    name = f'{model}_{condition}_{n_types}_{layer}_{pooling}'
    metadata = svs.metadata.VectorMetadatas(records, name, str(directory))
    index = svs.VectorIndex(vectors, metadata)
    index.save()
    with lv_locations.selection_path(condition, workspace).open('wb') as handle:
        pickle.dump(keys, handle)
    return index


class NeighbourExportTests(unittest.TestCase):
    def test_query_repr_shows_precision_only_for_cached_neighbours(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            save_index(workspace)
            paths = retrieval_evaluation.export_neighbours('hubert', 3,
                'center', workspace=workspace)
            with neighbour_results.Result(paths['word']) as result:
                query = result[0]
                self.assertEqual(repr(query), "<Query index=0 label='type-0'>")
                expected = query.precision(10)
                self.assertEqual(query.precision(), expected)
                self.assertEqual(query.precision(n=10), expected)
                for n in [0, -1, True, 1.5, 31]:
                    with self.assertRaises(ValueError):
                        query.precision(n=n)
                self.assertIsNotNone(query._same)
                self.assertIsNotNone(query._neighbours)
                self.assertEqual(result._tokens, {})
                with mock.patch.object(neighbour_results.Query, '_array',
                    side_effect=AssertionError('must use cached matches')):
                    self.assertEqual(query.precision(), expected)
                    self.assertEqual(repr(query), '<Query index=0 '
                        f"label='type-0' precision@10={expected:.1f}>")
                self.assertEqual(result._tokens, {})
                self.assertIsNone(result._store)

    def test_each_condition_exports_compact_results_from_current_pipeline_index(self):
        cases = [('hubert', 'hubert', 'word', 3, 'center'),
            ('hubert', 'hubert', 'cross_boundary', 6, 'mean'),
            ('wav2vec2', 'w2v', 'word', 9, 'center'),
            ('w2v', 'w2v', 'cross_boundary', 12, 'mean')]
        for model_name, model, condition, layer, pooling in cases:
            with self.subTest(model=model, condition=condition, pooling=pooling):
                with tempfile.TemporaryDirectory() as directory:
                    workspace = Path(directory).resolve()
                    index = save_index(workspace, model, condition, layer,
                        pooling)
                    paths = retrieval_evaluation.export_neighbours(model_name,
                        layer, pooling, condition=condition, workspace=workspace,
                        batch_size=8)
                    self.assertEqual(set(paths), {condition})
                    path = paths[condition]
                    self.assertEqual(path,
                        workspace / 'neighbours' / condition / model /
                        f'layer{layer:02d}_{pooling}.h5')
                    self.assertEqual(list(workspace.rglob('*.h5')), [path])
                    other = 'word' if condition == 'cross_boundary' else 'cross_boundary'
                    self.assertFalse((workspace / 'neighbours' / other).exists())
                    self.assertTrue((path.parent.parent / 'token_mapping.npz').exists())
                    with mock.patch(
                        'lexicon_validation.lv_stores.load_echoframe_store',
                        side_effect=AssertionError('numeric reads need no store')):
                        with neighbour_results.Result(path) as results:
                            self.assertEqual(len(results), 36)
                            for i in range(len(results)):
                                query = results[i]
                                self.assertEqual(query.token.label,
                                    index.metadatas[i].label)
                                self.assertEqual(len(query.neighbour_indices), 30)
                                self.assertNotIn(i, query.neighbour_indices)
                                scores = query.neighbour_similarities
                                self.assertTrue(np.all(scores[:-1] >= scores[1:]))
                                reference = index.query_by_index(i, top_k=31)
                                neighbours = [j for j in reference['indices']
                                    if j != i][:30]
                                np.testing.assert_array_equal(
                                    query.neighbour_indices, neighbours)
                                expected = [j for j, md in enumerate(index.metadatas)
                                    if md.label == query.token.label]
                                np.testing.assert_array_equal(
                                    query.same_type_indices, expected)
                                self.assertEqual(len(query.same_type_ranks), 12)
                                self.assertEqual(query.neighbour_indices.dtype,
                                    np.dtype('uint32'))
                                self.assertEqual(scores.dtype, np.dtype('float32'))

    def test_default_condition_is_word_and_custom_index_directory_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            custom = workspace / 'custom_vectors'
            save_index(workspace, directory=custom)
            paths = retrieval_evaluation.export_neighbours('hubert', 3,
                'center', directory=custom, workspace=workspace, top_k=5,
                compression=None)
            self.assertEqual(set(paths), {'word'})
            store = mock.Mock()
            metadata = SimpleNamespace(phraser_object=object())
            store.load_metadata.return_value = metadata
            with neighbour_results.Result(paths['word'],
                store=store) as results:
                self.assertEqual(len(results[0].neighbour_indices), 5)
                self.assertIs(results[7].token.metadata, metadata)
                store.load_metadata.assert_called_once_with(results.source_key(7))
            store.close.assert_not_called()

    def test_existing_result_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            save_index(workspace)
            options = dict(workspace=workspace, top_k=5)
            path = retrieval_evaluation.export_neighbours('hubert', 3,
                'center', **options)['word']
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                retrieval_evaluation.export_neighbours('hubert', 3, 'center',
                    **options)
            self.assertEqual(path.read_bytes(), before)
            retrieval_evaluation.export_neighbours('hubert', 3, 'center',
                overwrite=True, **options)
            with neighbour_results.Result(path) as results:
                self.assertEqual(len(results), 36)

    def test_multiple_matching_indexes_fail_before_loading_or_exporting(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            save_index(workspace, n_types=3)
            save_index(workspace, n_types=4)
            with mock.patch.object(svs.VectorIndex, 'load') as loader:
                with self.assertRaisesRegex(ValueError, 'multiple vector indexes'):
                    retrieval_evaluation.export_neighbours('hubert', 3,
                        'center', workspace=workspace)
                loader.assert_not_called()
            self.assertFalse((workspace / 'neighbours').exists())

    def test_saved_selection_order_is_checked_before_result_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            index = save_index(workspace)
            keys = [key_helper.unpack_echoframe_key(md.source_echoframe_keys[0])[
                'phraser_key'] for md in index.metadatas]
            with lv_locations.selection_path('word', workspace).open('wb') as handle:
                pickle.dump(list(reversed(keys)), handle)
            with self.assertRaisesRegex(ValueError, 'selection order'):
                retrieval_evaluation.export_neighbours('hubert', 3, 'center',
                    workspace=workspace)
            self.assertEqual(list(workspace.rglob('*.h5')), [])

    def test_invalid_arguments_fail_before_loading(self):
        cases = [('unknown', 9, 'mean'), ('hubert', 4, 'mean'),
            ('hubert', 3.0, 'mean'), ('hubert', True, 'mean'),
            ('hubert', 9, 'unknown')]
        with mock.patch.object(svs.VectorIndex, 'load') as loader:
            for args in cases:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    retrieval_evaluation.export_neighbours(*args)
            with self.assertRaises(ValueError):
                retrieval_evaluation.export_neighbours('hubert', 3, 'mean',
                    condition='unknown')
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
