from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np
from echoframe.key_helper import pack_hidden_state_key
import speech_vector_search as svs

from lexicon_validation import neighbour_results as nr
from lexicon_validation import frame_offset_evaluation as evaluation


class FrameOffsetResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.filename = self.workspace / 'offsets.h5'
        self.mapping = self.workspace / 'mapping.npz'
        self.keys = [bytes([i + 1]) * 22 for i in range(6)]
        vectors = np.array([[1., 0.], [0., 1.], [1., 0.],
            [-1., 0.], [0., -1.], [0., 1.]])
        metadata = [svs.metadata.VectorMetadata(label, 'word',
            [pack_hidden_state_key(1, 9, key, 500)], 0)
            for label, key in zip(['a', 'b', 'a', 'b', 'a', 'b'], self.keys)]
        metadatas = svs.metadata.VectorMetadatas(metadata,
            'hubert_word_2_9_mean', str(self.workspace))
        self.index = svs.VectorIndex(vectors, metadatas)
        self.index.save()

    def queries(self, maximum=1):
        for i, key in enumerate(self.keys):
            records = []
            for offset in range(-maximum, maximum + 1):
                reason = ('outside_recording' if i == 0 and offset < 0 else
                    'outside_model_outputs' if i == 1 and offset > 0 else None)
                # Offsets deliberately use different vectors from their token.
                vector = np.array([1., 0.]) if offset == 0 else np.array([0., 1.])
                records.append(dict(offset=offset, center_index=100,
                    query_time=2.0 + offset * .02 if reason is None else None,
                    invalid_reason=reason, vector=vector if reason is None else None))
            yield dict(key=key, records=records, window_start=0., window_end=5.,
                reference_start=1.9, reference_end=2.1, reference_kind='word')

    def write(self, **kwargs):
        options = dict(max_offset=1, top_k=3, batch_size=2,
            workspace=self.workspace, model='hubert')
        options.update(kwargs)
        return nr.write_frame_offsets(self.index, self.filename, self.mapping,
            self.keys, self.queries(options['max_offset']), **options)

    def test_round_trip_matches_exhaustive_ranks_for_external_queries(self):
        self.write()
        with nr.Result(self.filename) as result:
            self.assertEqual(result.result_kind, 'frame_offsets')
            self.assertEqual(result.pooling, 'mean')
            self.assertEqual(result.layer, 9)
            np.testing.assert_array_equal(result.offsets, [-1, 0, 1])
            for token_index, packet in enumerate(self.queries()):
                token_queries = result[token_index]
                self.assertEqual(token_queries.token.phraser_key, packet['key'])
                for record in packet['records']:
                    query = token_queries.offset(record['offset'])
                    self.assertIs(query.token, token_queries.token)
                    self.assertEqual(query.offset, record['offset'])
                    self.assertEqual(query.valid, record['invalid_reason'] is None)
                    if not query.valid: continue
                    vector = record['vector'] / np.linalg.norm(record['vector'])
                    scores = self.index.vectors @ vector
                    order = np.argsort(-scores, kind='stable')
                    ranks = np.empty(len(order), dtype=int)
                    ranks[order] = np.arange(len(order))
                    expected = order[order != token_index][:3]
                    np.testing.assert_array_equal(query.neighbour_indices, expected)
                    np.testing.assert_allclose(query.neighbour_similarities, scores[expected])
                    np.testing.assert_array_equal(query.same_type_ranks,
                        ranks[query.same_type_indices])
                    np.testing.assert_allclose(query.same_type_similarities,
                        scores[query.same_type_indices])
                    self.assertEqual([m.token.index for m in query.neighbours], list(expected))
                    self.assertEqual([m.rank for m in query.neighbours], list(ranks[expected]))
                    self.assertNotIn(token_index, query.neighbour_indices)
                    matches = sum(self.index.metadatas[int(j)].label ==
                        self.index.metadatas[token_index].label for j in expected)
                    self.assertEqual(query.precision(3), matches / 3)
                    self.assertAlmostEqual(query.query_time, record['query_time'])
                    self.assertEqual(len(query.same), 3)
            self.assertIs(result[2].offset(-1), result[2].offset(-1))
            with self.assertRaises(ValueError): result[0].offset(-2)
            with self.assertRaises(ValueError): result[0].offset(True)
            with self.assertRaises(ValueError): result[0].offset(.5)
            self.assertIsNone(result._store)

    def test_invalid_offsets_preserve_sentinels_and_return_missing_results(self):
        self.write()
        with nr.Result(self.filename, index=self.index) as result:
            for token_index, offset, reason in ((0, -1, 'outside_recording'),
                (1, 1, 'outside_model_outputs')):
                query = result[token_index].offset(offset)
                self.assertFalse(query.valid)
                self.assertEqual(query.invalid_reason, reason)
                self.assertEqual(query.neighbours, ())
                self.assertEqual(query.same, ())
                self.assertTrue(np.isnan(query.query_time))
                self.assertTrue(np.isnan(query.precision(3)))
                self.assertTrue(np.all(query.neighbour_indices == nr.INVALID_INDEX))
                self.assertTrue(np.isnan(query.same_type_similarities).all())
                with self.assertRaises(ValueError): query.precision(4)

    def test_defaults_include_minus100_center_and_plus100(self):
        self.write(max_offset=100)
        with h5py.File(self.filename) as output:
            self.assertEqual(output['neighbour_indices'].shape, (6, 201, 3))
            np.testing.assert_array_equal(output['offsets'][:], np.arange(-100, 101))
            self.assertEqual(output['status'].dtype, np.dtype('uint8'))
            self.assertEqual(output['neighbour_indices'].compression, 'lzf')

    def test_batch_reads_cache_and_closed_access(self):
        self.write()
        with nr.Result(self.filename, index=self.index) as result:
            batch = result.read(0, 2)
            self.assertEqual(batch['neighbour_indices'].shape, (2, 3, 3))
            self.assertEqual(batch['status'].shape, (2, 3))
            result.collect_queries()
            query = result[2].offset(0)
            match = query.neighbours[0]
            self.assertIs(match.token, result.token(match.token.index))
        with self.assertRaisesRegex(ValueError, 'closed'): query.neighbours
        with self.assertRaisesRegex(ValueError, 'closed'): query.precision(1)
        with self.assertRaisesRegex(ValueError, 'closed'): match.similarity

    def test_failed_stream_does_not_publish_or_replace_existing_results(self):
        queries = list(self.queries())
        queries[1]['key'] = self.keys[0]
        with self.assertRaisesRegex(ValueError, 'selection order'):
            nr.write_frame_offsets(self.index, self.filename, self.mapping,
                self.keys, queries, max_offset=1, top_k=3, workspace=self.workspace)
        self.assertFalse(self.filename.exists())
        self.assertFalse(list(self.workspace.glob('*.tmp')))
        self.write()
        before = self.filename.read_bytes()
        with self.assertRaises(FileExistsError): self.write()
        self.assertEqual(before, self.filename.read_bytes())

    def test_evaluation_derives_layer_model_and_condition_from_index(self):
        words = [SimpleNamespace(key=key, start_seconds=2., end_seconds=2.4,
            audio=SimpleNamespace(duration=6000, filename='audio.wav'))
            for key in self.keys]
        hidden = [np.tile([1., 0.], (1, 400, 1)) for _ in range(10)]
        hidden[9] = np.tile([0., 1.], (1, 400, 1))
        outputs = SimpleNamespace(hidden_states=tuple(hidden))
        for model in ('hubert', 'w2v'):
            for condition in ('word', 'cross_boundary'):
                with self.subTest(model=model, condition=condition):
                    self.index.metadatas.name = f'{model}_{condition}_2_9_mean'
                    filename = self.workspace / f'{model}_{condition}.h5'
                    with mock.patch.object(evaluation, 'compute_contextual_embedding',
                        return_value=outputs) as compute:
                        path = evaluation.evaluate_words(words, self.index, 'model',
                            filename, max_offset=1, top_k=3, workspace=self.workspace)
                    self.assertEqual(compute.call_count, len(words))
                    with nr.Result(path, index=self.index) as result:
                        self.assertEqual(result.model, model)
                        self.assertEqual(result.condition, condition)
                        self.assertEqual(result.layer, 9)
                        query = result[0].offset(0)
                        self.assertTrue(query.valid)
                        np.testing.assert_array_equal(query.neighbour_indices, [1, 5, 2])
                        self.assertEqual(result[0].token.phraser_key, words[0].key)
                        self.assertEqual(result._file['reference_kinds'][0], b'word')

    def test_malformed_offset_shapes_are_rejected(self):
        self.write()
        with h5py.File(self.filename, 'r+') as output:
            output['offsets'][0] = -2
        with self.assertRaisesRegex(ValueError, 'offsets'):
            nr.Result(self.filename, index=self.index)

    def test_name_layer_disagreement_fails_before_computing_or_writing(self):
        self.index.metadatas.name = 'hubert_word_2_6_mean'
        self.assert_evaluation_rejected('name specifies layer 6, but source keys specify layer 9')

    def test_mixed_source_layers_fail_before_computing_or_writing(self):
        self.index.metadatas[3].source_echoframe_keys[0] = pack_hidden_state_key(
            1, 6, self.keys[3], 500)
        self.assert_evaluation_rejected(r'expected one source layer; found \[6, 9\]')

    def test_malformed_name_fails_before_computing_or_writing(self):
        self.index.metadatas.name = 'not-a-pipeline-index'
        self.assert_evaluation_rejected('invalid index name')

    def assert_evaluation_rejected(self, message):
        words = [SimpleNamespace(key=key) for key in self.keys]
        with (mock.patch.object(evaluation, 'compute_contextual_embedding') as compute,
            mock.patch.object(nr, 'write_frame_offsets') as write,
            self.assertRaisesRegex(ValueError, message)):
            evaluation.evaluate_words(words, self.index, 'model', self.filename,
                workspace=self.workspace)
        compute.assert_not_called()
        write.assert_not_called()
        self.assertFalse(self.filename.exists())


if __name__ == '__main__':
    unittest.main()
