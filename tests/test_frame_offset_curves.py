from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from echoframe.key_helper import pack_hidden_state_key
import speech_vector_search as svs

from lexicon_validation import frame_offset_curves as foc
from lexicon_validation import frame_offset_plots as fop
from lexicon_validation import lv_locations
from lexicon_validation import neighbour_computation as nc
from lexicon_validation import neighbour_results as nr


class FrameOffsetCurveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
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
        self.filename = lv_locations.frame_offset_path('hubert', 'word', 9,
            'mean', self.workspace)
        self.filename.parent.mkdir(parents=True)
        nc.write_frame_offsets(self.index, self.filename,
            self.workspace / 'mapping.npz', self.keys, self.queries(),
            max_offset=1, top_k=3, batch_size=2, workspace=self.workspace,
            model='hubert')

    def queries(self):
        for i, key in enumerate(self.keys):
            records = []
            for offset in (-1, 0, 1):
                invalid = i == 0 and offset < 0
                vector = np.array([1., 0.]) if offset == 0 else np.array([0., 1.])
                records.append(dict(offset=offset, center_index=100,
                    query_time=None if invalid else 2.0 + offset * .02,
                    invalid_reason='outside_recording' if invalid else None,
                    vector=None if invalid else vector))
            yield dict(key=key, records=records, window_start=0.,
                window_end=5., reference_start=1.9, reference_end=2.1,
                reference_kind='word')

    def adjacency(self):
        labels = np.array(['b', 'a', 'zzz', 'a', 'b', ''])
        gaps = np.array([50., 20., 10., 150., 0., np.nan])
        return dict(prev_label=labels, prev_gap=gaps, next_label=labels,
            next_gap=gaps)

    def grid(self, value):
        return int(np.argmin(abs(foc.POSITION_GRID - value)))

    def test_precision_and_self_rank_follow_stored_offsets(self):
        with nr.Result(self.filename, index=self.index) as result:
            curves = foc.compute_curves(result, n_neighbours=3)
            types = result._mapping['token_types']
            arrays = result.read(0, 6)
        middle = self.grid(0.5)
        for token in range(6):
            neighbours = arrays['neighbour_indices'][token, 1]
            expected = np.mean(types[neighbours] == types[token])
            self.assertAlmostEqual(curves['precision'][token, middle], expected,
                places=6)
        column = list(result._mapping['type_indices'][types[1]]).index(1)
        rank = arrays['same_type_ranks'][1, 1, column]
        self.assertAlmostEqual(curves['self_rank'][1, middle],
            np.log10(rank + 1), places=5)
        self.assertTrue(np.isnan(curves['precision'][:, self.grid(0.3)]).all())
        self.assertTrue(np.isnan(curves['precision'][0, self.grid(0.4333)]))
        self.assertFalse(np.isnan(curves['precision'][1, self.grid(0.4333)]))
        self.assertTrue(np.isnan(curves['prev']).all())

    def test_onset_and_offset_grids_use_milliseconds(self):
        with nr.Result(self.filename, index=self.index) as result:
            curves = foc.compute_curves(result, n_neighbours=3)
        onset = list(foc.ONSET_MS).index(100)
        offset = list(foc.OFFSET_MS).index(-100)
        # Token 0 starts at offset 0, exactly on the grid edge.
        expected = curves['precision'][1:, self.grid(0.5)]
        np.testing.assert_allclose(curves['onset'][1:, onset], expected,
            atol=1e-6)
        np.testing.assert_allclose(curves['offset'][1:, offset], expected,
            atol=1e-6)

    def test_adjacent_curves_need_lexicon_type_within_gap(self):
        with nr.Result(self.filename, index=self.index) as result:
            curves = foc.compute_curves(result, self.adjacency(),
                n_neighbours=3)
            types = result._mapping['token_types']
            arrays = result.read(0, 6)
        middle = self.grid(0.5)
        usable = ~np.isnan(curves['prev']).all(1)
        np.testing.assert_array_equal(usable,
            [True, True, False, False, True, False])
        b = types[1]
        expected = np.mean(types[arrays['neighbour_indices'][0, 1]] == b)
        self.assertAlmostEqual(curves['prev'][0, middle], expected, places=6)

    def test_too_few_saved_neighbours_are_rejected(self):
        with nr.Result(self.filename, index=self.index) as result:
            with self.assertRaisesRegex(ValueError, 'fewer than n_neighbours'):
                foc.compute_curves(result, n_neighbours=10)

    def test_curves_are_cached_and_checked_against_the_selection(self):
        with (mock.patch.object(foc, 'load_adjacency',
            return_value=self.adjacency()) as adjacency,
            mock.patch.object(foc, 'compute_curves',
                wraps=foc.compute_curves) as compute):
            first = foc.load_curves('hubert', 'word', 9, 'mean',
                n_neighbours=3, workspace=self.workspace)
            second = foc.load_curves('hubert', 'word', 9, 'mean',
                n_neighbours=3, workspace=self.workspace)
        self.assertEqual(compute.call_count, 1)
        adjacency.assert_called_once()
        for name in foc.CURVES:
            np.testing.assert_array_equal(first[name], second[name])
        with self.assertRaisesRegex(ValueError, 'overwrite=True'):
            foc.load_curves('hubert', 'word', 9, 'mean', n_neighbours=2,
                workspace=self.workspace)

    def test_cross_boundary_has_no_adjacency(self):
        with self.assertRaisesRegex(ValueError, 'only defined for word'):
            foc.load_adjacency('cross_boundary', workspace=self.workspace)

    def test_build_adjacency_uses_stream_neighbours_and_gaps(self):
        stream = [SimpleNamespace(key=k, label=k, start=s, end=e,
            audio_key='r', speaker_key='s')
            for k, s, e in (('w1', 0, 100), ('w2', 150, 300), ('w3', 300, 400))]
        adjacency = foc.build_adjacency([stream[2], stream[0], stream[1]],
            stream_words=lambda item: stream)
        np.testing.assert_array_equal(adjacency['prev_label'], ['w2', '', 'w1'])
        np.testing.assert_array_equal(adjacency['prev_gap'], [0, np.nan, 50])
        np.testing.assert_array_equal(adjacency['next_label'], ['', 'w2', 'w3'])
        np.testing.assert_array_equal(adjacency['next_gap'], [np.nan, 50, 0])

    def test_type_weighted_mean_weights_tokens_and_skips_missing(self):
        values = np.array([[1., np.nan], [3., 2.], [5., 4.]])
        mean, low, high = foc.type_weighted_mean(values, np.array([0, 0, 1]))
        np.testing.assert_allclose(mean, [3., 3.])
        self.assertTrue((low <= mean).all() and (mean <= high).all())

    def test_plot_saves_png_and_pdf(self):
        curves = {name: np.random.default_rng(0).random((4, len(grid)))
            for name, grid in foc._grids().items()}
        curves['token_types'] = np.array([0, 0, 1, 1])
        with mock.patch.object(fop.foc, 'load_curves', return_value=curves):
            fop.plot_frame_offsets('hubert', 'word', 9,
                workspace=self.workspace)
        figure = lv_locations.figure_path('frame_offsets_hubert_word_layer09',
            self.workspace)
        for suffix in ('png', 'pdf'):
            self.assertTrue(figure.with_name(f'{figure.name}.{suffix}').exists())


if __name__ == '__main__':
    unittest.main()
