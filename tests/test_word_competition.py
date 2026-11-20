from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
from echoframe.key_helper import pack_hidden_state_key
import speech_vector_search as svs

from lexicon_validation import frame_offset_plots as fop
from lexicon_validation import lv_locations
from lexicon_validation import neighbour_computation as nc
from lexicon_validation import word_competition as wc
from lexicon_validation.neighbour_results import Result

WORD_LABELS = ['w1', 'w2', 'w3', 'w4'] * 2
MARKER_LABELS = ['x', 'y', 'x', 'y']
# Per marker: (label, word-index row or None, start ms, end ms) per slot.
SLOTS = [
    dict(prev=('w3', 2, 1930, 1975), a=('w1', 0, 1975, 2000),
        b=('w2', 1, 2000, 2030), next=('w4', 3, 2030, 2050)),
    dict(prev=('', None, np.nan, np.nan), a=('zzz', None, 1975, 2000),
        b=('w2', 5, 2000, 2030), next=('', None, np.nan, np.nan)),
    dict(prev=('w3', 6, 1700, 1775), a=('w1', 4, 1975, 2000),
        b=('w2', 5, 2000, 2030), next=('w4', 7, 2030, 2050)),
    dict(prev=('w1', 4, 1930, 1975), a=('w1', 0, 1975, 2000),
        b=('w2', 1, 2000, 2030), next=('w4', 3, 2030, 2050))]


def build_index(labels, name, keys, seed, workspace):
    vectors = np.random.default_rng(seed).normal(size=(len(labels), 3))
    metadata = [svs.metadata.VectorMetadata(label, 'word',
        [pack_hidden_state_key(1, 9, key, 500)], 0)
        for label, key in zip(labels, keys)]
    metadatas = svs.metadata.VectorMetadatas(metadata, name, str(workspace))
    index = svs.VectorIndex(vectors, metadatas)
    index.save()
    return index


class WordCompetitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.word_keys = [bytes([i + 1]) * 22 for i in range(8)]
        self.marker_keys = [bytes([i + 101]) * 22 for i in range(4)]
        self.words = build_index(WORD_LABELS, 'hubert_word_2_9_mean',
            self.word_keys, 0, self.workspace / 'words')
        self.markers = build_index(MARKER_LABELS,
            'hubert_cross_boundary_2_9_mean', self.marker_keys, 1,
            self.workspace / 'markers')
        self.own_path = self.workspace / 'own.h5'
        self.cross_path = self.workspace / 'cross.h5'
        options = dict(max_offset=3, top_k=3, batch_size=2,
            workspace=self.workspace, model='hubert')
        nc.write_frame_offsets(self.markers, self.own_path,
            self.workspace / 'mapping.npz', self.marker_keys, self.queries(),
            **options)
        context = [[self.key(slots, slot) for slot in wc.SLOTS]
            for slots in SLOTS]
        labels = [(slots['a'][0], slots['b'][0]) for slots in SLOTS]
        nc.write_cross_frame_offsets(self.words, self.cross_path,
            self.marker_keys, self.queries(), context, labels, **options)

    def key(self, slots, slot):
        row = slots[slot][1]
        return b'' if row is None else self.word_keys[row]

    def vector(self, token, offset):
        return np.random.default_rng(10 * token + offset + 50).normal(size=3)

    def queries(self):
        for token, key in enumerate(self.marker_keys):
            records = [dict(offset=offset, center_index=100,
                query_time=2.0 + offset * .02, invalid_reason=None,
                vector=self.vector(token, offset)) for offset in range(-3, 4)]
            yield dict(key=key, records=records, window_start=0.,
                window_end=5., reference_start=1.98, reference_end=2.02,
                reference_kind='marker')

    def adjacency(self):
        adjacency = {}
        for slot in wc.SLOTS:
            values = [slots[slot] for slots in SLOTS]
            adjacency[f'{slot}_label'] = np.array([v[0] for v in values])
            adjacency[f'{slot}_key'] = np.array([self.key(slots, slot)
                for slots in SLOTS])
            adjacency[f'{slot}_start'] = np.array([v[2] for v in values], float)
            adjacency[f'{slot}_end'] = np.array([v[3] for v in values], float)
        adjacency['prev_gap'] = adjacency['a_start'] - adjacency['prev_end']
        adjacency['next_gap'] = adjacency['next_start'] - adjacency['b_end']
        return adjacency

    def brute_force_shares(self, token, offset, k):
        vector = self.vector(token, offset)
        vector = vector / np.linalg.norm(vector)
        excluded = [slots[1] for slots in SLOTS[token].values()]
        candidates = [('marker', row, score) for row, score
            in enumerate(self.markers.vectors @ vector) if row != token]
        candidates += [('word', row, score) for row, score
            in enumerate(self.words.vectors @ vector) if row not in excluded]
        top = sorted(candidates, key=lambda c: -c[2])[:k]
        own = np.mean([kind == 'marker'
            and MARKER_LABELS[row] == MARKER_LABELS[token]
            for kind, row, _ in top])
        a = np.mean([kind == 'word' and WORD_LABELS[row] == 'w1'
            for kind, row, _ in top])
        marker = np.mean([kind == 'marker' for kind, _, _ in top])
        return own, a, marker

    def test_slot_table_statuses(self):
        lookup = {label: i for i, label in enumerate(WORD_LABELS[:4])}
        types, statuses = wc.slot_table(self.adjacency(), lookup)
        np.testing.assert_array_equal(statuses['a'], [0, 4, 0, 3])
        np.testing.assert_array_equal(statuses['prev'], [0, 4, 2, 3])
        np.testing.assert_array_equal(statuses['next'], [0, 4, 0, 0])
        np.testing.assert_array_equal(types['a'], [0, -1, 0, -1])
        np.testing.assert_array_equal(types['b'], [1, -1, 1, 1])

    def test_merged_shares_match_brute_force(self):
        with Result(self.own_path) as own, h5py.File(self.cross_path) as cross:
            arrays = own.read(0, 4)
            crossed = {name: cross[name][:] for name in ('status',
                'neighbour_indices', 'neighbour_similarities',
                'a_type_ranks', 'b_type_ranks')}
            word_types, lookup = wc._word_types(cross)
            types, _ = wc.slot_table(self.adjacency(), lookup)
            marker_types = own._mapping['token_types'].astype('int64')
            columns = wc.foc._self_columns(own._mapping['type_indices'], 4)
            values, valid = wc.offset_values(arrays, crossed, marker_types,
                word_types, dict(own=marker_types, **types), columns, 3)
        self.assertTrue(valid.all())
        for position, offset in enumerate(range(-3, 4)):
            own_share, a_share, marker = self.brute_force_shares(0, offset, 3)
            self.assertAlmostEqual(values['own'][0, position], own_share)
            self.assertAlmostEqual(values['a'][0, position], a_share)
            self.assertAlmostEqual(values['marker'][0, position], marker)

    def test_curves_follow_sections_and_populations(self):
        with Result(self.own_path) as own, h5py.File(self.cross_path) as cross:
            curves = wc.compute_curves(own, cross, self.adjacency(), 3)
        present = ~np.isnan(curves['own']).all(1)
        np.testing.assert_array_equal(present, [True, False, True, True])
        self.assertTrue(np.isnan(curves['prev'][[1, 2, 3]]).all())
        self.assertTrue(np.isnan(curves['a'][3]).all())
        # Marker 2's prev word is beyond the gap limit: no prev section.
        prev = slice(0, wc.SECTION_POINTS)
        self.assertTrue(np.isnan(curves['own'][2, prev]).all())
        self.assertFalse(np.isnan(curves['own'][0, prev]).all())
        self.assertAlmostEqual(curves['marker_start'][0], 1 + 5 / 25)
        self.assertAlmostEqual(curves['marker_end'][0], 2 + 20 / 30)
        self.assertEqual(curves['n_words'], 8)
        counts = wc.exclusion_table(curves)
        self.assertEqual(counts['prev'], dict(used=1, not_in_lexicon=0, gap=1,
            collision=1, marker_excluded=1))

    def test_section_positions_drop_gaps(self):
        bounds = np.array([[0, 10], [20, 30], [30, 40], [np.nan, np.nan]])
        sections, fractions = wc.section_positions([5, 15, 25, 40], bounds)
        np.testing.assert_array_equal(sections, [0, -1, 1, 2])
        np.testing.assert_allclose(fractions, [.5, np.nan, .5, 1.])

    def test_simultaneous_band_is_wider_than_pointwise(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(200, 20))
        types = np.repeat(np.arange(40), 5)
        mean, low, high = wc.simultaneous_band(values, types)
        _, point_low, point_high = wc.foc.type_weighted_mean(values, types)
        self.assertTrue((low <= point_low).all() and (high >= point_high).all())
        x = np.arange(5)
        self.assertEqual(wc.regions(x, np.array([0, 1, 1, 0, 1], bool)),
            [(1, 2), (4, 4)])

    def test_plot_saves_png_and_pdf(self):
        rng = np.random.default_rng(0)
        grid = len(wc.SECTION_GRID)
        curves = {name: rng.random((4, grid)) for name in wc.CURVES}
        curves.update(token_types=np.array([0, 0, 1, 1]),
            marker_start=rng.random(4) + 1, marker_end=rng.random(4) + 2,
            n_markers=4, n_words=8,
            **{f'{slot}_status': np.zeros(4, 'int8') for slot in wc.SLOTS})
        with mock.patch.object(fop.wc, 'load_curves', return_value=curves):
            fop.plot_word_competition('hubert', 9, workspace=self.workspace)
        figure = lv_locations.figure_path('word_competition_hubert_layer09',
            self.workspace)
        for suffix in ('png', 'pdf'):
            path = figure.with_name(f'{figure.name}.{suffix}')
            self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
