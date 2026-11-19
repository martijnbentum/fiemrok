from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from phraser import Lexicon, Phone, Syllable, Word
from phraser.sampling.lexicon import Entry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import frequencies as fq


def lexicon(segment_class, counts):
    entries = {label: Entry(label, [], count)
        for label, count in counts.items()}
    return Lexicon(segment_class, entries)


COUNTS = dict(word={'de': 99, 'boom': 9, 'staat': 0},
    syllable={'d ə': 99, 'b oː m': 9}, phone={'d': 999, 'ə': 9, 'b': 99},
    cross_boundary_sequence={'ə b': 9})


def table(**columns):
    rows = dict(label=['boom', 'ə b'], left_word=['', 'de'],
        right_word=['', 'boom'], boundary_index=[-1, 1],
        syllables=['b oː m', 'd ə | b oː m'], phones=['b oː m', 'ə b'])
    rows.update(columns)
    return {name: np.asarray(values) for name, values in rows.items()}


class FrequencyTests(unittest.TestCase):
    def test_columns_split_word_and_cross_boundary_rows(self):
        columns = fq.frequency_columns(table(), COUNTS)
        np.testing.assert_array_equal(columns['word_log_freq'], [1, np.nan])
        np.testing.assert_array_equal(columns['left_word_log_freq'],
            [np.nan, 2])
        np.testing.assert_array_equal(columns['right_word_log_freq'],
            [np.nan, 1])
        np.testing.assert_array_equal(columns['sequence_log_freq'],
            [np.nan, 1])
        np.testing.assert_array_equal(columns['syllable_log_freq_mean'],
            [1, 1.5])
        np.testing.assert_array_equal(columns['syllable_log_freq_min'], [1, 1])
        # oː is unseen and counts as log10(0 + 1) = 0
        np.testing.assert_allclose(columns['phone_log_freq_mean'],
            [2 / 3, 1.5])

    def test_build_saves_unfiltered_counts_and_load_reads_them(self):
        builders = {name: mock.Mock(return_value=lexicon(cls, COUNTS[name]))
            for name, cls in (('word', Word), ('syllable', Syllable),
            ('phone', Phone))}
        store = mock.Mock()
        with (tempfile.TemporaryDirectory() as workspace,
            mock.patch.dict(fq.LEXICONS, builders),
            mock.patch.object(fq, 'count_cross_boundary_sequences',
                return_value=COUNTS['cross_boundary_sequence']) as sequences):
            paths = fq.build_frequencies(store, workspace=workspace)
            self.assertEqual(set(paths), set(fq.NAMES))
            for builder in builders.values():
                builder.assert_called_once_with(store=store, verbose=True,
                    **fq.UNFILTERED)
            self.assertEqual(fq.load_frequencies(workspace), COUNTS)
            fq.build_frequencies(store, workspace=workspace)
            sequences.assert_called_once()
            builders['word'].assert_called_once()
            fq.build_frequencies(store, overwrite=True, workspace=workspace)
            self.assertEqual(sequences.call_count, 2)

    def test_missing_files_are_reported(self):
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(FileNotFoundError,
                'build_frequencies'):
                fq.load_frequencies(workspace)


if __name__ == '__main__':
    unittest.main()
