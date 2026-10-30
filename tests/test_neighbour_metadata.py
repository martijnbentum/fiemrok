import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neighbour_metadata as nm


def fixture():
    speaker = NS(gender='female', age=40)
    audio = NS(phrases=[])
    words = []
    for i, label in enumerate(['before', 'left', 'right', 'after']):
        word = NS(key=bytes([i + 1]), label=label, start=i * 20,
            end=(i + 1) * 20, audio=audio, audio_key=b'a', speaker=speaker,
            speaker_key=b's', speaker_id=b's', object_type='Word')
        word.phones = [NS(key=bytes([20 + i * 2 + j]), label=str(i * 2 + j),
            start=i * 20 + j * 10, end=i * 20 + (j + 1) * 10, word=word)
            for j in range(2)]
        words.append(word)
        audio.phrases.append(NS(speaker_id=b's', words=[word],
            phones=word.phones))
    # Another speaker's simultaneous annotation must not enter context.
    audio.phrases.insert(1, NS(speaker_id=b'other',
        phones=[NS(label='intruder')], words=[NS(label='intruder')]))
    marker = NS(key=b'm', start=30, end=50, audio=audio, audio_key=b'a',
        speaker=speaker, speaker_key=b's', speaker_id=b's', object_type='Marker')
    segments = {'word': words[1], 'edge': words[0], 'marker': marker}
    by_key = {segment.key: segment for segment in segments.values()}
    store = NS(load_metadata=lambda key: NS(phraser_object=segments[key]),
        load=lambda key: by_key.get(key))
    return store


class MetadataTests(unittest.TestCase):
    def test_cross_phrase_context_and_cross_boundary_words(self):
        record = nm.echoframe_key_to_metadata('marker', fixture(), 4, 3)
        self.assertEqual(record['phone'], dict(target=['3', '4'],
            left_context=['*', '0', '1', '2'],
            right_context=['5', '6', '7', '*'], boundary_index=1))
        self.assertEqual(record['word'], dict(target=['left', 'right'],
            left_context=['*', '*', 'before'],
            right_context=['after', '*', '*'], boundary_index=1))

    def test_word_padding_zero_context_and_deduplication(self):
        store = fixture()
        record = nm.echoframe_key_to_metadata('edge', store, 2, 0)
        self.assertEqual(record['phone']['left_context'], ['*', '*'])
        self.assertEqual(record['phone']['right_context'], ['2', '3'])
        self.assertIsNone(record['phone']['boundary_index'])
        self.assertEqual(record['word']['left_context'], [])
        result = nm.build_metadata_dicts([b'\x02', b'\x02', b'\x01', b'bad'],
            store)
        self.assertEqual(len(result['key_dict']), 2)
        self.assertEqual(len(result['speaker_dict']), 1)
        self.assertEqual(len(result['word_dict']), 2)
        self.assertIn(b'bad'.hex(), result['errors'])
        self.assertIn('02', result['key_dict'])
        self.assertNotIn('phraser_key', result['key_dict']['02'])

    def test_missing_metadata_is_not_padding(self):
        store = NS(load=lambda key: None)
        result = nm.build_metadata_dicts([b'bad'], store)
        self.assertEqual(result['phone_dict'], {})
        self.assertIn(b'bad'.hex(), result['errors'])

    def test_inspection_preserves_all_neighbours_and_counts_index_once(self):
        keys = {str(i): dict(phraser_key=str(i), speaker_key=str(i % 2),
            recording_key=str(i // 2)) for i in range(3)}
        rows = [dict(label=str(i), source_keys_hex=[str(i)],
            neighbours=[dict(label=str(j), source_keys_hex=[str(j)],
                similarity=0.9) for j in range(3) if j != i])
            for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'neighbours.jsonl'
            content = '\n'.join(json.dumps(row) for row in rows)
            path.write_text(content)
            total = nm.metadata_contributions(keys)
            with patch.object(nm, 'echoframe_key_to_phraser_key',
                side_effect=lambda key: key):
                result = list(nm.query_distributions(path, keys))
            self.assertEqual(path.read_text(), content)
        self.assertEqual(total['speaker_counts'], {'0': 2, '1': 1})
        self.assertEqual(total['recording_counts'], {'0': 2, '1': 1})
        self.assertEqual(result[0]['n_neighbours'], 2)
        self.assertEqual(result[0]['same_speaker'], 1)
        self.assertEqual(result[0]['same_recording'], 1)

    def test_cache_loading_context_filenames_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'word_metadata_phones10_words3.json'
            path.write_text('{"cached": true}')
            self.assertEqual(nm.load_or_build_metadata(directory=directory),
                {'cached': True})
            with self.assertRaises(FileNotFoundError):
                nm.load_or_build_metadata(directory=directory, overwrite=True)
            with self.assertRaises(FileNotFoundError):
                nm.load_or_build_metadata(directory=directory, n_word_context=4)

    def test_real_key_decoder_and_store_enumeration(self):
        try:
            from echoframe import key_helper
        except ImportError:
            self.skipTest('Echoframe is unavailable in this interpreter')
        phraser_key = b'p' * 22
        first = key_helper.pack_hidden_state_key(1, 9, phraser_key, 500)
        alternate = key_helper.pack_hidden_state_key(1, 9, phraser_key, 0)
        other_model = key_helper.pack_hidden_state_key(2, 9, b'q' * 22, 500)
        store = NS(index=NS(all_echoframe_keys=[first, alternate, other_model]),
            model_registry=NS(load_model_id=lambda name: 1))
        self.assertEqual(nm.store_phraser_keys(store, 'model'), [phraser_key])
        self.assertEqual(nm.echoframe_key_to_phraser_key(first.hex()),
            phraser_key.hex())


if __name__ == '__main__':
    unittest.main()
