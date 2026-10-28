import pathlib
import inspect
import pickle
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cross_boundary_lexicon as cbl

try:
    import phraser
except ImportError:
    phraser = None


@unittest.skipIf(phraser is None, 'phraser is required for marker integration')
class MarkerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = directory.name
        self.store = phraser.Store(path=self.path)
        self.addCleanup(lambda: self.store.close())
        audio = self.store.create(phraser.Audio, filename='sample.wav',
            duration=400, save=True)
        speaker = self.store.create(phraser.Speaker, name='speaker',
            dataset='test', save=True)
        identity = dict(audio_id=audio.identifier, speaker_id=speaker.identifier)
        phrase = phraser.Phrase('phrase', 0, 400, **identity)
        for start, labels in ((0, ['x', 'a']), (200, ['n', 'x'])):
            word = phraser.Word('word', start, start + 200, **identity)
            syllable = phraser.Syllable('syllable', start, start + 200, **identity)
            phrase.add_child(word)
            word.add_child(syllable)
            for i, label in enumerate(labels):
                phone_start = start + i * 100
                syllable.add_child(phraser.Phone(label, phone_start,
                    phone_start + 100, **identity))
        self.store.save_many(phrase.items)
        self.phrase = phrase
        self.source_keys = [item.key for item in phrase.items]
        self.lexicon = cbl.CrossBoundaryLexicon([phrase])
        plan = cbl.plan_sampling(self.lexicon, {2: 1}, tokens_per_type=1)
        self.tokens = self.lexicon.sample_lexicon(plan, min_types=1,
            max_attempts=1)

    def test_preview_preserves_sample_order_links_and_phones_without_saving(self):
        keys_path = pathlib.Path(self.path) / 'preview.pkl'
        markers = self.lexicon.make_markers(self.store, keys_path=keys_path)
        self.assertFalse(keys_path.exists())
        self.assertEqual(len(markers), len(self.tokens))
        for marker, token in zip(markers, self.tokens):
            self.assertEqual(marker.label, token.phoneme_sequence)
            self.assertEqual((marker.start, marker.end), (token.start, token.end))
            self.assertIs(marker.phrase, token.phrase)
            self.assertEqual(marker.audio.key, token.audio.key)
            self.assertEqual(marker.speaker_id, token.phrase.speaker_id)
            self.assertEqual([p.key for p in marker.overlap_phone],
                [p.key for p in token.phones])
            self.assertFalse(self.store.DB.key_exists(marker.key))
            self.assertIsNone(marker.parent)
        self.assertEqual([item.key for item in self.phrase.items], self.source_keys)

    def test_saved_marker_round_trip_preserves_exact_source_phones(self):
        keys_path = pathlib.Path(self.path) / 'sample' / 'marker_keys.pkl'
        markers = self.lexicon.make_markers(self.store, save=True,
            keys_path=keys_path)
        with keys_path.open('rb') as f:
            keys = pickle.load(f)
        self.assertEqual(keys, [marker.key for marker in markers])
        self.store.close()
        self.store = phraser.Store(path=self.path)
        restored = cbl.load_markers(self.store, keys_path=keys_path)
        self.assertEqual([marker.key for marker in restored], keys)
        for loaded, token in zip(restored, self.tokens):
            self.assertEqual(loaded.label, token.phoneme_sequence)
            self.assertEqual((loaded.start, loaded.end), (token.start, token.end))
            self.assertEqual(loaded.phrase.key, token.phrase.key)
            self.assertNotEqual(loaded.key, loaded.phrase.key)
            self.assertEqual(loaded.audio.key, token.audio.key)
            self.assertEqual([p.key for p in loaded.overlap_phone],
                [p.key for p in token.phones])
            self.assertEqual([item.key for item in loaded.phrase.items],
                self.source_keys)

    def test_failed_save_does_not_replace_the_key_file(self):
        keys_path = pathlib.Path(self.path) / 'marker_keys.pkl'
        original = b'existing sample index'
        keys_path.write_bytes(original)
        with mock.patch.object(phraser, 'make_markers',
            side_effect=RuntimeError('database save failed')):
            with self.assertRaisesRegex(RuntimeError, 'database save failed'):
                self.lexicon.make_markers(self.store, save=True,
                    keys_path=keys_path)
        self.assertEqual(keys_path.read_bytes(), original)

    def test_key_file_can_be_disabled(self):
        markers = self.lexicon.make_markers(self.store, save=True, keys_path=None)
        self.assertTrue(all(self.store.DB.key_exists(m.key) for m in markers))

    def test_default_key_directory_is_svs_data(self):
        keys_path = cbl.locations.repo / 'svs_data/cross_boundary_marker_keys.pkl'
        for function in (cbl.load_markers, cbl.CrossBoundaryLexicon.make_markers):
            self.assertEqual(inspect.signature(function).parameters[
                'keys_path'].default, keys_path)


if __name__ == '__main__':
    unittest.main()
