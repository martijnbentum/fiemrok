import pathlib
import sys
import tempfile
import unittest

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
        markers = self.lexicon.make_markers(self.store)
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
        markers = self.lexicon.make_markers(self.store, save=True)
        self.store.close()
        self.store = phraser.Store(path=self.path)
        for marker, token in zip(markers, self.tokens):
            loaded = self.store.load(marker.key)
            self.assertEqual(loaded.label, token.phoneme_sequence)
            self.assertEqual((loaded.start, loaded.end), (token.start, token.end))
            self.assertEqual(loaded.phrase.key, token.phrase.key)
            self.assertNotEqual(loaded.key, loaded.phrase.key)
            self.assertEqual(loaded.audio.key, token.audio.key)
            self.assertEqual([p.key for p in loaded.overlap_phone],
                [p.key for p in token.phones])
            self.assertEqual([item.key for item in loaded.phrase.items],
                self.source_keys)


if __name__ == '__main__':
    unittest.main()
