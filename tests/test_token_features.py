from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import token_features as tf


SYLLABLES = {}


def phone(label, start, end, syllable):
    '''Make a phone linked to a shared fake syllable per syllable key.'''
    if syllable not in SYLLABLES:
        SYLLABLES[syllable] = SimpleNamespace(key=syllable, label='',
            start=start, end=end)
    parent = SYLLABLES[syllable]
    parent.label = (parent.label + ' ' + label).strip()
    parent.start, parent.end = min(parent.start, start), max(parent.end, end)
    return SimpleNamespace(label=label, start=start, end=end,
        key=f'p{start}'.encode(), parent_key=syllable, parent=parent)


def word(label, start, end, phones, phrase=None):
    return SimpleNamespace(object_type='Word', label=label, start=start,
        end=end, duration=end - start, key=label.encode(), phones=phones,
        phrase=phrase)


def make_phrase():
    '''Three words: de (1 syllable), boom (1), staat (1); speaker age 0.'''
    SYLLABLES.clear()
    de = word('de', 0, 100, [phone('d', 0, 50, b's1'),
        phone('ə', 50, 100, b's1')])
    boom = word('boom', 100, 300, [phone('b', 100, 150, b's2'),
        phone('o', 150, 250, b's2'), phone('m', 250, 300, b's2')])
    staat = word('staat', 300, 500, [phone('s', 300, 350, b's3'),
        phone('t', 350, 400, b's3'), phone('a', 400, 450, b's3'),
        phone('t', 450, 500, b's3')])
    phrase = SimpleNamespace(key=b'phrase', speaker_id=1,
        words=[staat, de, boom],
        phones=staat.phones + de.phones + boom.phones)
    for item in (de, boom, staat): item.phrase = phrase
    return phrase, de, boom, staat


def make_audio(phrase):
    '''Same-speaker phrase after the target phrase; other speaker between.'''
    other = SimpleNamespace(speaker_id=2, words=[word('ja', 520, 600, [])],
        phones=[phone('j', 520, 600, b's4')])
    later = SimpleNamespace(speaker_id=1, words=[word('hier', 700, 900, []),
        word('nu', 600, 700, [])], phones=[phone('h', 700, 800, b's6'),
        phone('i', 800, 900, b's6'), phone('n', 600, 650, b's5'),
        phone('u', 650, 700, b's5')])
    return SimpleNamespace(filename='/data/audio/wav/comp-g/nl/fn000173.wav',
        phrases=[later, other, phrase])


def shared(item, audio):
    item.speaker_key, item.speaker_id = b'spk', 1
    item.audio_key, item.audio = b'audio', audio
    item.speaker = SimpleNamespace(name='N00001', gender=lambda: 'female',
        age=0)
    return item


class TokenFeatureTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(tf.clear_cache)
        self.phrase, self.de, self.boom, self.staat = make_phrase()
        self.marker = SimpleNamespace(object_type='Marker', label='o m s t',
            start=150, end=400, duration=250, key=b'marker',
            phrase=self.phrase, overlap_word=[self.staat, self.boom],
            overlap_phone=self.staat.phones[:2] + self.boom.phones[1:])
        self.audio = make_audio(self.phrase)
        for item in (self.de, self.boom, self.marker): shared(item, self.audio)

    def test_word_rows_use_linked_phones_and_phrase_position(self):
        features = tf.build_token_features([self.boom, self.de])
        self.assertEqual(tuple(features), tf.COLUMNS)
        self.assertEqual(features['phones'].tolist(), ['b o m', 'd ə'])
        self.assertEqual(features['n_phones'].tolist(), [3, 2])
        self.assertEqual(features['n_syllables'].tolist(), [1, 1])
        self.assertEqual(features['syllables'].tolist(), ['b o m', 'd ə'])
        self.assertEqual(features['word_position'].tolist(), [1, 0])
        self.assertEqual(features['phrase_initial'].tolist(), [False, True])
        self.assertEqual(features['phrase_final'].tolist(), [False, False])
        self.assertEqual(features['component'].tolist(), ['g', 'g'])
        self.assertEqual(features['recording'].tolist(), ['fn000173'] * 2)
        self.assertEqual(features['gender'].tolist(), ['female'] * 2)
        self.assertEqual(features['age'].tolist(), [0, 0])
        self.assertEqual(features['left_word'].tolist(), ['', ''])
        self.assertEqual(features['boundary_index'].tolist(), [-1, -1])

    def test_marker_counts_unique_syllables_and_boundary(self):
        features = tf.build_token_features([self.marker])
        row = {name: values[0] for name, values in features.items()}
        self.assertEqual(row['phones'], 'o m s t')
        self.assertEqual(row['n_phones'], 4)
        self.assertEqual(row['n_syllables'], 2)
        self.assertEqual(row['syllables'], 'b o m | s t a t')
        self.assertEqual((row['left_word'], row['right_word']),
            ('boom', 'staat'))
        self.assertEqual(row['boundary_index'], 2)
        self.assertEqual(row['word_position'], 1)
        self.assertTrue(row['phrase_final'])

    def test_context_words_follow_speaker_across_phrases(self):
        features = tf.build_token_features([self.de, self.marker],
            n_context_words=2)
        self.assertEqual(features['left_words'].tolist(), ['* *', '* de'])
        self.assertEqual(features['right_words'].tolist(),
            ['boom staat', 'nu hier'])
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            tf.build_token_features([self.de], n_context_words=-1)

    def test_rows_keep_input_order_across_recordings(self):
        phrase, de, boom, _ = make_phrase()
        for item in (de, boom):
            shared(item, make_audio(phrase)).audio_key = b'another'
        items = [self.boom, de, self.de, boom]
        features = tf.build_token_features(items)
        self.assertEqual(features['label'].tolist(),
            ['boom', 'de', 'de', 'boom'])
        self.assertEqual(features['right_words'].tolist(),
            ['staat nu hier', 'boom staat nu', 'boom staat nu',
            'staat nu hier'])

    def test_articulation_rate_excludes_token_and_other_speakers(self):
        features = tf.build_token_features([self.boom, self.marker])
        np.testing.assert_allclose(features['articulation_rate'],
            [10 / 0.6, 9 / 0.55])
        np.testing.assert_allclose(features['rate_speech_s'], [0.6, 0.55])
        with mock.patch.object(tf, 'RATE_WINDOW_MS', 100):
            features = tf.build_token_features([self.boom])
        self.assertEqual(features['articulation_rate'][0], 20.)
        self.assertEqual(features['rate_speech_s'][0], 0.2)

    def test_rate_is_nan_without_surrounding_speech(self):
        phrase, de, _, _ = make_phrase()
        phrase.phones = de.phones
        audio = make_audio(phrase)
        audio.phrases = [phrase]
        features = tf.build_token_features([shared(de, audio)])
        self.assertTrue(np.isnan(features['articulation_rate'][0]))
        self.assertEqual(features['rate_speech_s'][0], 0.)

    def test_marker_cutting_a_phone_is_rejected(self):
        self.marker.start = 160
        with self.assertRaisesRegex(ValueError, 'cut through a phone'):
            tf.build_token_features([self.marker])

    def test_saved_table_round_trips_and_checks_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            keys = [self.de.key, self.boom.key]
            with (workspace / 'word_keys.pkl').open('wb') as handle:
                pickle.dump(keys, handle)
            store = mock.Mock()
            store.load_many.return_value = [self.de, self.boom]
            built = tf.load_token_features('word', store, workspace=workspace)
            path = workspace / 'token_features' / 'word.npz'
            self.assertTrue(path.exists())
            store.load_many.reset_mock()
            loaded = tf.load_token_features('word', store, workspace=workspace)
            store.load_many.assert_not_called()
            self.assertEqual(tuple(loaded), tf.COLUMNS)
            for name in tf.COLUMNS:
                np.testing.assert_array_equal(loaded[name], built[name])
            with self.assertRaisesRegex(ValueError, 'does not match results'):
                tf.load_token_features('word', store, workspace=workspace,
                    selection_fingerprint='other')
            with self.assertRaisesRegex(ValueError, 'context size'):
                tf.load_token_features('word', store, workspace=workspace,
                    n_context_words=2)
            with (workspace / 'word_keys.pkl').open('wb') as handle:
                pickle.dump(keys[::-1], handle)
            with self.assertRaisesRegex(ValueError, 'rebuild it'):
                tf.load_token_features('word', store, workspace=workspace)
            store.load_many.return_value = [self.boom, self.de]
            rebuilt = tf.load_token_features('word', store, overwrite=True,
                workspace=workspace)
            self.assertEqual(rebuilt['label'].tolist(), ['boom', 'de'])


if __name__ == '__main__':
    unittest.main()
