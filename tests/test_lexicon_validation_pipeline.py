from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import echoframe
from echoframe import batch_segment_features, key_helper
import numpy as np
import phraser
import speech_vector_search as svs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import pipeline
from lexicon_validation import lv_locations
from lexicon_validation import lv_stores
from lexicon_validation import embeddings


class PipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = self.root / 'validation'
        self.cgn = phraser.Store(path=self.root / 'cgn')
        self.addCleanup(self.cgn.close)
        filename = self.root / 'comp-b/nl/sample.wav'
        filename.parent.mkdir(parents=True)
        filename.touch()
        audio = self.cgn.create(phraser.Audio, filename=str(filename),
            duration=4000, save=True)
        speaker = self.cgn.create(phraser.Speaker, name='speaker',
            dataset='test', save=True)
        identity = dict(audio_id=audio.identifier,
            speaker_id=speaker.identifier)
        phrase = phraser.Phrase('phrase', 1000, 1400, **identity)
        for start, label, phones in ((1000, 'left', ('x', 'a')),
            (1200, 'right', ('n', 'x'))):
            word = phraser.Word(label, start, start + 200, **identity)
            syllable = phraser.Syllable(label, start, start + 200, **identity)
            phrase.add_child(word)
            word.add_child(syllable)
            for index, phone in enumerate(phones):
                phone_start = start + index * 100
                syllable.add_child(phraser.Phone(phone, phone_start,
                    phone_start + 100, **identity))
        self.cgn.save_many(phrase.items)
        self.phrase = phrase
        second = phraser.Phrase('second phrase', 2000, 2400, **identity)
        for start, label, phones in ((2000, 'left', ('x', 'b')),
            (2200, 'right', ('m', 'x'))):
            word = phraser.Word(label, start, start + 200, **identity)
            syllable = phraser.Syllable(label, start, start + 200, **identity)
            second.add_child(word)
            word.add_child(syllable)
            for index, phone in enumerate(phones):
                phone_start = start + index * 100
                syllable.add_child(phraser.Phone(phone, phone_start,
                    phone_start + 100, **identity))
        self.cgn.save_many(second.items)
        self.cgn.refresh_query_roots()
        local = pipeline.build_local_phraser_store(source_store=self.cgn,
            workspace=self.workspace)
        local.close()

    def prepare(self, min_types=None):
        return pipeline.prepare_samples(n_tokens=1, min_types=min_types,
            max_attempts=1, tolerance=0,
            workspace=self.workspace)

    def test_selection_files_are_ordered_keys_at_workspace_root(self):
        selected = self.prepare()
        self.assertEqual(len(selected['word']), 2)
        self.assertEqual(len(selected['cross_boundary']), 2)
        for condition in pipeline.CONDITIONS:
            path = lv_locations.selection_path(condition, self.workspace)
            self.assertEqual(path.parent, self.workspace)
            with path.open('rb') as handle:
                self.assertEqual(pickle.load(handle), selected[condition])
        local = phraser.Store(path=lv_locations.phraser_store_path(
            workspace=self.workspace))
        self.addCleanup(local.close)
        markers = local.load_many(selected['cross_boundary'])
        self.assertEqual({marker.label for marker in markers}, {'a n', 'b m'})
        self.assertEqual(list(self.cgn.DB.audio_id_to_child_keys(
            self.phrase.audio_id, 'Marker')), [])

    def test_infeasible_floor_does_not_publish_partial_selection_files(self):
        with self.assertRaisesRegex(ValueError, 'exceeds the word-type goal'):
            self.prepare(min_types=3)
        self.assertFalse(lv_locations.selection_path('word',
            self.workspace).exists())
        self.assertFalse(lv_locations.selection_path('cross_boundary',
            self.workspace).exists())

    def test_saved_samples_build_four_stores_and_all_32_indexes(self):
        selected = self.prepare()

        def inference(filenames, **kwargs):
            base = np.arange(100 * 8, dtype=np.float32).reshape(1, 100, 8)
            return iter(SimpleNamespace(hidden_states=[base + layer + i * 10
                for layer in range(13)]) for i, _ in enumerate(filenames))

        with (mock.patch.object(embeddings.model_paths,
            'get_pretrained_model', return_value=Path('/unused/checkpoint')),
            mock.patch.object(echoframe.Store, 'load_model',
            return_value=object()), mock.patch.object(
            batch_segment_features.to_vector, 'iter_filename_batch_to_vector',
            side_effect=inference) as extract):
            stores = pipeline.build_embeddings(workspace=self.workspace,
                verbose=False)
        self.assertEqual(extract.call_count, 4)
        self.assertEqual({p.name for p in stores.values()},
            {'w2v_word_2', 'hubert_word_2',
            'w2v_cross_boundary_2', 'hubert_cross_boundary_2'})
        with mock.patch.object(batch_segment_features.to_vector,
            'iter_filename_batch_to_vector') as extract:
            resumed = pipeline.build_embeddings(model='hubert',
                condition='word', workspace=self.workspace, verbose=False)
        extract.assert_not_called()
        self.assertEqual(resumed, {('hubert', 'word'):
            stores['hubert', 'word']})
        paths = pipeline.build_indexes(workspace=self.workspace)
        self.assertEqual(len(paths), 32)
        for (model, condition, layer, pooling), path in paths.items():
            with self.subTest(condition=(model, condition, layer, pooling)):
                index = svs.VectorIndex.load(str(path.parent), path.stem)
                keys = [key_helper.unpack_echoframe_key(
                    md.source_echoframe_keys[0])['phraser_key']
                    for md in index.metadatas]
                self.assertEqual(keys, selected[condition])
                self.assertEqual(index.metadatas.unit_type,
                    'word' if condition == 'word' else 'marker')
                self.assertTrue(path.exists())
        self.assertEqual(len(list((self.workspace / 'svs_data').glob('*'))), 64)
        self.assertEqual({p.name for p in self.workspace.iterdir()},
            {'word_keys.pkl', 'cross_boundary_marker_keys.pkl',
            'echoframe_stores', 'svs_data', 'cgn_befgijklno_nl'})

    def test_embedding_selectors_only_load_requested_datasets_and_stores(self):
        self.prepare()
        cases = [({'model': 'hubert', 'condition': 'word'},
            {('hubert', 'word')}),
            ({'model': 'w2v'},
                {('w2v', condition) for condition in pipeline.CONDITIONS}),
            ({'condition': 'cross_boundary'},
                {(model, 'cross_boundary') for model in pipeline.MODELS})]
        for selectors, expected in cases:
            with (self.subTest(selectors=selectors),
                mock.patch.object(pipeline, '_load_items',
                    wraps=pipeline._load_items) as load,
                mock.patch.object(embeddings, 'create_store') as create,
                mock.patch.object(embeddings, 'compute_embeddings')
                    as compute):
                paths = pipeline.build_embeddings(workspace=self.workspace,
                    **selectors)
                self.assertEqual(set(paths), expected)
                self.assertEqual({call.args[0] for call in load.call_args_list},
                    {condition for _, condition in expected})
                self.assertEqual(create.call_count, len(expected))
                self.assertEqual(compute.call_count, len(expected))

    def test_word_embedding_does_not_require_cross_boundary_selection(self):
        self.prepare()
        lv_locations.selection_path('cross_boundary', self.workspace).unlink()
        with (mock.patch.object(embeddings, 'create_store'),
            mock.patch.object(embeddings, 'compute_embeddings')):
            paths = pipeline.build_embeddings(model='hubert', condition='word',
                workspace=self.workspace)
        self.assertEqual(set(paths), {('hubert', 'word')})

    def test_embedding_selectors_reject_invalid_values_before_loading(self):
        for selectors in ({'model': 'wav2vec2'}, {'condition': 'words'}):
            with (self.subTest(selectors=selectors),
                mock.patch.object(lv_stores, 'load_phraser_store') as load):
                with self.assertRaises(ValueError):
                    pipeline.build_embeddings(**selectors)
                load.assert_not_called()

    def test_embedding_store_type_count_mismatch_and_ambiguity_fail(self):
        self.prepare()
        wrong = lv_locations.store_path('hubert', 'word', 3, self.workspace)
        wrong.mkdir(parents=True)
        for ambiguous in (False, True):
            if ambiguous:
                lv_locations.store_path('hubert', 'word', 2,
                    self.workspace).mkdir()
            with (self.subTest(ambiguous=ambiguous),
                mock.patch.object(embeddings, 'create_store') as create,
                mock.patch.object(embeddings, 'compute_embeddings')
                    as compute):
                with self.assertRaises(ValueError):
                    pipeline.build_embeddings(model='hubert', condition='word',
                        workspace=self.workspace)
                create.assert_not_called()
                compute.assert_not_called()

    def test_material_stage_filters_audio_and_preserves_hierarchy(self):
        for component, language in (('a', 'nl'), ('b', 'vl'), ('o', 'nl')):
            self.cgn.create(phraser.Audio,
                filename=f'/cgn/comp-{component}/{language}/extra.wav',
                duration=1000, save=True)
        phraser.make_markers([dict(label='old marker', start=1100, end=1300,
            audio=self.phrase.audio, phrase=self.phrase)], self.cgn, save=True)
        self.cgn.refresh_query_roots()
        goal = pipeline.build_local_phraser_store(components='obbb',
            language='nl', source_store=self.cgn,
            workspace=self.root / 'filtered')
        self.addCleanup(goal.close)
        self.assertEqual(Path(goal.path).name, 'cgn_bo_nl')
        self.assertEqual(len(list(goal.audios.all())), 2)
        self.assertEqual(list(goal.markers.all()), [])
        for item in [self.phrase.audio, self.phrase.speaker,
            *self.phrase.items]:
            self.assertEqual(goal.DB.load(item.key), self.cgn.DB.load(item.key))

    def test_material_stage_language_is_parameterized(self):
        audio = self.cgn.create(phraser.Audio,
            filename='/cgn/comp-a/vl/sample.wav', duration=1000, save=True)
        self.cgn.refresh_query_roots()
        goal = pipeline.build_local_phraser_store(components='a',
            language='vl', source_store=self.cgn, workspace=self.workspace)
        self.addCleanup(goal.close)
        self.assertEqual(Path(goal.path).name, 'cgn_a_vl')
        self.assertEqual([a.key for a in goal.audios.all()], [audio.key])

    def test_material_stage_rejects_any_existing_destination(self):
        for components in (lv_locations.default_components, 'a'):
            root = lv_locations.phraser_store_path(components,
                workspace=self.workspace)
            root.mkdir(exist_ok=True)
            with self.subTest(components=components):
                with self.assertRaises(FileExistsError):
                    pipeline.build_local_phraser_store(components=components,
                        source_store=self.cgn, workspace=self.workspace)

    def test_no_matching_audio_does_not_create_a_store(self):
        root = lv_locations.phraser_store_path('a', 'vl', self.workspace)
        with self.assertRaisesRegex(ValueError, 'no recordings match'):
            pipeline.build_local_phraser_store(components='a', language='vl',
                source_store=self.cgn, workspace=self.workspace)
        self.assertFalse(root.exists())

    def test_default_sampling_requires_local_store_without_fallback(self):
        with mock.patch.object(pipeline.stores, 'load_phraser_cgn_store',
            side_effect=AssertionError('must not open the source CGN store')):
            with self.assertRaisesRegex(FileNotFoundError, 'local Phraser'):
                pipeline.prepare_samples(workspace=self.root / 'missing')

    def test_store_name_validation(self):
        for components, language in (('', 'nl'), ('z', 'nl'), ('b', '../nl')):
            with self.subTest(components=components, language=language):
                with self.assertRaises(ValueError):
                    lv_locations.phraser_store_path(components, language)

    def test_cross_boundary_bounds_follow_word_count(self):
        self.assertEqual(pipeline.cross_boundary_type_bounds(1655),
            (1490, 1655))
        self.assertEqual(pipeline.cross_boundary_type_bounds(10), (9, 10))
        self.assertEqual(pipeline.cross_boundary_type_bounds(1), (1, 1))
        self.assertEqual(pipeline.cross_boundary_type_bounds(1655, 1600),
            (1600, 1655))
        with self.assertRaisesRegex(ValueError, 'min_types=2000'):
            pipeline.cross_boundary_type_bounds(1655, 2000)

    def test_store_loaders_attach_local_materials_without_opening_source(self):
        path = lv_locations.store_path('w2v', 'word', 2, self.workspace)
        created = echoframe.Store(path)
        created.close()
        with mock.patch.object(pipeline.stores, 'load_phraser_cgn_store',
            side_effect=AssertionError('must not open shared source')):
            materials = lv_stores.load_phraser_store(workspace=self.workspace)
            self.addCleanup(materials.close)
            embeddings = lv_stores.load_echoframe_store('w2v', 'word',
                materials=materials, workspace=self.workspace)
            self.addCleanup(embeddings.close)
            attached = embeddings.load_phraser_store(lv_stores.CGN_SOURCE_ID)
            self.assertIs(attached, materials)
            key = self.phrase.words[0].key
            self.assertEqual(attached.load(key).label, 'left')
            embeddings.close()
            self.assertTrue(materials.is_open())
            materials.close()

            automatic = lv_stores.load_echoframe_store('w2v', 'word',
                workspace=self.workspace)
            self.addCleanup(automatic.close)
            local = automatic.load_phraser_store(lv_stores.CGN_SOURCE_ID)
            self.assertEqual(local.load(key).label, 'left')
            automatic.close_phraser_stores()
            self.assertFalse(local.is_open())

    def test_store_loaders_do_not_create_missing_directories(self):
        workspace = self.root / 'not-built'
        with self.assertRaises(FileNotFoundError):
            lv_stores.load_phraser_store(workspace=workspace)
        with self.assertRaises(FileNotFoundError):
            lv_stores.load_echoframe_store('hubert', 'cross_boundary',
                workspace=workspace)
        self.assertFalse(workspace.exists())

    def test_store_discovery_distinguishes_conditions_and_rejects_ambiguity(self):
        directory = self.workspace / 'echoframe_stores'
        for name in ('w2v_word_2013', 'w2v_cross_boundary_2000',
            'hubert_word_2013', 'hubert_cross_boundary_2000',
            'w2v_word_2013_backup'):
            (directory / name).mkdir(parents=True)
        self.assertEqual(lv_locations.find_store_path('w2v', 'word',
            self.workspace).name, 'w2v_word_2013')
        self.assertEqual(lv_locations.find_store_path('hubert',
            'cross_boundary', self.workspace).name, 'hubert_cross_boundary_2000')
        (directory / 'w2v_word_2200').mkdir()
        with self.assertRaisesRegex(ValueError, 'multiple w2v/word'):
            lv_stores.load_echoframe_store('w2v', 'word',
                workspace=self.workspace)


if __name__ == '__main__':
    unittest.main()
