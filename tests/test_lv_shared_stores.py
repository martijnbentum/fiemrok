from pathlib import Path
import tempfile
import unittest
from unittest import mock

import echoframe
import phraser

from lexicon_validation import lv_locations, lv_stores


class SharedStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(lv_stores.close_stores)
        self.workspace = Path(self.temporary.name)
        path = lv_locations.phraser_store_path(workspace=self.workspace)
        phraser.Store(path=path).close()
        for condition in ('word', 'cross_boundary'):
            path = lv_locations.store_path('hubert', condition, 2, self.workspace)
            echoframe.Store(path).close()

    def load(self, condition='word', workspace=None):
        return lv_stores.load_echoframe_store('hubert', condition,
            workspace=self.workspace if workspace is None else workspace)

    def test_equal_paths_reuse_handles_and_different_conditions_share_materials(self):
        with (mock.patch.object(lv_stores, 'PhraserStore', wraps=phraser.Store) as materials,
            mock.patch.object(lv_stores, 'EchoframeStore', wraps=echoframe.Store) as embeddings):
            word = self.load()
            self.assertIs(self.load(), word)
            alias = self.workspace / 'alias'
            alias.symlink_to(self.workspace, target_is_directory=True)
            self.assertIs(self.load(workspace=alias), word)
            cross_boundary = self.load('cross_boundary')
            self.assertIsNot(word, cross_boundary)
            local = word.load_phraser_store('cgn')
            self.assertIs(cross_boundary.load_phraser_store('cgn'), local)
            self.assertFalse(hasattr(local, 'echoframe_store'))
            self.assertEqual(materials.call_count, 1)
            self.assertEqual(embeddings.call_count, 2)

    def test_phraser_first_and_echoframe_first_use_the_same_registry(self):
        for first in ('phraser', 'echoframe'):
            with self.subTest(first=first):
                lv_stores.close_stores()
                with mock.patch.object(lv_stores, 'PhraserStore', wraps=phraser.Store) as opener:
                    if first == 'phraser':
                        materials = lv_stores.load_phraser_store(workspace=self.workspace)
                        embeddings = self.load()
                    else:
                        embeddings = self.load()
                        materials = lv_stores.load_phraser_store(workspace=self.workspace)
                    self.assertIs(embeddings.load_phraser_store('cgn'), materials)
                    self.assertIs(self.load(), embeddings)
                    self.assertEqual(opener.call_count, 1)

    def test_supplied_materials_are_reused_and_remain_caller_owned(self):
        path = lv_locations.phraser_store_path(workspace=self.workspace)
        materials = phraser.Store(path=path)
        self.addCleanup(materials.close)
        store = lv_stores.load_echoframe_store('hubert', 'word',
            materials=materials, workspace=self.workspace)
        self.assertIs(lv_stores.load_phraser_store(workspace=self.workspace), materials)
        self.assertIs(self.load(), store)
        lv_stores.close_stores()
        self.assertTrue(materials.is_open())

    def test_cleanup_preserves_module_ownership_when_cached_materials_are_supplied(self):
        materials = lv_stores.load_phraser_store(workspace=self.workspace)
        lv_stores.load_echoframe_store('hubert', 'word',
            materials=materials, workspace=self.workspace)
        lv_stores.close_stores()
        self.assertFalse(materials.is_open())

    def test_closed_caller_materials_are_replaced_and_reattached(self):
        path = lv_locations.phraser_store_path(workspace=self.workspace)
        materials = phraser.Store(path=path)
        store = lv_stores.load_echoframe_store('hubert', 'word',
            materials=materials, workspace=self.workspace)
        materials.close()
        replacement = lv_stores.load_phraser_store(workspace=self.workspace)
        self.assertIsNot(replacement, materials)
        self.assertIs(self.load(), store)
        self.assertIs(store.load_phraser_store('cgn'), replacement)
        self.assertTrue(replacement.is_open())
        self.assertFalse(hasattr(replacement, 'echoframe_store'))

    def test_explicit_cleanup_closes_each_handle_once_and_allows_reopening(self):
        first = self.load()
        materials = first.load_phraser_store('cgn')
        other = self.load('cross_boundary')
        with (mock.patch.object(first, 'close', wraps=first.close) as close_first,
            mock.patch.object(other, 'close', wraps=other.close) as close_other,
            mock.patch.object(materials, 'close', wraps=materials.close) as close_materials):
            lv_stores.close_stores()
            lv_stores.close_stores()
            close_first.assert_called_once()
            close_other.assert_called_once()
            close_materials.assert_called_once()
        self.assertFalse(materials.is_open())
        reopened = self.load()
        self.assertIsNot(reopened, first)
        self.assertTrue(reopened.load_phraser_store('cgn').is_open())

    def test_failed_open_closes_new_materials_and_leaves_no_cached_handle(self):
        with (mock.patch.object(lv_stores, 'EchoframeStore',
            side_effect=RuntimeError('cannot open')),
            mock.patch.object(lv_stores, 'PhraserStore', wraps=phraser.Store) as loader):
            with self.assertRaisesRegex(RuntimeError, 'cannot open'):
                self.load()
            self.assertEqual(loader.call_count, 1)
        self.assertTrue(self.load().load_phraser_store('cgn').is_open())

    def test_failure_for_another_condition_keeps_existing_materials_open(self):
        first = self.load()
        materials = first.load_phraser_store('cgn')
        with mock.patch.object(lv_stores, 'EchoframeStore',
            side_effect=RuntimeError('cannot open')):
            with self.assertRaises(RuntimeError):
                self.load('cross_boundary')
        self.assertTrue(materials.is_open())
        self.assertIs(self.load(), first)

    def test_cleanup_does_not_close_caller_owned_handles(self):
        self.load()
        path = self.workspace / 'caller_materials'
        own = phraser.Store(path=path)
        self.addCleanup(own.close)
        lv_stores.close_stores()
        self.assertTrue(own.is_open())


if __name__ == '__main__':
    unittest.main()
