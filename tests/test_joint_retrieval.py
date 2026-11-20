from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon_validation import joint_retrieval as jr


class FakeResult:
    vectors_by_condition = {}

    def __init__(self, filename):
        self.vectors = self.vectors_by_condition[Path(filename).parts[-3]]

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False


class JointRetrievalTests(unittest.TestCase):
    def test_neighbours_span_both_datasets_and_exclude_self(self):
        generator = np.random.default_rng(0)
        FakeResult.vectors_by_condition = dict(
            word=generator.normal(size=(6, 4)),
            cross_boundary=generator.normal(size=(5, 4)))
        with (tempfile.TemporaryDirectory() as workspace,
            mock.patch.object(jr, 'Result', FakeResult)):
            joint = jr.build_joint_neighbours(top_k=10, batch_size=4,
                workspace=workspace)
            again = jr.build_joint_neighbours(top_k=10, workspace=workspace)
        neighbours = joint['neighbours']
        self.assertEqual(neighbours.shape, (11, 10))
        self.assertFalse(np.any(neighbours == np.arange(11)[:, None]))
        np.testing.assert_array_equal(joint['sizes'], [6, 5])
        np.testing.assert_array_equal(again['neighbours'], neighbours)
        vectors = np.concatenate(list(
            FakeResult.vectors_by_condition.values()))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        scores = vectors @ vectors.T
        np.fill_diagonal(scores, -np.inf)
        np.testing.assert_array_equal(neighbours[:, 0], scores.argmax(1))

    def test_other_dataset_share_and_prefixed_labels(self):
        neighbours = np.array([[1, 2], [0, 2], [0, 1]])
        share = jr.other_dataset_share(neighbours, [2, 1], top_k=2)
        self.assertEqual(share, dict(word=0.5, cross_boundary=1.0))
        parts = [dict(condition='word', features=dict(
            label=np.array(['ə b']), speaker=np.array(['s1']))),
            dict(condition='cross_boundary', features=dict(
            label=np.array(['ə b']), speaker=np.array(['s1'])))]
        labels = jr.joint_features(parts)['label']
        self.assertNotEqual(labels[0], labels[1])


if __name__ == '__main__':
    unittest.main()
