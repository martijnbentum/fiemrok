import json
from pathlib import Path
import sys
import tempfile
import unittest

import faiss
import numpy as np
import speech_vector_search as svs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyse_results


class NeighbourExportTests(unittest.TestCase):
    def test_saved_indexes_export_all_tokens_with_ranked_non_self_matches(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(35, 16))
        tokens = [svs.metadata.VectorMetadata(f'type-{i % 3}', 'word',
            [i.to_bytes(2, 'big')], 0) for i in range(len(vectors))]
        cases = [('hubert', 'hubert1', 3, 'center'),
            ('hubert', 'hubert1', 6, 'mean'),
            ('wav2vec2', 'w2v', 9, 'center'),
            ('wav2vec2', 'w2v', 12, 'mean')]
        for model_name, model, layer, pooling in cases:
            with self.subTest(model=model, pooling=pooling):
                with tempfile.TemporaryDirectory() as directory:
                    suffix = '_mean' if pooling == 'mean' else ''
                    names = [f'{model}_{layer}{suffix}',
                        f'cross_boundary_lexicon100_2200_'
                        f'{model}_{layer}_{pooling}']
                    for name in names:
                        metadata = svs.metadata.VectorMetadatas(tokens,
                            name, directory)
                        svs.VectorIndex(vectors, metadata).save()
                    threads = faiss.omp_get_max_threads()
                    paths = analyse_results.export_neighbours(model_name,
                        layer, pooling, directory=directory, batch_size=8)
                    self.assertEqual(faiss.omp_get_max_threads(), threads)
                    self.assertEqual(set(paths), {'word', 'cross_boundary'})
                    for path in paths.values():
                        rows = [json.loads(line)
                            for line in path.read_text().splitlines()]
                        self.assertEqual(len(rows), len(tokens))
                        for query, row in zip(tokens, rows):
                            self.assertEqual(row['label'], query.label)
                            self.assertEqual(row['source_keys_hex'],
                                query.source_echoframe_keys_hex)
                            neighbours = row['neighbours']
                            self.assertEqual(len(neighbours), 30)
                            for neighbour in neighbours:
                                keys = neighbour['source_keys_hex']
                                self.assertTrue(set(keys).isdisjoint(
                                    row['source_keys_hex']))
                                key = bytes.fromhex(keys[0])
                                token = tokens[int.from_bytes(key, 'big')]
                                self.assertEqual(neighbour['label'], token.label)
                            scores = [n['similarity'] for n in neighbours]
                            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_invalid_arguments_fail_before_loading(self):
        cases = [('unknown', 9, 'mean'), ('hubert', 4, 'mean'),
            ('hubert', 3.0, 'mean'), ('hubert', 9, 'unknown')]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                analyse_results.export_neighbours(*args)


if __name__ == '__main__':
    unittest.main()
