"""Checkpoint/resume tests with fake embeddings; no network calls."""
import json
from contextlib import closing
import hashlib
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest.mock import MagicMock, patch

from build import embeddings


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.provider = patch.object(embeddings.llm, 'PROVIDER', 'gemini')
        self.model = patch.object(embeddings.llm, 'EMBED_DEPLOYMENT', 'test-embedding')
        self.provider.start()
        self.model.start()
        self.addCleanup(self.provider.stop)
        self.addCleanup(self.model.stop)

    def cache(self, texts, fn):
        return embeddings.cached_embed(texts, embed_fn=fn, cache_dir=self.folder)

    def rows(self):
        path = next(self.folder.glob('*.sqlite3'))
        with closing(sqlite3.connect(path)) as connection:
            return dict(connection.execute('SELECT hash, vector FROM embeddings'))

    def test_reuses_unique_texts_preserves_original_order(self):
        fn = MagicMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
        first = self.cache(['a', 'b', 'a'], fn)
        self.assertEqual(first, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        fn.assert_called_once_with(['a', 'b'], task_type='RETRIEVAL_DOCUMENT')
        self.assertEqual(self.cache(['b', 'a'], fn), [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(fn.call_count, 1)

    def test_successful_batch_survives_later_quota_failure(self):
        from bot.gemini import MAX_EMBED_BATCH
        texts = [str(i) for i in range(MAX_EMBED_BATCH + 1)]
        fn = MagicMock(side_effect=[[[1.0, 0.0]] * MAX_EMBED_BATCH, RuntimeError('quota 429')])
        with self.assertRaisesRegex(RuntimeError, 'quota 429'):
            self.cache(texts, fn)
        self.assertEqual(len(self.rows()), MAX_EMBED_BATCH)
        retry = MagicMock(return_value=[[0.0, 1.0]])
        final = self.cache(texts, retry)
        retry.assert_called_once_with([texts[-1]], task_type='RETRIEVAL_DOCUMENT')
        self.assertEqual(len(final), len(texts))
        self.assertEqual(final[-1], [0.0, 1.0])

    def test_changed_model_uses_separate_cache(self):
        fn = MagicMock(return_value=[[1.0]])
        self.cache(['a'], fn)
        with patch.object(embeddings.llm, 'EMBED_DEPLOYMENT', 'other-model'):
            self.cache(['a'], fn)
        self.assertEqual(fn.call_count, 2)
        self.assertEqual(len(list(self.folder.glob('*.sqlite3'))), 2)

    def test_bad_dimensions_do_not_overwrite_successful_cache(self):
        self.cache(['a'], MagicMock(return_value=[[1.0, 0.0]]))
        original = self.rows()
        with self.assertRaisesRegex(RuntimeError, 'required=2, actual=1'):
            self.cache(['a', 'b'], MagicMock(return_value=[[1.0]]))
        self.assertEqual(self.rows(), original)

    def test_bad_count_and_nonfinite_do_not_persist(self):
        for output in [[], [[float('inf')]], [[True]], [[]]]:
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                self.cache(['a'], MagicMock(return_value=output))
            self.assertEqual(self.rows(), {})

    def test_empty_input_no_network_no_cache(self):
        fn = MagicMock()
        self.assertEqual(self.cache([], fn), [])
        fn.assert_not_called()
        self.assertFalse(list(self.folder.iterdir()))

    def test_tampered_identity_is_rejected(self):
        self.cache(['a'], MagicMock(return_value=[[1.0]]))
        path = next(self.folder.glob('*.sqlite3'))
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE metadata SET value='wrong-provider' WHERE name='provider'")
        with self.assertRaisesRegex(RuntimeError, 'identity required='):
            self.cache(['a'], MagicMock())

    def test_legacy_json_migration_reuses_previous_gateway_calls(self):
        key = hashlib.sha256(b'a').hexdigest()
        payload = {'provider': 'gemini', 'model': 'test-embedding',
                   'task_type': 'RETRIEVAL_DOCUMENT', 'dimensions': 2,
                   'vectors': {key: [1.0, 0.0]}}
        path = self.folder / 'embeddings-gemini-test-embedding.json'
        path.write_text(json.dumps(payload))
        fn = MagicMock()
        self.assertEqual(self.cache(['a'], fn), [[1.0, 0.0]])
        fn.assert_not_called()
        # Once imported, obsolete legacy bytes are no longer read on every run.
        path.write_text('invalid json')
        self.assertEqual(self.cache(['a'], fn), [[1.0, 0.0]])

    def test_invalid_legacy_vectors_do_not_partially_import(self):
        path = self.folder / 'embeddings-gemini-test-embedding.json'
        path.write_text(json.dumps({'provider': 'gemini', 'model': 'test-embedding',
                                   'task_type': 'RETRIEVAL_DOCUMENT',
                                   'vectors': {'a': [1.0], 'b': [1.0, 0.0]}}))
        with self.assertRaisesRegex(RuntimeError, 'dimensions required=1, actual=2'):
            self.cache(['a'], MagicMock())
        self.assertEqual(self.rows(), {})


if __name__ == '__main__':
    unittest.main()
