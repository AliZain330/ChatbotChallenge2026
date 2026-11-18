"""Incremental publishing, restart safety and user-requested minute pacing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build import expand


def record(identity: str) -> dict:
    return {'id': identity, 'text': 'Evidence ' + identity, 'metadata': {'kind': 'text'}}


class Store:
    def __init__(self):
        self.ids = {'existing-image'}
        self.batches = []

    def get(self, **kwargs):
        return {'ids': list(self.ids)}

    def upsert(self, **kwargs):
        self.ids.update(kwargs['ids'])
        self.batches.append(kwargs)


class ExpansionTests(unittest.TestCase):
    def test_cached_batches_publish_without_waiting_and_restart_skips_them(self):
        store = Store()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'progress.json'
            with patch.object(expand, 'cached_embed', side_effect=lambda texts, **kw: [[1., 0.] for _ in texts]) as cached, patch.object(expand.time, 'sleep') as sleep:
                state = expand.expand_records([record('a'), record('b'), record('c')], store, path, batch_size=2)
                self.assertEqual(cached.call_count, 2)
                self.assertEqual(state['added_this_run'], 3)
                self.assertIn('existing-image', store.ids)
                sleep.assert_not_called()
                cached.reset_mock()
                state = expand.expand_records([record('a'), record('b'), record('c')], store, path, batch_size=2)
                cached.assert_not_called()
                self.assertEqual(state['added_this_run'], 0)
            self.assertEqual(json.loads(path.read_text())['status'], 'complete')

    def test_api_batches_wait_only_remaining_minute(self):
        clock = [0.]

        def api(texts, **kwargs):
            clock[0] += 3.
            return [[1., 0.] for _ in texts]

        def cached(texts, **kwargs):
            return kwargs['embed_fn'](texts, task_type='RETRIEVAL_DOCUMENT')

        with tempfile.TemporaryDirectory() as folder, patch.object(expand, 'cached_embed', side_effect=cached), patch.object(expand, 'embed', side_effect=api), patch.object(expand.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(expand.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)) as sleep:
            expand.expand_records([record('a'), record('b'), record('c')], Store(), Path(folder) / 'progress.json', batch_size=2)
            sleep.assert_called_once_with(57.)

    def test_later_failure_keeps_published_records_and_images(self):
        store = Store()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'progress.json'
            with patch.object(expand, 'cached_embed', side_effect=[[[1., 0.], [0., 1.]], RuntimeError('daily quota exhausted; wait for reset')]):
                with self.assertRaisesRegex(RuntimeError, 'daily quota'):
                    expand.expand_records([record('a'), record('b'), record('c')], store, path, batch_size=2)
            self.assertEqual(store.ids, {'existing-image', 'a', 'b'})
            state = json.loads(path.read_text())
            self.assertEqual(state['status'], 'blocked')
            self.assertEqual(state['indexed_target_records'], 2)
            self.assertIn('daily quota', state['error'])

    def test_wrong_embedding_count_does_not_publish(self):
        store = Store()
        with tempfile.TemporaryDirectory() as folder, patch.object(expand, 'cached_embed', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'required=1, actual=0'):
                expand.expand_records([record('a')], store, Path(folder) / 'progress.json')
        self.assertEqual(store.ids, {'existing-image'})

    def test_invalid_limits_and_empty_input_are_actionable(self):
        for batch in [0, expand.MAX_EMBED_BATCH + 1]:
            with self.assertRaisesRegex(ValueError, f'actual={batch}'):
                expand.expand_records([record('a')], Store(), Path('unused'), batch_size=batch)
        with self.assertRaisesRegex(ValueError, 'actual=0'):
            expand.expand_records([], Store(), Path('unused'))


if __name__ == '__main__':
    unittest.main()
