"""Image ingestion tests; gateway calls are mocked, never paid."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from build import images


class ImageTests(unittest.TestCase):
    def test_workshop_inventory_has_provenance_and_zones(self):
        records = images.workshop_images()
        self.assertEqual(len(records), len(images.PHOTO_ZONES))
        self.assertTrue(all(Path(r['local_path']).is_file() for r in records))
        self.assertTrue(all(r['zone'] and r['provenance'] for r in records))

    def test_cached_description_invalidates_on_image_and_prompt_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'photo.png'
            path.write_bytes(b'first image')
            cache = Path(directory) / 'cache.json'
            record = {'src': 'local:photo', 'local_path': str(path)}
            with patch.object(images, 'describe_image', return_value='visible text') as describe:
                self.assertEqual(images.describe_all([record], cache), {'local:photo': 'visible text'})
                images.describe_all([record], cache)
                self.assertEqual(describe.call_count, 1)
                path.write_bytes(b'changed image')
                images.describe_all([record], cache)
                self.assertEqual(describe.call_count, 2)
                with patch.object(images, 'DESCRIPTION_PROMPT', 'new prompt'):
                    images.describe_all([record], cache)
                self.assertEqual(describe.call_count, 3)
            saved = json.loads(cache.read_text(encoding='utf-8'))
            self.assertIn('sha256', saved['entries']['local:photo'])

    def test_failed_input_does_not_cache_a_description(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cache.json'
            with self.assertRaisesRegex(RuntimeError, 'actual=1 of 1'):
                images.describe_all([{'src': 'local:missing', 'local_path': str(Path(directory) / 'absent.png')}], cache)
            self.assertFalse(cache.exists())

    def test_stable_unique_ids_and_no_year_from_upload_path(self):
        records = [{'src': 'https://site/2025/photo.jpg', 'page': 'https://site/page'},
                   {'src': 'poster', 'caption': 'Workshop poster 2024'}]
        descriptions = {r['src']: 'visible evidence' for r in records}
        first = images.image_records(descriptions, records + [records[0]])
        second = images.image_records(descriptions, list(reversed(records)))
        self.assertEqual(len(first), 2)
        self.assertEqual({r['id'] for r in first}, {r['id'] for r in second})
        self.assertEqual(first[0]['metadata']['year'], '')
        self.assertEqual(first[1]['metadata']['year'], '2024')
        self.assertEqual(first[1]['metadata']['type'], 'poster')

    def test_index_uses_upsert_with_same_id_on_rerun(self):
        store = MagicMock()
        record = {'src': 'local:photo', 'zone': 'Makerspace A'}
        with patch.object(images, 'get_store', return_value=store), patch.object(images, 'cached_embed', return_value=[[1.0, 0.0]]):
            images.index_descriptions({'local:photo': 'two visible tables'}, [record])
            images.index_descriptions({'local:photo': 'two visible tables'}, [record])
        calls = store.upsert.call_args_list
        self.assertEqual(calls[0].kwargs['ids'], calls[1].kwargs['ids'])
        self.assertEqual(calls[0].kwargs['metadatas'][0]['zone'], 'Makerspace A')
        store.add.assert_not_called()

    def test_ocr_metadata_keeps_unknown_year_ambiguous(self):
        records = images.image_records(
            {'a': 'SCENE: A workshop poster\nVISIBLE TEXT: Workshop 2025',
             'b': 'SCENE: A room with posters\nVISIBLE TEXT: 2024 and 2025'},
            [{'src': 'a'}, {'src': 'b', 'type': 'photo'}])
        self.assertEqual(records[0]['metadata']['type'], 'poster')
        self.assertEqual(records[0]['metadata']['year'], '2025')
        self.assertEqual(records[0]['metadata']['topic'], 'workshop')
        self.assertEqual(records[1]['metadata']['year'], '')
        self.assertEqual(records[1]['metadata']['type'], 'photo')

    def test_gateway_failure_preserves_successful_descriptions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'photo.png'
            path.write_bytes(b'photo')
            cache = Path(directory) / 'cache.json'
            manifest = [{'src': 'a', 'local_path': str(path)}, {'src': 'b', 'local_path': str(path)}]
            with patch.object(images, 'describe_image', side_effect=[RuntimeError('gateway unavailable'), 'visible text']):
                with self.assertRaisesRegex(RuntimeError, 'actual=1 of 2'):
                    images.describe_all(manifest, cache)
            self.assertEqual(list(json.loads(cache.read_text())['entries']), ['b'])

    def test_explicit_gallery_inventory_is_searchable_without_vision(self):
        record = {"src": "poster.png", "page": "https://innoacademy.engg.hku.hk/workshop/",
                  "heading": "Year 2025", "year": "2025", "type": "poster",
                  "metadata_basis": "gallery section heading", "inventory_page_complete": True}
        result = images.image_records({}, [record])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"]["year"], "2025")
        self.assertEqual(result[0]["metadata"]["topic"], "workshop")
        self.assertTrue(result[0]["metadata"]["inventory_page_complete"])
        self.assertIn("not been transcribed", result[0]["text"])
        self.assertEqual(images.image_records({}, [{"src": "unknown.png"}]), [])

    def test_http_error_is_checked_before_vision(self):
        response = MagicMock()
        response.raise_for_status.side_effect = images.requests.HTTPError('404')
        with patch.object(images.requests, 'get', return_value=response), patch.object(images, 'describe_image') as describe:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, 'actual=1 of 1'):
                    images.describe_all([{'src': 'https://site/missing.png'}], Path(directory) / 'cache.json')
            describe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
