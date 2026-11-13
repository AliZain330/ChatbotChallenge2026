"""Offline tests: no website or gateway requests."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from build.index import build_index, chunk, page_metadata, prepare_chunks
from build.scrape import allowed, canonical_url, crawl, extract, sitemap_urls, main as scrape_main


class ExtractionTests(unittest.TestCase):
    def test_content_scope_and_lazy_images(self):
        page = extract('''<html><title>Laser cutter</title><nav>Menu noise</nav>
        <main><h1>Safe use</h1><p>PVC and teflon are banned.</p>
        <table><tr><td>Power</td><td>80W</td></tr></table>
        <figure><a href="/poster.png"><img data-src="/small.png" alt="Safety"></a>
        <figcaption>Training poster</figcaption></figure></main>
        <footer>Footer noise</footer></html>''', 'https://innowings.engg.hku.hk/equipment/')
        self.assertNotIn('noise', page['text'])
        self.assertIn('PVC and teflon', page['text'])
        self.assertIn('80W', page['text'])
        self.assertEqual(page['images'][0]['src'], 'https://innowings.engg.hku.hk/poster.png')
        self.assertEqual(page['images'][0]['caption'], 'Training poster')
        self.assertEqual(page['images'][0]['heading'], 'Safe use')

    def test_workshop_inventory_uses_same_section_year_and_excludes_footer(self):
        html = '''<main><section class="elementor-top-section"><h2>Year 2025</h2>
        <img src="/poster-a.png"><img src="/poster-b.png"></section>
        <section class="elementor-top-section"><img src="/social.png"></section>
        <section class="elementor-top-section"><h2>Year 2024</h2>
        <img src="/poster-c.png"></section></main>'''
        page = extract(html, 'https://innoacademy.engg.hku.hk/workshop/')
        self.assertTrue(page['inventory_complete'])
        self.assertEqual([im.get('year') for im in page['images']], ['2025', '2025', None, '2024'])
        self.assertEqual(page['images'][0]['metadata_basis'], 'gallery section heading')
        self.assertNotIn('inventory_complete', page['images'][2])
        unrelated = extract(html, 'https://innoacademy.engg.hku.hk/another-page/')
        self.assertFalse(unrelated['inventory_complete'])
        self.assertFalse(any(im.get('type') == 'poster' for im in unrelated['images']))

    def test_scope_rejects_external_queries_assets_and_scripts(self):
        base = 'https://innowings.engg.hku.hk/'
        for url in ['https://example.com/', base + '?s=printer',
                    base + 'photo.JPG', 'javascript:alert(1)', base + 'wp-json/']:
            self.assertFalse(allowed(url, base), url)
        self.assertTrue(allowed(base + 'equipment/#laser', base))
        self.assertEqual(canonical_url(base + 'equipment/?utm_source=x#laser'), base + 'equipment/')

    def test_sitemap_index_recurses_and_deduplicates(self):
        session = Mock()
        docs = {
            'https://example.com/sitemap.xml': '<sitemapindex><sitemap><loc>https://example.com/pages.xml</loc></sitemap></sitemapindex>',
            'https://example.com/wp-sitemap.xml': '<urlset/>',
            'https://example.com/pages.xml': '<urlset><url><loc>https://example.com/a/</loc></url><url><loc>https://outside.com/</loc></url></urlset>',
        }
        def response(url):
            return Mock(url=url, content=docs[url].encode())
        with patch('build.scrape.fetch', side_effect=lambda url, _: response(url)):
            self.assertEqual(sitemap_urls('https://example.com/', session), ['https://example.com/a/'])

    def test_crawl_does_not_download_external_links(self):
        def response(url, _):
            return Mock(url=url, headers={'Content-Type': 'text/html'},
                        text='<main>Hello</main><a href="/two/">two</a><a href="https://outside.com/">external</a>')
        with patch('build.scrape.sitemap_urls', return_value=[]), patch('build.scrape.fetch', side_effect=response) as fetch:
            urls = crawl('https://example.com/')
        self.assertEqual(urls, ['https://example.com/', 'https://example.com/two/'])
        self.assertEqual(fetch.call_count, 2)


    def test_network_failure_does_not_replace_existing_corpus_with_empty_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            pages = Path(temporary) / 'pages.json'
            pages.write_text('[{"url":"previous"}]', encoding='utf-8')
            with patch('sys.argv', ['scrape', '--output', temporary]), patch('build.scrape.crawl', return_value=[]):
                with self.assertRaisesRegex(RuntimeError, 'collected=0'):
                    scrape_main()
            self.assertEqual(pages.read_text(encoding='utf-8'), '[{"url":"previous"}]')

    def test_unlimited_resume_reuses_saved_links_without_http(self):
        records = {
            'https://example.com/': {'links': ['https://example.com/two/']},
            'https://example.com/two/': {'links': []},
        }
        report = {}
        with patch('build.scrape.sitemap_urls', return_value=[]), patch('build.scrape.fetch') as fetch:
            urls = crawl('https://example.com/', max_pages=None, existing_pages=records, report=report)
        self.assertEqual(urls, list(records))
        self.assertFalse(report['truncated'])
        self.assertEqual(report['downloaded'], 0)
        fetch.assert_not_called()


class ChunkTests(unittest.TestCase):
    def test_invalid_limits_fail_with_values(self):
        for size, overlap in [(0, 0), (10, 10), (10, -1)]:
            with self.assertRaisesRegex(ValueError, f'size={size}'):
                chunk('Text', size, overlap)

    def test_paragraphs_and_long_words_are_preserved(self):
        text = 'First paragraph.\nSecond paragraph with a complete fact.'
        self.assertEqual(chunk(text), [text])
        long = ' '.join(f'word{i}' for i in range(90))
        pieces = chunk(long, size=80, overlap=15)
        self.assertTrue(all(0 < len(p) <= 80 for p in pieces))
        for word in long.split():
            self.assertTrue(any(word in p.split() for p in pieces), word)
        self.assertEqual(''.join(chunk('x' * 185, 80, 0)), 'x' * 185)
        self.assertEqual(chunk(' \n '), [])

    def test_metadata_ignores_footer_year(self):
        page = {'url': 'https://example.com/workshop-2025/', 'title': 'Workshop 2025',
                'text': 'Copyright 2026'}
        self.assertEqual(page_metadata(page)['year'], '2025')
        self.assertEqual(page_metadata(page)['page_type'], 'workshop')
        page['title'] = 'Workshop 2024 and 2025'
        self.assertEqual(page_metadata(page)['year'], '')

    def test_failed_embeddings_preserve_existing_collection(self):
        page = {'url': 'https://example.com/a/', 'title': 'A', 'text': 'Fact'}
        with patch('build.embeddings.cached_embed', side_effect=RuntimeError('gateway unavailable')), patch('bot.store.get_store') as store:
            with self.assertRaisesRegex(RuntimeError, 'gateway unavailable'):
                build_index([page])
        store.assert_not_called()

    def test_index_uses_gateway_vectors_and_chroma_advertised_batch_limit(self):
        page = {'url': 'https://example.com/a/', 'title': 'A', 'text': 'Fact'}
        store = Mock()
        store._client.get_max_batch_size.return_value = 1
        with patch('build.embeddings.cached_embed', return_value=[[1.0, 0.0]]) as embed, patch('bot.store.get_store', return_value=store):
            self.assertIs(build_index([page]), store)
        embed.assert_called_once_with(['A\nFact'])
        store.upsert.assert_called_once()
        self.assertEqual(store.upsert.call_args.kwargs['embeddings'], [[1.0, 0.0]])
        self.assertEqual(store.upsert.call_args.kwargs['metadatas'][0]['url'], page['url'])

    def test_chunk_ids_are_stable_deduplicated_and_source_specific(self):
        page = {'url': 'https://example.com/a/', 'title': 'A', 'text': 'Fact'}
        a = prepare_chunks([page, page])
        self.assertEqual(len(a), 1)
        self.assertEqual(a, prepare_chunks([page]))
        self.assertNotEqual(a[0]['id'], prepare_chunks([{**page, 'url': 'https://example.com/b/'}])[0]['id'])
        self.assertEqual(a[0]['metadata']['position'], 0)


if __name__ == '__main__':
    unittest.main()
