"""Offline ingestion: crawl only the two challenge websites, preserving provenance.

Run from submission_repo: python -m build.scrape
No function in this module is called while answering a question.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

SITES = ['https://innowings.engg.hku.hk/', 'https://innoacademy.engg.hku.hk/']
# Retained starter timeout and crawl ceiling; CLI can explicitly override the ceiling.
REQUEST_TIMEOUT = 20
DEFAULT_MAX_PAGES = 500
# Four-page live probe: workers=1/2/4 took 12.640/4.578/2.641s; all HTTP200.
# Four is an ingestion throughput experiment, not a runtime latency guarantee.
DEFAULT_WORKERS = 4
ASSET_SUFFIXES = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg',
                  '.zip', '.mp4', '.css', '.js', '.xml', '.ico'}


def canonical_url(url: str) -> str:
    """Strip fragments and tracking queries; avoid crawling WordPress search/feed APIs."""
    parts = urlparse(urldefrag(url)[0])
    return urlunparse((parts.scheme.lower(), parts.netloc.lower(), parts.path or '/', '', '', ''))


def allowed(url: str, start_url: str) -> bool:
    parts, root = urlparse(url), urlparse(start_url)
    return (parts.scheme in {'http', 'https'} and parts.netloc == root.netloc
            and Path(parts.path).suffix.lower() not in ASSET_SUFFIXES
            and not any(piece in parts.path.split('/') for piece in ['wp-json', 'wp-admin', 'feed'])
            and not parts.query)


def fetch(url: str, session: requests.Session) -> requests.Response:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    if response.encoding == 'ISO-8859-1':
        response.encoding = response.apparent_encoding
    return response


def sitemap_urls(start_url: str, session: requests.Session) -> list[str]:
    """Read standard and WordPress sitemap indexes, recursively and within the host."""
    origin = f'{urlparse(start_url).scheme}://{urlparse(start_url).netloc}'
    queue = deque([origin + '/sitemap.xml', origin + '/wp-sitemap.xml'])
    seen, pages = set(), set()
    while queue:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            response = fetch(url, session)
            if urlparse(response.url).netloc != urlparse(start_url).netloc:
                continue
            root = ET.fromstring(response.content)
        except (requests.RequestException, ET.ParseError):
            continue  # Not every server publishes either sitemap convention.
        locations = [node.text.strip() for node in root.iter()
                     if node.tag.rsplit('}', 1)[-1] == 'loc' and node.text]
        if root.tag.rsplit('}', 1)[-1] == 'sitemapindex':
            queue.extend(loc for loc in locations
                         if urlparse(loc).netloc == urlparse(start_url).netloc)
        else:
            pages.update(canonical_url(loc) for loc in locations if allowed(loc, start_url))
    return sorted(pages)


def crawl(start_url: str, max_pages: int | None = DEFAULT_MAX_PAGES,
          html_cache: dict | None = None, report: dict | None = None,
          workers: int = DEFAULT_WORKERS, on_page=None, existing_pages: dict | None = None) -> list[str]:
    """Discover pages with bounded parallel ingestion and explicit completeness reporting."""
    if (max_pages is not None and max_pages < 1) or workers < 1:
        raise ValueError(f'crawl limits max_pages>=1 and workers>=1; actual max_pages={max_pages}, '
                         f'workers={workers}; ask for positive limits')
    existing_pages = existing_pages or {}
    started = time.monotonic()
    errors = []
    cache = html_cache if html_cache is not None else {}
    session = requests.Session()
    session.headers['User-Agent'] = 'InnoWingChallengeLearningBot/1.0'
    seeds = sitemap_urls(start_url, session)
    queue = deque(dict.fromkeys([canonical_url(start_url), *seeds]))
    seen, out = set(), []

    def download(url):
        if url in existing_pages and 'links' in existing_pages[url]:
            return url, None, None
        try:
            # Each worker owns its session; requests.Session is not shared across threads.
            with requests.Session() as worker_session:
                worker_session.headers['User-Agent'] = session.headers['User-Agent']
                return url, fetch(url, worker_session), None
        except requests.RequestException as exc:
            return url, None, exc

    def update_report():
        pending = len(set(queue) - seen)
        if report is not None:
            report.update({'site': start_url, 'sitemap_pages': len(seeds), 'visited': len(out),
                           'discovered': len(set(queue) | seen), 'pending': pending,
                           'failures': errors, 'truncated': bool(pending),
                           'max_pages': max_pages, 'workers': workers, 'downloaded': len(cache),
                           'elapsed_seconds': time.monotonic() - started})
        return pending

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while queue and (max_pages is None or len(out) < max_pages):
            batch = []
            while queue and len(batch) < (min(workers, max_pages - len(out)) if max_pages is not None else workers):
                url = queue.popleft()
                if url not in seen:
                    seen.add(url)
                    batch.append(url)
            for url, response, error in pool.map(download, batch):
                if error is not None:
                    errors.append({'url': url, 'error': type(error).__name__})
                    print(f'crawl failed url={url}: {error}; retry ingestion to recover this page', file=sys.stderr)
                    continue
                if response is None:
                    out.append(url)
                    queue.extend(link for link in existing_pages[url]['links']
                                 if allowed(link, start_url) and link not in seen)
                    update_report()
                    continue
                if not allowed(response.url, start_url):
                    continue
                if 'html' not in response.headers.get('Content-Type', '').lower():
                    continue
                cache[url] = response.text
                out.append(url)
                for a in BeautifulSoup(cache[url], 'html.parser').select('a[href]'):
                    link = urljoin(url, a['href'])
                    if allowed(link, start_url):
                        link = canonical_url(link)
                        if link not in seen:
                            queue.append(link)
                update_report()
                if on_page is not None:
                    on_page(url, cache[url])
    pending = update_report()
    if pending:
        print(f'crawl incomplete max_pages={max_pages}, indexed={len(out)}, '
              f'pending={pending}; rerun with a higher --max-pages', file=sys.stderr)
    return out


def extract(html: str, url: str) -> dict:
    """Keep article/main content, paragraph boundaries, tables, headings and image records."""
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.get_text(' ', strip=True) if soup.title else ''
    links = sorted({canonical_url(urljoin(url, a['href'])) for a in soup.select('a[href]')
                    if allowed(urljoin(url, a['href']), url)})
    for node in soup.select('script, style, noscript, nav, footer, header, '
                            '[role="navigation"], .site-header, .site-footer, '
                            '.elementor-location-header, .elementor-location-footer'):
        node.decompose()
    body = soup.select_one('main, article, #primary, #content, .site-content') or soup
    images, seen = [], set()
    for img in body.select('img'):
        src = img.get('data-src') or img.get('data-lazy-src') or img.get('src')
        if not src or src.startswith('data:'):
            continue
        # Prefer full-resolution linked images over resized WordPress thumbnails.
        parent = img.find_parent('a', href=True)
        if parent and Path(urlparse(parent['href']).path).suffix.lower() in ASSET_SUFFIXES - {'.pdf', '.zip', '.mp4', '.css', '.js', '.xml', '.ico'}:
            src = parent['href']
        src = urljoin(url, src)
        if src in seen:
            continue
        seen.add(src)
        fig = img.find_parent('figure')
        caption = fig.find('figcaption') if fig else None
        heading = img.find_previous(re.compile(r'^h[1-6]$'))
        record = {'src': src, 'alt': img.get('alt', ''),
                  'caption': caption.get_text(' ', strip=True) if caption else '',
                  'heading': heading.get_text(' ', strip=True) if heading else '',
                  'page': url, 'title': title}
        # The workshop gallery explicitly groups posters under Year headings.
        # Require shared section ancestry: a footer image must not inherit a year.
        section = img.find_parent(class_='elementor-top-section')
        heading_section = heading.find_parent(class_='elementor-top-section') if heading else None
        year = re.fullmatch(r'Year\s+((?:19|20)\d{2})', record['heading'])
        workshop_gallery = (urlparse(url).netloc == 'innoacademy.engg.hku.hk'
                            and urlparse(url).path.rstrip('/') == '/workshop')
        if workshop_gallery and section is not None and heading_section is section and year:
            record.update({'type': 'poster', 'topic': 'workshop', 'year': year.group(1),
                           'metadata_basis': 'gallery section heading',
                           'inventory_complete': True})
        images.append(record)
    for br in body.select('br'):
        br.replace_with('\n')
    lines = [' '.join(line.split()) for line in body.get_text('\n', strip=True).splitlines()]
    text = '\n'.join(line for line in lines if line)
    return {'url': url, 'title': title, 'text': text, 'images': images, 'links': links,
            'inventory_complete': any(im.get('inventory_complete') for im in images)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument('--all-pages', action='store_true', help='Exhaust the same-host link graph without a page ceiling')
    parser.add_argument('--resume', action='store_true', help='Reuse saved pages with saved links; refresh older records lacking links')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Concurrent downloads; measured at 1, 2, 4 workers before selecting 4')
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    pages, failures, reports = [], [], []
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume and (args.output / 'pages.json').exists():
        pages = json.loads((args.output / 'pages.json').read_text(encoding='utf-8'))
    existing_pages = {page['url']: page for page in pages}

    def save():
        images = [image for page in pages for image in page['images']]
        outputs = [('crawl_report.json', reports)]
        if pages:
            outputs.extend([('pages.json', pages), ('images.json', images)])
        for name, data in outputs:
            target = args.output / name
            temp = target.with_suffix('.tmp')
            temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            temp.replace(target)

    def collect(url, html):
        page = extract(html, url)
        if page['text'] or page['images']:
            pages[:] = [p for p in pages if p['url'] != url]
            pages.append(page)
            save()  # A stopped crawl still leaves a usable, explicitly partial corpus.

    for site in SITES:
        report = {'site': site, 'truncated': True, 'visited': 0}
        reports.append(report)
        urls = crawl(site, None if args.all_pages else args.max_pages, report=report,
                     workers=args.workers, on_page=collect, existing_pages=existing_pages)
        save()
        if not urls:
            failures.append(site)
    if not pages:
        raise RuntimeError(f'collected=0 pages, sites={SITES}; check network and rerun; existing data preserved')
    images = [image for page in pages for image in page['images']]
    print(f'{len(pages)} pages, {len(images)} image references; failed sites={failures}')
    if failures:
        raise RuntimeError(f'failed_sites={len(failures)}, actual={failures}; rerun to complete the corpus')


if __name__ == '__main__':
    main()
