"""Paragraph-aware text chunks and persistent gateway embeddings.

Run after scraping: python -m build.index
The 800/100 character settings are the challenge starter baseline, not tuned claims.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHUNK_SIZE, OVERLAP = 800, 100


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Pack paragraphs, split long paragraphs at words, retain contextual overlap."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(f'chunk limits size>0 and 0<=overlap<size; actual size={size}, '
                         f'overlap={overlap}; choose a smaller nonnegative overlap')
    paragraphs = [' '.join(line.split()) for line in text.splitlines() if line.strip()]
    pieces, current = [], ''
    for paragraph in paragraphs:
        while paragraph:
            separator = '\n' if current else ''
            available = size - len(current) - len(separator)
            if len(paragraph) <= available:
                current += separator + paragraph
                break
            if current:
                pieces.append(current)
                tail = current[-overlap:] if overlap else ''
                # Exclude a partial first word from the overlap.
                current = tail.split(' ', 1)[1] if ' ' in tail else ''
                available = size - len(current) - (1 if current else 0)
            end = paragraph.rfind(' ', 0, available + 1)
            if end <= 0:
                end = available  # Long URLs/identifiers must still fit the size limit.
            current += ('\n' if current else '') + paragraph[:end]
            paragraph = paragraph[end:].lstrip()
            if paragraph:
                pieces.append(current)
                tail = current[-overlap:] if overlap else ''
                current = tail.split(' ', 1)[1] if ' ' in tail else ''
    if current and (not pieces or current != pieces[-1]):
        pieces.append(current)
    return pieces


def page_metadata(page: dict) -> dict:
    """Conservative year/type tags: never infer an event year from the footer."""
    identity = f"{page.get('title', '')} {page['url']}".lower()
    years = set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', identity))
    page_type = 'page'
    for label, words in [('equipment', ['equipment', 'printer', 'laser-cut']),
                         ('workshop', ['workshop']), ('funding', ['funding']),
                         ('event', ['event', 'innoshow', 'pitch']),
                         ('venue', ['venue', 'makerspace']), ('news', ['news'])]:
        if any(word in identity for word in words):
            page_type = label
            break
    return {'url': page['url'], 'title': page.get('title', ''), 'kind': 'text',
            'year': next(iter(years)) if len(years) == 1 else '',
            'page_type': page.get('page_type', page_type),
            'site': page['url'].split('/')[2] if '://' in page['url'] else 'local'}


def prepare_chunks(pages: list[dict]) -> list[dict]:
    """Stable IDs distinguish sources and positions; identical page URLs are merged."""
    records, seen = [], set()
    for page in pages:
        metadata = page_metadata(page)
        for position, piece in enumerate(chunk(page['text'])):
            text = f"{page.get('title', '')}\n{piece}".strip()
            key = sha256(f"{page['url']}\0{position}\0{text}".encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            records.append({'id': f'text_{key}', 'text': text,
                            'metadata': {**metadata, 'position': position}})
    return records


def build_index(pages: list[dict], reset: bool = True, path: str = None):
    """Embed all chunks before replacing the collection, avoiding loss on API failure."""
    from build.embeddings import cached_embed
    from bot.store import get_store, DEFAULT_PATH

    records = prepare_chunks(pages)
    if not records:
        raise ValueError('index requires >=1 nonempty chunk, actual=0; scrape pages before indexing')
    vectors = cached_embed([record['text'] for record in records])
    if len(vectors) != len(records):
        raise RuntimeError(f'embedding count expected={len(records)}, actual={len(vectors)}; rerun embedding')
    store = get_store(path=path or DEFAULT_PATH, reset=reset)
    # Chroma advertises its actual accepted write size; do not invent a batch limit.
    maximum = store._client.get_max_batch_size()
    for start in range(0, len(records), maximum):
        batch = records[start:start + maximum]
        store.upsert(ids=[r['id'] for r in batch], documents=[r['text'] for r in batch],
                     embeddings=vectors[start:start + maximum],
                     metadatas=[r['metadata'] for r in batch])
    print(f'indexed {len(records)} text chunks from {len(pages)} pages')
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', type=Path, default=Path('data/pages.json'))
    parser.add_argument('--store', default=None)
    args = parser.parse_args()
    pages = json.loads(args.pages.read_text(encoding='utf-8-sig'))
    build_index(pages, path=args.store)


if __name__ == '__main__':
    main()
