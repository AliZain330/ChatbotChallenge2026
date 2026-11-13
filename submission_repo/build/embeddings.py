"""Transactional document-embedding cache for ingestion, never runtime queries.

SQLite commits only each new successful batch, avoiding repeated rewrites of
all prior vectors. Failed API batches leave committed progress intact.
"""
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Optional

from bot import llm


def _validate(vector: list, expected: int = None) -> int:
    if not isinstance(vector, list) or not vector or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in vector):
        raise RuntimeError('cached embedding required=nonempty finite numeric vector, actual=invalid; remove invalid cache and rebuild')
    if expected is not None and len(vector) != expected:
        raise RuntimeError(f'cached embedding dimensions required={expected}, actual={len(vector)}; use matching model or rebuild cache')
    return len(vector)


def _open_cache(connection: sqlite3.Connection, identity: dict, legacy: Path) -> Optional[int]:
    connection.executescript('''
        CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, vector TEXT NOT NULL);
    ''')
    metadata = dict(connection.execute('SELECT name, value FROM metadata'))
    if metadata and any(metadata.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f'embedding cache identity required={identity}, actual=mismatched; use correct provider/model cache or rebuild')
    with connection:
        connection.executemany('INSERT OR IGNORE INTO metadata VALUES (?, ?)', identity.items())
    dimensions = int(metadata['dimensions']) if 'dimensions' in metadata else None
    # Preserve the earlier JSON cache's successful gateway calls. Validate all
    # legacy vectors before importing; a bad migration changes no cached rows.
    if legacy.exists() and not metadata.get('legacy_imported'):
        payload = json.loads(legacy.read_text(encoding='utf-8'))
        if any(payload.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f'legacy embedding cache identity required={identity}, actual=mismatched; use matching cache or move invalid JSON aside')
        vectors = payload.get('vectors', {})
        legacy_dimensions = payload.get('dimensions', dimensions)
        for vector in vectors.values():
            legacy_dimensions = _validate(vector, legacy_dimensions)
            _validate(vector, dimensions)
        dimensions = legacy_dimensions if dimensions is None else dimensions
        with connection:
            connection.executemany('INSERT OR IGNORE INTO embeddings VALUES (?, ?)',
                                   [(key, json.dumps(vector, separators=(',', ':'), allow_nan=False))
                                    for key, vector in vectors.items()])
            if dimensions is not None:
                connection.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('dimensions', str(dimensions)))
            connection.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('legacy_imported', 'yes'))
    return dimensions


def _quota_delay(error: RuntimeError) -> Optional[float]:
    """Accept only provider-reported minute-quota delays, never daily limits."""
    message = str(error)
    identifiers = re.findall(r'\bquotaId=([^;,\s]+)', message)
    delay = re.search(r'\bretry_after=(\d+(?:\.\d+)?)s(?:[;\s]|$)', message)
    if ('actual=429' not in message or not identifiers or 'PerDay' in message
            or not all('PerMinute' in identifier for identifier in identifiers) or not delay):
        return None
    seconds = float(delay.group(1))
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def cached_embed(texts: list[str], *, embed_fn=None, cache_dir: Path = Path('data'),
                 wait_for_quota: bool = False) -> list[list[float]]:
    """Return vectors in input order; checkpoint missing unique texts per batch.

    Provider/model/document-task identity lives in the database; rows use SHA256
    of exact UTF-8 document text. Query embeddings must not enter this cache.
    Optional ingestion-only quota waits use exactly a provider's retry hint;
    default behavior propagates failures immediately and runtime never calls it.
    """
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('embedding documents required=nonempty strings, actual=empty/non-string; clean corpus before indexing')
    provider, model = llm.PROVIDER, llm.EMBED_DEPLOYMENT
    safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', f'{provider}-{model}')
    path = Path(cache_dir) / f'embeddings-{safe_name}.sqlite3'
    identity = {'provider': provider, 'model': model, 'task_type': 'RETRIEVAL_DOCUMENT'}
    unique = {hashlib.sha256(text.encode('utf-8')).hexdigest(): text for text in texts}
    if provider == 'gemini':
        from bot.gemini import MAX_EMBED_BATCH
        batch_size = MAX_EMBED_BATCH
    else:
        # Receipt: supplied bot.llm.embed's existing Azure request batch default.
        batch_size = 256
    embedding = embed_fn or llm.embed
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        dimensions = _open_cache(connection, identity, path.with_suffix('.json'))
        cache, missing = {}, []
        for key, text in unique.items():
            row = connection.execute('SELECT vector FROM embeddings WHERE hash=?', (key,)).fetchone()
            if row is None:
                missing.append((key, text))
            else:
                vector = json.loads(row[0])
                dimensions = _validate(vector, dimensions)
                cache[key] = vector
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            while True:
                try:
                    vectors = embedding([text for _, text in batch], task_type='RETRIEVAL_DOCUMENT')
                    break
                except RuntimeError as error:
                    delay = _quota_delay(error) if provider == 'gemini' and wait_for_quota else None
                    if delay is None:
                        raise
                    print(f'[embedding ingestion] batch inputs={len(batch)}, provider minute quota reached; '
                          f'waiting retry_after={delay:g}s before the same batch', file=sys.stderr)
                    time.sleep(delay)
            if len(vectors) != len(batch):
                raise RuntimeError(f'embedding cache batch count required={len(batch)}, actual={len(vectors)}; rerun ingestion after fixing gateway')
            for vector in vectors:
                dimensions = _validate(vector, dimensions)
            with connection:
                connection.executemany('INSERT OR REPLACE INTO embeddings VALUES (?, ?)',
                                       [(key, json.dumps(vector, separators=(',', ':'), allow_nan=False))
                                        for (key, _), vector in zip(batch, vectors)])
                connection.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('dimensions', str(dimensions)))
            cache.update({key: vector for (key, _), vector in zip(batch, vectors)})
    return [cache[hashlib.sha256(text.encode('utf-8')).hexdigest()] for text in texts]
