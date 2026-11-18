"""Expand the existing Gemini index each minute without resetting it.

Run from submission_repo: python -m build.expand
Successful batches are immediately searchable. Rerunning skips indexed IDs and
reuses the embedding cache. Daily quota/authentication errors stop the run.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from bot.llm import PROVIDER, EMBED_DEPLOYMENT, embed
from bot.gemini import MAX_EMBED_BATCH
from bot.store import get_store
from build.embeddings import cached_embed
from build.scrape import write_snapshot

# Receipt: the user explicitly requested an attempt each minute. Each request
# uses the existing Gemini batch limit; provider retry hints remain authoritative.
REQUEST_INTERVAL_SECONDS = 60


def expand_records(records: list[dict], store, progress_path: Path,
                   batch_size: int = MAX_EMBED_BATCH) -> dict:
    if not 1 <= batch_size <= MAX_EMBED_BATCH:
        raise ValueError(f'embedding batch min=1, max={MAX_EMBED_BATCH}, actual={batch_size}; choose an allowed size')
    unique = {record['id']: record for record in records}
    if not unique:
        raise ValueError('embedding target min=1 record, actual=0; build local text/image records first')
    known = set(store.get(include=[])['ids'])
    targets = set(unique)
    pending = [record for identity, record in unique.items() if identity not in known]
    state = {'provider': PROVIDER, 'model': EMBED_DEPLOYMENT,
             'target_records': len(targets), 'added_this_run': 0}

    def checkpoint(status: str, *, required: bool = False, error: str = None) -> None:
        state.update(status=status, indexed_target_records=len(targets & known),
                     collection_records=len(known),
                     updated_at=datetime.now(timezone.utc).isoformat())
        if error:
            state['error'] = error
        write_snapshot(progress_path, state, required=required)

    checkpoint('running')
    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            last_request = None

            def request_embeddings(texts: list[str], **kwargs) -> list[list[float]]:
                nonlocal last_request
                last_request = time.monotonic()
                return embed(texts, **kwargs)

            vectors = cached_embed([record['text'] for record in batch],
                                   embed_fn=request_embeddings, wait_for_quota=True)
            if len(vectors) != len(batch):
                raise RuntimeError(f'index embedding count required={len(batch)}, actual={len(vectors)}; fix the provider before resuming')
            store.upsert(ids=[record['id'] for record in batch],
                         documents=[record['text'] for record in batch],
                         metadatas=[record['metadata'] for record in batch], embeddings=vectors)
            known.update(record['id'] for record in batch)
            state['added_this_run'] += len(batch)
            checkpoint('running')
            print(f"[semantic index] {state['indexed_target_records']}/{len(targets)} target records; "
                  f"collection={len(known)}, added={state['added_this_run']}", flush=True)
            if start + len(batch) < len(pending) and last_request is not None:
                remaining = REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_request)
                if remaining > 0:
                    checkpoint('waiting_for_next_minute')
                    time.sleep(remaining)
        checkpoint('complete', required=True)
        return state
    except KeyboardInterrupt:
        checkpoint('interrupted')
        raise
    except Exception as exc:
        checkpoint('blocked', error=f'{type(exc).__name__}: {exc}')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-records', type=Path, default=Path('data/text_records.json'))
    parser.add_argument('--image-records', type=Path, default=Path('data/image_records.json'))
    parser.add_argument('--progress', type=Path, default=Path('data/embedding_progress.json'))
    args = parser.parse_args()
    if PROVIDER != 'gemini':
        parser.error(f'provider required=gemini, actual={PROVIDER}; set LLM_PROVIDER=gemini in the local .env')
    records = []
    for source in [args.text_records, args.image_records]:
        if source.exists():
            records.extend(json.loads(source.read_text(encoding='utf-8')))
    try:
        expand_records(records, get_store(reset=False), args.progress)
    except Exception as exc:
        print(f'[semantic index stopped] {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
