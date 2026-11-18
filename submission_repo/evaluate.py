"""Execute the real development questions. Exact match is a lower bound, not a judge."""
import argparse
from contextlib import redirect_stderr
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bot.answer import rag_answer, retrieve


def normalise(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def evaluate(questions: list[dict], retrieval_only: bool = False, local_pages: Path = None) -> dict:
    local_chunks = None
    if local_pages:
        from build.index import prepare_chunks
        local_chunks = prepare_chunks(json.loads(local_pages.read_text(encoding="utf-8")))
    rows = []
    for item in questions:
        row = {**item, "expected": item["answer"]}
        started = time.monotonic()
        diagnostics = io.StringIO()
        with redirect_stderr(diagnostics):
            try:
                if retrieval_only:
                    if local_chunks is None:
                        chunks = retrieve(item["question"])
                    else:
                        from bot.answer import lexical_rank, CONFIG
                        chunks = lexical_rank(item["question"], local_chunks)[:CONFIG["k"]]
                    row["sources"] = [c["metadata"] for c in chunks]
                    row["evidence"] = [c["text"] for c in chunks]
                    row["expected_phrase_present"] = normalise(item["answer"]) in normalise(" ".join(c["text"] for c in chunks))
                    row["answer"] = ""
                else:
                    row["answer"] = rag_answer(item["question"])
                    row["exact_match"] = normalise(row["answer"]) == normalise(item["answer"])
                row["error"] = None
            except Exception as exc:
                row.update(answer="", exact_match=False, error=f"{type(exc).__name__}: {exc}")
        warning_text = diagnostics.getvalue()
        row["diagnostics"] = warning_text.splitlines()
        row["semantic_fallback_observed"] = "[retrieval] semantic call failed" in warning_text
        sys.stderr.write(warning_text)
        row["seconds"] = time.monotonic() - started
        row["within_30_seconds"] = row["seconds"] <= 30  # Fixed grading limit.
        row["manual_score"] = None  # A human should score 0, 0.5 or 1; never invent grades.
        rows.append(row)
        print(f"Lv{item['level']} {row['seconds']:.2f}s: {row['answer'] or row.get('error') or '[retrieval only]'}")
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": ("local-lexical-retrieval" if local_pages else "retrieval") if retrieval_only else "answers", "questions": len(rows),
            "completed": sum(r["error"] is None for r in rows),
            "semantic_fallback_questions": sum(r["semantic_fallback_observed"] for r in rows),
            "exact_matches": sum(r.get("exact_match", False) for r in rows),
            "note": "Exact matches are not semantic grades. Missing-key errors are blocked runs, not model accuracy. Phrase presence does not establish that evidence answers the question.",
            "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("dev_set.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--local-pages", type=Path, help="offline lexical diagnosis of pages.json; requires --retrieval-only")
    args = parser.parse_args()
    if args.local_pages and not args.retrieval_only:
        parser.error("--local-pages requires --retrieval-only; it does not run the answer model")
    report = evaluate(json.loads(args.questions.read_text(encoding="utf-8")), args.retrieval_only, args.local_pages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed {report['completed']}/{report['questions']}; exact matches {report['exact_matches']}; saved {args.output}")
