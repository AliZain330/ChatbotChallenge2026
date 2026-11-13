"""Check the configured self-training provider without changing the original Azure checker."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from bot.llm import PROVIDER, CHAT_DEPLOYMENT, VISION_DEPLOYMENT, EMBED_DEPLOYMENT, chat, embed, describe_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("../Labs/images/Brainstorming aerial.JPG"))
    parser.add_argument("--output", type=Path, default=Path("reports/provider_check.json"))
    args = parser.parse_args()
    checks = []
    for name, fn in [
        ("chat", lambda: chat([{"role": "user", "content": "Reply with the single word connected."}])),
        ("embeddings", lambda: [len(v) for v in embed(["additive manufacturing", "3D printer"])]),
        ("vision", lambda: describe_image(str(args.image), "Transcribe the large words on the rear wall exactly. Give the words only.")),
    ]:
        started = time.monotonic()
        try:
            result = fn()
            row = {"check": name, "passed": True, "result": result}
        except Exception as exc:
            row = {"check": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        row["seconds"] = time.monotonic() - started
        checks.append(row)
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'} {row['seconds']:.3f}s {row.get('result', row.get('error'))}")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "provider": PROVIDER,
              "models": {"chat": CHAT_DEPLOYMENT, "vision": VISION_DEPLOYMENT, "embedding": EMBED_DEPLOYMENT},
              "independent_adaptation": PROVIDER == "gemini", "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(0 if all(row['passed'] for row in checks) else 1)
