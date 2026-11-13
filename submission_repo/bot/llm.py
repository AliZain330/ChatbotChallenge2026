"""Azure gateway client plus explicit Gemini self-training mode.

Three deployments, three clients, because this gateway takes the full
path (deployment, operation and api-version) as the endpoint, so one
client can only ever talk to one deployment.

Reads settings from the environment. Copy .env.example to .env and
fill in the values you were given.
"""
import base64
import mimetypes
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from openai import AzureOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
CHAT_BASE   = os.getenv("AZURE_CHAT_BASE",  "https://api-iw.azure-api.net/sig-shared-jpeast-increased")
EMBED_BASE  = os.getenv("AZURE_EMBED_BASE", "https://api-iw.azure-api.net/sig-embedding")

CHAT_DEPLOYMENT   = os.getenv("CHAT_DEPLOYMENT",   "gpt-4o-mini")
VISION_DEPLOYMENT = os.getenv("VISION_DEPLOYMENT", "gpt-5-mini")
EMBED_DEPLOYMENT  = os.getenv("EMBED_DEPLOYMENT",  "text-embedding-3-small")

# Gemini is deliberately opt-in: official challenge runtime permits Azure only.
PROVIDER = os.getenv("LLM_PROVIDER", "azure").strip().lower()
if PROVIDER not in {"azure", "gemini"}:
    raise ValueError(f"LLM_PROVIDER allowed=azure/gemini, actual={PROVIDER!r}; choose a supported provider")
if PROVIDER == "gemini":
    # Receipt: Google pricing/model catalogue checked 2026-11-13. These models
    # list free-tier access; the key's project must remain on the free tier.
    CHAT_DEPLOYMENT = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
    VISION_DEPLOYMENT = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
    EMBED_DEPLOYMENT = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    allowed_chat = {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"}
    if CHAT_DEPLOYMENT not in allowed_chat or VISION_DEPLOYMENT not in allowed_chat or EMBED_DEPLOYMENT != "gemini-embedding-001":
        raise ValueError("Gemini model allowed=reviewed free-tier Flash-Lite/embedding-001, actual=unapproved model; check pricing before extending the allowlist")

# The chat route has no /openai segment; the embedding route does. This
# asymmetry is real, confirmed against the live gateway. Do not "tidy"
# the two into matching, or one of them will 404.
CHAT_URL   = f"{CHAT_BASE}/deployments/{CHAT_DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
VISION_URL = f"{CHAT_BASE}/deployments/{VISION_DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
EMBED_URL  = f"{EMBED_BASE}/openai/deployments/{EMBED_DEPLOYMENT}/embeddings?api-version={API_VERSION}"

# .strip() matters: pasting a key into a shell or .env often picks up a
# trailing newline, and that alone produces a 401.
_KEY = os.environ.get("AZURE_OPENAI_KEY", "").strip()

# Clients are built lazily, on first real use, not here at import time.
# The openai SDK itself raises if api_key is empty, and this module is
# imported by main.py before main() gets a chance to run its own error
# handling. Building clients eagerly meant a missing key crashed with a
# raw traceback on every single run, including `python main.py` with no
# arguments. check_setup.py is what should catch a missing key, in a
# readable way, before you ever run main.py for real.
_clients: dict = {}
# Receipt: main.py and competition rules specify 30 seconds per question.
QUESTION_TIME_LIMIT = 30
_deadline = ContextVar("gateway_deadline", default=None)


@contextmanager
def question_budget():
    token = _deadline.set(time.monotonic() + QUESTION_TIME_LIMIT)
    try:
        yield
    finally:
        _deadline.reset(token)


def _timeout() -> float:
    deadline = _deadline.get()
    remaining = QUESTION_TIME_LIMIT if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"question max={QUESTION_TIME_LIMIT}s, remaining={remaining:.3f}s; reduce latency")
    return remaining


def _request_client(url: str) -> AzureOpenAI:
    client = _client(url)
    deadline = _deadline.get()
    if deadline is None:
        return client
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"question deadline max={QUESTION_TIME_LIMIT}s, remaining={remaining:.3f}s; "
                           "reduce retrieval/context or gateway latency")
    # Disable automatic retries, which otherwise hide delays beyond the grading limit.
    return client.with_options(timeout=remaining, max_retries=0)



def _client(url: str) -> AzureOpenAI:
    if url not in _clients:
        if not _KEY:
            raise RuntimeError(
                "AZURE_OPENAI_KEY is not set. Copy .env.example to .env "
                "and fill in your key, then run check_setup.py."
            )
        _clients[url] = AzureOpenAI(azure_endpoint=url, api_key=_KEY, api_version=API_VERSION)
    return _clients[url]


def chat(messages: list[dict], model: str = None, temperature: float = 0.0,
         max_tokens: int = 512) -> str:
    """One completion; Gemini is independent self-training, Azure is official."""
    if PROVIDER == "gemini":
        from bot import gemini
        chosen = model or CHAT_DEPLOYMENT
        if chosen not in {CHAT_DEPLOYMENT, VISION_DEPLOYMENT}:
            raise ValueError(f"chat model allowed={CHAT_DEPLOYMENT}/{VISION_DEPLOYMENT}, actual={chosen}; use configured free-tier models")
        return gemini.chat(messages, chosen, temperature, max_tokens, _timeout())
    r = _request_client(CHAT_URL).chat.completions.create(
        model=model or CHAT_DEPLOYMENT, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return r.choices[0].message.content or ""


def ask(prompt: str, system: str = None, **kw) -> str:
    """Shorthand for a single-turn call."""
    msgs = ([{"role": "system", "content": system}] if system else [])
    return chat(msgs + [{"role": "user", "content": prompt}], **kw)


def embed(texts: list[str], batch_size: int = 256, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a LIST of strings. One call per string is the slow mistake:
    thousands of round trips at ~200ms each is minutes spent waiting on
    the network. Always batch."""
    if batch_size <= 0:
        raise ValueError(f"embedding batch min=1, actual={batch_size}; choose a positive batch size")
    if PROVIDER == "gemini":
        from bot import gemini
        return gemini.embed(texts, EMBED_DEPLOYMENT, batch_size, _timeout(), task_type)
    texts = [t.replace("\n", " ") for t in texts]
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        r = _request_client(EMBED_URL).embeddings.create(
            model=EMBED_DEPLOYMENT, input=texts[i:i + batch_size]
        )
        ordered = sorted(r.data, key=lambda item: item.index)
        expected = len(texts[i:i + batch_size])
        if len(ordered) != expected or [d.index for d in ordered] != list(range(expected)):
            raise RuntimeError(f"embedding response expected={expected} ordered vectors, "
                               f"actual={len(ordered)}; check gateway response indices")
        out.extend(d.embedding for d in ordered)
    return out


def describe_image(image_path: str, prompt: str) -> str:
    """Send one image plus an instruction to the vision deployment.

    No default prompt: writing it is the Workshop 2 block 3 exercise,
    and it belongs in build/images.py, not here.
    """
    if PROVIDER == "gemini":
        from bot import gemini
        return gemini.describe_image(image_path, prompt, VISION_DEPLOYMENT, _timeout())
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    r = _client(VISION_URL).chat.completions.create(
        model=VISION_DEPLOYMENT, max_tokens=800,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
    )
    return r.choices[0].message.content or ""
