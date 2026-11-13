"""Small Gemini REST adapter for self-training, not official gateway submission.

Protocol references: https://ai.google.dev/api/generate-content,
https://ai.google.dev/api/embeddings, https://ai.google.dev/api/models.
No search tools, retries, configurable hostnames, or API keys in URLs.
"""
import base64
import math
import mimetypes
import os
from pathlib import Path
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()
BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
# Synchronous batchEmbedContents has a 100-request server limit. Split an
# existing caller's larger Azure batch; never silently truncate its inputs.
MAX_EMBED_BATCH = 100


def _model_name(model: str) -> str:
    name = model.removeprefix('models/')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', name):
        raise ValueError('Gemini model required=plain model identifier, actual=invalid; use list_models()')
    return 'models/' + name


def _request(method: str, resource: str, timeout: float, payload: dict = None,
             params: dict = None) -> dict:
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if not key:
        raise RuntimeError('GEMINI_API_KEY required=nonempty, actual=missing; set it in ignored .env for self-training')
    if not math.isfinite(timeout) or timeout <= 0:
        raise TimeoutError(f'Gemini time budget required>0, actual={timeout}; start a new question')
    try:
        response = requests.request(method, f'{BASE_URL}/{resource}',
                                    headers={'x-goog-api-key': key}, json=payload,
                                    params=params, timeout=timeout, allow_redirects=False)
    except requests.RequestException:
        # Do not propagate requests exceptions: they may retain headers/secrets.
        raise RuntimeError('Gemini connection required=successful, actual=network failure; check connectivity and timeout') from None
    if not 200 <= response.status_code < 300:
        hints = {400: 'check model and payload', 401: 'check GEMINI_API_KEY',
                 403: 'check key permissions and regional availability',
                 404: 'list available models and choose a supported model',
                 429: 'free-tier quota exhausted; wait or reduce ingestion, do not enable billing',
                 503: 'service unavailable; retry manually later'}
        hint = hints.get(response.status_code, 'inspect provider status and configuration')
        # Omit remote error bodies, which can echo user input or credentials.
        raise RuntimeError(f'Gemini HTTP required=2xx, actual={response.status_code}; {hint}')
    try:
        result = response.json()
    except ValueError:
        raise RuntimeError('Gemini response required=JSON object, actual=invalid JSON; retry manually later') from None
    if not isinstance(result, dict):
        raise RuntimeError('Gemini response required=JSON object, actual=other JSON type; inspect provider compatibility')
    return result


def _text(result: dict) -> str:
    candidates = result.get('candidates', [])
    if not candidates:
        reason = result.get('promptFeedback', {}).get('blockReason', 'NO_CANDIDATES')
        reason = re.sub(r'[^A-Z_]', '', str(reason))
        raise RuntimeError(f'Gemini candidates required>=1, actual=0 ({reason}); inspect prompt or provider safety response')
    candidate = candidates[0]
    reason = candidate.get('finishReason', '')
    if reason not in {'', 'STOP', 'MAX_TOKENS'}:
        safe_reason = re.sub(r'[^A-Z_]', '', str(reason))
        raise RuntimeError(f'Gemini completion required=STOP, actual={safe_reason}; inspect safety or generation settings')
    text = ''.join(part.get('text', '') for part in candidate.get('content', {}).get('parts', [])
                   if not part.get('thought') and isinstance(part.get('text', ''), str)).strip()
    if not text:
        raise RuntimeError(f'Gemini visible answer required=nonempty, actual=empty (finish={reason}); increase output budget or use non-thinking model')
    if reason == 'MAX_TOKENS':
        raise RuntimeError('Gemini completion required=STOP, actual=MAX_TOKENS; increase output token budget before accepting a truncated answer')
    return text


def chat(messages: list[dict], model: str, temperature: float, max_tokens: int,
         timeout: float) -> str:
    """Translate the starter's text-only messages into generateContent."""
    if max_tokens <= 0:
        raise ValueError(f'Gemini output tokens min=1, actual={max_tokens}; choose a positive token budget')
    contents, system = [], []
    for message in messages:
        role, text = message['role'], message['content']
        if not isinstance(text, str):
            raise ValueError('Gemini chat content required=text string, actual=non-string; use describe_image for images')
        if role == 'system':
            system.append({'text': text})
        elif role in {'user', 'assistant'}:
            contents.append({'role': 'model' if role == 'assistant' else 'user', 'parts': [{'text': text}]})
        else:
            raise ValueError(f'Gemini role allowed=system/user/assistant, actual={role!r}; supply supported text messages')
    if not contents:
        raise ValueError('Gemini messages required>=1 conversational turn, actual=0; include a user question')
    body = {'contents': contents, 'generationConfig': {'temperature': temperature, 'maxOutputTokens': max_tokens}}
    if system:
        body['systemInstruction'] = {'parts': system}
    return _text(_request('POST', _model_name(model) + ':generateContent', timeout, body))


def embed(texts: list[str], model: str, batch_size: int, timeout: float,
          task_type: str = 'RETRIEVAL_DOCUMENT') -> list[list[float]]:
    """Preserve input order and use retrieval task type for asymmetric search."""
    if batch_size <= 0:
        raise ValueError(f'Gemini embedding batch min=1, actual={batch_size}; choose a positive batch size')
    if task_type not in {'RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY'}:
        raise ValueError(f'Gemini embedding task allowed=RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY, actual={task_type!r}; choose a retrieval task')
    if not texts:
        return []
    name = _model_name(model)
    batch_size = min(batch_size, MAX_EMBED_BATCH)
    deadline = time.monotonic() + timeout
    output, dimensions = [], None
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        if any(not isinstance(text, str) or not text.strip() for text in batch):
            raise ValueError('Gemini embedding input required=nonempty strings, actual=empty/non-string; clean document text')
        payload = {'requests': [{'model': name, 'content': {'parts': [{'text': text}]},
                                 'taskType': task_type} for text in batch]}
        result = _request('POST', name + ':batchEmbedContents', deadline - time.monotonic(), payload)
        vectors = result.get('embeddings', [])
        if len(vectors) != len(batch):
            raise RuntimeError(f'Gemini embedding count required={len(batch)}, actual={len(vectors)}; retry before indexing')
        for item in vectors:
            values = item.get('values', [])
            if not values or any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in values):
                raise RuntimeError('Gemini vector required=nonempty finite numbers, actual=invalid vector; retry before indexing')
            dimensions = len(values) if dimensions is None else dimensions
            if len(values) != dimensions:
                raise RuntimeError(f'Gemini embedding dimensions required={dimensions}, actual={len(values)}; use one embedding model and rebuild')
            output.append([float(value) for value in values])
    return output


def describe_image(image_path: str, prompt: str, model: str, timeout: float) -> str:
    """Inline local bytes only; no provider URL-context or search tools."""
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {'image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/heic', 'image/heif'}:
        raise ValueError(f'Gemini image required=supported raster MIME, actual={mime!r}; convert image first')
    body = {'contents': [{'role': 'user', 'parts': [
        {'text': prompt}, {'inlineData': {'mimeType': mime, 'data': base64.b64encode(path.read_bytes()).decode('ascii')}}]}]}
    return _text(_request('POST', _model_name(model) + ':generateContent', timeout, body))


def list_models(timeout: float) -> list[dict]:
    """List names/methods without generation, following provider pagination."""
    deadline = time.monotonic() + timeout
    result, token, seen = [], None, set()
    while True:
        payload = _request('GET', 'models', deadline - time.monotonic(),
                           params={'pageToken': token} if token else None)
        result.extend({'name': model.get('name', ''),
                       'supportedGenerationMethods': model.get('supportedGenerationMethods', [])}
                      for model in payload.get('models', []))
        token = payload.get('nextPageToken')
        if not token:
            return result
        if token in seen:
            raise RuntimeError('Gemini model pagination required=advancing token, actual=repeated token; retry model listing later')
        seen.add(token)
