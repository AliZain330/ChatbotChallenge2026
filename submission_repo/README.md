# Independent InnoWing chatbot training

This is a self-training adaptation, **not an official challenge submission**. The original competition permits only the InnoWing gateway at answer time; the opt-in Gemini provider is outside that rule. `main.py` is unchanged, and `rag_answer(question)` / `rag_answer_batch(questions)` preserve the grading contract.

## Run locally

Commands below run from `submission_repo`. The environment created for this exercise is `../.venv` (Python 3.12). Set `PYTHONUTF8=1` on Windows when printing multilingual source text.

```powershell
../.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
# Copy .env.example to .env; never put a real key in .env.example.
# In the ignored .env, set LLM_PROVIDER=gemini and GEMINI_API_KEY locally.
../.venv/Scripts/python.exe check_provider.py
../.venv/Scripts/python.exe main.py "What does FDM stand for in 3D printing?"
../.venv/Scripts/python.exe evaluate.py
../.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The supplied `requirements.txt` still includes optional workshop/Jupyter/Google Drive dependencies. `requirements.lock.txt` records the actual tested runtime environment; it deliberately does not require the unused Google Drive tooling.

## Ingest authorised sources

```powershell
../.venv/Scripts/python.exe -m build.scrape --all-pages --resume
../.venv/Scripts/python.exe -m build.index --lexical-only
../.venv/Scripts/python.exe -m build.images --inventory-only --lexical-only
../.venv/Scripts/python.exe -m build.images --workshop-only
```

The scraper targets only `innowings.engg.hku.hk` and `innoacademy.engg.hku.hk`, uses their sitemaps and internal links, and records partial coverage/failures. Resume reuses saved pages and their links; after changing extraction logic, do a fresh scrape to refresh old records.

`text_records.json` and `image_records.json` provide local keyword coverage of all extracted text and selected image records. Chroma supplies semantic retrieval for embedded records. This separation keeps new evidence searchable when free embedding quota is exhausted. The five workshop photographs are authorised organiser material; other physical-world questions require additional authorised photographs/data.

To expand Gemini coverage continuously, run `python -u -m build.expand`. It reads the existing text/image sidecars, preserves all existing Chroma records, skips indexed IDs, and publishes each successful batch immediately. New API batches are spaced one minute apart, with provider retry hints respected. Cached batches can publish without waiting. Progress is saved to `data/embedding_progress.json`; rerun the same command to resume. Daily quota/authentication errors stop with previous work intact. Query embeddings share the provider quota, so answering may use the existing keyword fallback while ingestion consumes it.

For an intentional full semantic rebuild, run `python -m build.index --wait-for-quota`. For a selected corpus use `--pages path/to/pages.json`. The observed Gemini limit is **100 embedding inputs per minute**, so a full site index can take hours. The opt-in wait uses only a server-provided minute-quota delay; daily quotas and unknown errors stop the build. Successful vectors are cached in SQLite, and the existing Chroma collection is preserved until all requested vectors are ready. A text rebuild resets the combined collection; reindex processed images afterwards.

For website image understanding, use `python -m build.images --manifest path/to/images.json`. Select relevant authorised images first: describing every site image can take substantial time and quota. Descriptions are cached by source bytes, prompt and model. `--inventory-only` reads explicit gallery placement, not image contents; `--lexical-only` avoids embeddings. `--alt-only` uses existing captions without vision. Image IDs and stable upserts prevent duplicates across reruns.

## Providers and restrictions

Azure remains the default. Restore it with `LLM_PROVIDER=azure` and `AZURE_OPENAI_KEY` in `.env`, then rebuild the Azure index. Azure and Gemini default to separate Chroma paths because their embedding spaces are incompatible. The original `check_setup.py` remains the Azure organiser check; it correctly reports NOT READY without an Azure key. `check_provider.py` tests chat, embeddings and a real local image for the selected provider.

Gemini uses `gemini-3.5-flash-lite` for chat and image reading, and `gemini-embedding-001` for document/query embeddings. These appeared in the authenticated model list and passed actual calls. Google lists free-tier access for them in its [pricing documentation](https://ai.google.dev/gemini-api/docs/pricing). Quota depends on project/region. The adapter never enables billing, selects paid fallback models, or retries at answer time. A key alone cannot establish project billing status; this exercise uses the user's stated free-tier project.

Answering reads only local evidence and calls the configured model provider. No web search, URL-context, scraping or vision runs during answers. Chroma telemetry is disabled. The shared network deadline follows the original 30-second question limit; HTTP timeout semantics are not a strict process kill, so verify actual timings with `evaluate.py`.

## Artifacts and results

See the root `IMPLEMENTATION_REPORT.md` for executed checks and remaining failures. The nine-question `submission_repo/dev_set.json` is the real development run used here; the separate workshop sample corpus describes a fictional Harbour Innovation Centre and is not used as InnoWing evidence. Answers from the development key are never ingested.

Credentials, generated indexes, cached descriptions, scraped data and raw evaluation files remain local and ignored. A fresh clone must run ingestion before grounded retrieval works. The supplied `submission.zip` is still the original starter archive; it is not a packaged version of this adaptation. `LEARNING_GUIDE.md` is a private local companion and is deliberately excluded from commits and pushes.
