# Running the chatbot

Run these commands from `submission_repo`. I tested this on Windows with Python 3.12.

## First setup

Create the environment from the repository root if you do not have one:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r submission_repo/requirements.lock.txt
cd submission_repo
```

Copy `.env.example` to `.env` if `.env` does not exist. Put your key in `.env`:

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
```

Keep your real key out of `.env.example` and Git. To use the original gateway, set `LLM_PROVIDER=azure`, add `AZURE_OPENAI_KEY`, and rebuild with Azure embeddings.

## Ask questions and run tests

```powershell
..\.venv\Scripts\python.exe main.py "What does FDM stand for in 3D printing?"
..\.venv\Scripts\python.exe evaluate.py
..\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

`evaluate.py` saves answers, timings and retrieval warnings in `reports/`. Check those warnings before comparing scores: a quota-blocked run uses keyword search instead of the vector search.

## Prepare local data

A fresh clone needs the saved data. These commands collect the websites, prepare keyword search and process the five workshop photos:

```powershell
..\.venv\Scripts\python.exe -m build.scrape --all-pages --resume
..\.venv\Scripts\python.exe -m build.index --lexical-only
..\.venv\Scripts\python.exe -m build.images --inventory-only --lexical-only
..\.venv\Scripts\python.exe -m build.images --workshop-only
```

For other authorised images, pass a selected image manifest with `--manifest`. The photos cover part of the building; they cannot establish its current condition.

## Add more Gemini embeddings

```powershell
..\.venv\Scripts\python.exe -u -m build.expand
```

This keeps the current index and adds a batch each minute. It reuses cached vectors and skips finished records. Google reported a daily limit of 1,000 embedding inputs for this project; the worker stops at that limit. Resume with the same command after quota resets. Check `data/embedding_progress.json` for progress.

The chatbot reads saved evidence while answering. It does not fetch websites or describe new images during a question. Gemini mode is for this practice fork; the official challenge requires the organisers' gateway.
