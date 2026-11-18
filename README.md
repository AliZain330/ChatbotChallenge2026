# InnoWing chatbot practice

I built this chatbot to practise the HKU InnoWing AI Chatbot Challenge. It answers questions using saved website text and image descriptions.

I used Gemini because I do not have the organisers' Azure key. The original challenge requires their gateway, so this version is for self-training. I kept `main.py` and the two required answer functions unchanged.

## Results

- 7 of 9 development answers correct on review; 4 exact matches.
- 72 tests pass.
- 1,973 saved pages and 20,346 text chunks available for keyword search.
- 1,352 records in the Gemini vector index, up from 252.

The larger index has not raised the score on these nine questions. I need the old funding deadline and a reliable printer inventory.

## Try it

With the local environment and `.env` ready, run from the repository folder:

```powershell
cd submission_repo
..\.venv\Scripts\python.exe main.py "Which two materials are banned from the laser cutter?"
```

See the [setup and test commands](submission_repo/README.md) or the [short practice report](IMPLEMENTATION_REPORT.md).

I kept credentials, generated indexes and the learning guide out of Git. A fresh clone needs credentials and local data before it can retrieve evidence.

The original materials remain in [Slides](Slides/) and [Labs](Labs/Notebook_Guide.md). The lab sample text describes a fictional centre, so I did not use it as InnoWing evidence.
