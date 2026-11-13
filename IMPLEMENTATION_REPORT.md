# Independent InnoWing chatbot: implementation and evidence

This is a **self-training adaptation, not an official challenge submission**. Gemini is explicitly opt-in and violates the original Azure-only runtime rule. The original Azure implementation remains available. No billing was enabled and no paid fallback was used. The user identified their Gemini project as free-tier; an API key alone cannot verify project billing status.

`submission_repo/main.py` is unchanged. Both required functions retain their signatures and batch ordering. No development answers were loaded as chatbot evidence.

## What works

| Challenge level | Executed evidence | Status |
|---|---|---|
| 1: general LLM QA | All three submission vocabulary/faculty questions correct after terminology routing | Working through Gemini |
| 2: website RAG | Real public-site ingestion and retrieval; PVC/Teflon safety answer correct | Working; historical funding deadline unavailable |
| 3: images | Actual vision descriptions of five organiser photos and the 2025 winner poster; winner names answered correctly | Working for the processed images |
| 4: multi-source/counting | Complete gallery inventory yields 31 posters for 2025; real CLI comparison yields 31−23=8 | Working for inventory counts; physical printer total remains wrong |
| 5: physical evidence | Supplied brainstorming photograph supports the wall answer “Innovation” | Limited to authorised provided photographs; no physical visit performed |

A source page lists equipment model names without the number of installed units. Counting models, duplicate references or historical pages cannot establish the physical printer inventory. The final bot run guessed 4 against the development reference 7; this is an unresolved failure, not a verified inventory count.

## Actual development results

The submission development file contains **9 questions**, despite teaching slides referring to 15. The separate Labs corpus describes the fictional **Harbour Innovation Centre**, including different deadlines/material restrictions; it was deliberately excluded from InnoWing factual evidence.

Progression: initial 5-page RAG baseline **3/9**, broader lexical/photos **5/9**, winner-image evidence **6/9**, a focused semantic rebuild regressed to **5/9**, then terminology routing and parent-image expansion restored/improved the result to **7/9**. The completed-corpus final regression also scored **7/9** by the same semantic review. These are assistant semantic reviews, not official grades. The completed-corpus final run had **4/9 strict normalized exact matches**; equivalent wording and a reference-name discrepancy account for the remaining accepted answers.

| Level | Question | Actual final answer | Review score | Seconds |
|---|---|---|---:|---:|
| 1 | What is 3D printing also known as? | Additive manufacturing | 1 | 2.69 |
| 1 | What does FDM stand for in 3D printing? | Fused Deposition Modeling | 1 | 1.91 |
| 1 | Which faculty would you expect an engineering innovation centre to belong to? | Faculty of Engineering | 1 | 5.89 |
| 2 | When was the deadline for the second round of the Funding Scheme in 2025-26? | October 30, 2026 | 0 | 5.97 |
| 2 | Which two materials are banned from the laser cutter? | PVC and Teflon | 1 | 5.86 |
| 3 | What were the winning teams in Pitch New Tech Ideas in 2025? | SmartSocks, Digitoe, and Gingtrolley | 1 | 5.00 |
| 4 | How many 3D printers does the Centre run in total? | 4 | 0 | 5.58 |
| 4 | How many workshop posters are on the Innovation Academy websites for 2025? | 31 | 1 | 5.61 |
| 5 | On the wall at the back of the brainstorming area, there are three words. HKU, Engineering, and...? | Innovation | 1 | 5.94 |

All nine final calls completed without errors and below the original 30-second limit: **1.91–5.97 seconds**, mean **4.94 seconds** per question. A cold `main.py` single FDM question took **8.906 seconds** including Python imports. A real three-question `main.py` batch returned exactly three lines (`PVC and Teflon`; the three winning teams; `8`) with no stderr in **19.282 seconds** total.

The winner poster and event page spell the middle team **DIGITOE/Digitoe**, while the development key says **Digitio**. The poster was independently visually inspected after Gemini OCR. The semantic review accepts the source spelling; an official literal judge might not.

The deadline failure is temporal: the current funding page states **30 October 2026**, while the question asks for the 2025–26 second-round deadline (**27 February 2026** in the reference). No corroborating historical page/poster was found. A current deadline is not evidence for an older round.

Raw per-question runs, source diagnostics and timings remain local under `submission_repo/reports/`. `evaluate.py` records exact matches, actual errors, elapsed time and empty semantic-score fields for later review. A phrase-presence retrieval check is explicitly not factual correctness.

Additional workshop-image checks were run separately from the nine-question score. Initially **2/3** were semantically correct: the slide title and door layout passed, but the cushion answer was 4 instead of 3. The actual photo was inspected and the vision prompt refined to enumerate continuous cushion pads by position/colour and separate furniture groups. Regenerating the description and rerunning all three gave **3/3** on semantic review (**2/3** literal matches), in **5.61–6.80 seconds** each. The revised answers were `3`, `Building RAG with Image Data`, and `glass double doors`. This addresses one measured description error; it does not prove all generated image counts are correct.

## Data and architecture

Ingestion: the two authorised sites → sitemap/internal crawl → cleaned HTML text plus image manifest → paragraph-aware 800-character chunks with 100-character overlap (retained workshop baseline) → optional document embeddings → Chroma. All extracted text also has a local lexical sidecar, independently of vector coverage.

Images: authorised local/website bytes → Gemini OCR/count/layout prompt → fingerprinted cached description → searchable text and metadata → local image sidecar plus optional Chroma vectors. The gallery's explicit Year headings provide poster inventory metadata without claiming the poster contents have been read. Current processed image scope is **six vision descriptions** and **102 structurally tagged gallery posters**; 31 are in the 2025 section and 23 in 2024. Different thumbnails or changed gallery markup can still affect identity/coverage.

Runtime: simple terminology questions use general model knowledge. Other questions create deterministic comparison views, batch query embeddings, merge local TF-IDF and Chroma rankings by reciprocal rank, expand retrieved parent pages to their processed images, and give evidence to the answer model. Counting queries inspect all matching inventory records rather than the nearest five; year differences are computed in Python. Output is one line per question. Source text is treated as untrusted data in the prompt.

**Semantic index:** 246 text chunks from 12 priority public pages plus 6 image descriptions = **252 Chroma records**. Full lexical coverage is intentionally broader than dense coverage because Gemini reported a limit of **100 embedding inputs per minute**. Dataset/crawl counts are recorded in the final verification section below.

Gemini `gemini-embedding-001` returned **3,072-dimensional vectors**. Documents use `RETRIEVAL_DOCUMENT`; questions use `RETRIEVAL_QUERY`. Azure and Gemini have separate default Chroma directories. Document vectors are cached transactionally in SQLite by provider/model/task/text hash. Cache successes survive a later 429. An optional ingestion-only wait uses the provider's measured retry delay; unknown/daily quotas fail rather than enabling billing. Answer-time calls never automatically retry.

## Verification and observed failures

- **65 automated tests passed** in the primary environment (latest primary suite: 3.835 seconds). Tests include a real local Chroma persistence/filter/reset subprocess; API unit tests are mocked and are not presented as live model validation.
- Python compilation and `pip check` passed. A separate clean environment installed all **89 locked dependencies**, then passed all **65 tests in 4.889 seconds** and `pip check`. `requirements.lock.txt` pins the actual tested Python 3.12 runtime packages.
- Original `check_setup.py`: required imports PASS; Azure credentials FAIL; NOT READY. No Azure gateway call could be authenticated without the organiser key.
- Authenticated Gemini model listing returned HTTP 200. `check_provider.py` actual calls: chat **2.469 s**, embeddings **2.828 s** (two 3,072-dimensional vectors), vision **6.782 s**. The wall OCR returned `ENGINEERING INNOVATION`.
- Gemini bulk embedding returned real HTTP 429, reporting `EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier`, quota 100, and a 28-second retry hint. Cache preservation and opt-in waits were executed. Later waits of 29 and 52 seconds came from provider responses.
- Windows briefly locked a crawl report during atomic replacement. Checkpoint publication now defers a locked intermediate snapshot to the next page; final publication remains mandatory. A regression test covers preservation/recovery.
- Public-site dead links return HTTP 404 and remain in crawl reports. A transient `/innowing2/techtalk/` timeout was retried successfully; data was preserved throughout.
- Retrieval regressions were measured, not hidden: finalist text displaced the winner image, and machine-specific examples confused a broad synonym. Parent-image expansion and terminology routing addressed those failures.

Chroma telemetry is disabled. At answer time the implementation performs local reads plus only the configured generation/embedding API calls; it does not search the web, scrape, invoke vision, enable URL context or use search tools. Gemini mode remains outside the original competition rules. HTTP timeouts share the original 30-second budget but are not a hard process-level execution kill; actual timings must still be measured.

## Files and private artifacts

Modified: `bot/answer.py`, `bot/llm.py`, `bot/store.py`, `build/scrape.py`, `build/index.py`, `build/images.py`, `.env.example`, `.gitignore`, and root README. Added: `bot/gemini.py`, `build/embeddings.py`, `evaluate.py`, `check_provider.py`, seven test files, runtime lock, submission runbook and this report.

Private/local only: `.env`, scraped data, raw evaluations, descriptions, embeddings, Chroma indexes and **LEARNING_GUIDE.md**. The guide is excluded through `.git/info/exclude` and is not committed or pushed. The real key was moved from the tracked example into ignored `.env` before any implementation commit; staged files and new commit content were checked against the actual key value.

The original `submission.zip` remains a starter archive. A fresh clone requires ingestion; this fork does not publish confidential or unreviewed generated indexes. An official submission would additionally need the authorised Azure key and a reviewed index package generated with matching Azure embeddings.

## Improvements worth discussing

Obtain an authoritative dated equipment inventory and historical funding-round evidence; collect authorised building photos for unobserved zones; process more relevant website posters; validate OCR/counts against original images; preserve table rows/section structure more precisely; broaden dense coverage within available free quota; benchmark retrieval changes independently from prompt changes. General multi-hop reasoning is heuristic, not a fully validated planner.

## Five supervisor questions and concise answers

1. **Why use RAG instead of just asking Gemini?** Gemini does not reliably know InnoWing-specific or recent facts. Retrieved website passages and image transcriptions supply inspectable evidence; general vocabulary questions can use prior knowledge.
2. **Why combine keyword and vector retrieval?** Keywords retain exact names and cover the entire collected corpus without embedding quota; vectors bridge paraphrases. Reciprocal-rank fusion combines their rankings, and source-page image expansion fixes facts hidden in posters.
3. **How did you count 31 posters without asking the model to count five search results?** The Academy gallery explicitly groups images by Year. We retain same-section metadata, filter every 2025 workshop-poster record, deduplicate image sources and count in Python. The two-year comparison computes 31−23=8.
4. **Why do embeddings and image descriptions need caches?** Vision and embeddings depend on source data rather than the current question. Fingerprinted descriptions and transactional vector checkpoints avoid repeated work, survive quota failures, and keep expensive ingestion outside the answer budget.
5. **What prevents this from being an official submission, and what remains unproven?** Gemini is not allowed by the original runtime rules; Azure must be selected with the organiser key and its index rebuilt. The historical deadline and physical printer count remain unresolved, broader physical coverage is absent, and the measured 7/9 review is not an official judge score.

## Final verification snapshot

Verified at 2026-11-13T10:15:06+08:00.

- Crawl process exited 0; both internal link graphs exhausted. **1,973 usable pages**: 985 InnoWing and 988 Academy. One additional empty HTML page was visited but not retained.
- Final local text index: **20,346 chunks**, **8,671,705 extracted characters**. There are **15,217 image references / 6,326 unique image URLs**, mostly not vision-described. Processed scope remains six visual descriptions and 102 explicit gallery-inventory records.
- **22 linked URLs returned HTTP404**, 11 per site. No outstanding timeouts. Full failure URLs and per-site counts are local in `data/crawl_report.json` and `data/corpus_stats.json`.
- Final clean-environment rerun: **65 tests passed in 2.808 seconds**, `pip check` clean, Python compilation successful. This was Windows/Python 3.12; Linux execution has not been tested.
- Final actual QA: **7/9 semantic review**, **4/9 normalized exact match**, **5.97 seconds maximum**; two source-data gaps remain as documented above.
- The compound real CLI question asking what FDM stands for **and** which laser materials are banned returned `Fused Deposition Modeling; PVC and Teflon`. Compound questions now retrieve each clause instead of being incorrectly routed entirely to general knowledge.
- A Python `requests` audit of an actual safety answer observed exactly two requests, both to `generativelanguage.googleapis.com`; answer was `PVC and Teflon` in 6.313 seconds. This is application instrumentation, not an operating-system network capture.
- `main.py` is unchanged from the original commit. Private guide, credentials and all generated data remain excluded. Source commits preserve real timestamps.

The tested source and this report are prepared for a normal fast-forward push to the user's fork. The private learning guide is not part of that push.

