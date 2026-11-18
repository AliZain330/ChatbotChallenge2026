# My chatbot practice report

I used this fork to practise the InnoWing challenge. I chose Gemini because I do not have the organisers' Azure key. This practice version does not meet the original gateway rule. I kept the required answer functions and `main.py` unchanged.

## Results

| Check | Result |
|---|---|
| Development questions, 18 November | 7/9 correct on review; 4/9 exact matches |
| Tests | 72 passed |
| Answer times | 1.58 to 6.25 seconds, below the 30-second limit |
| Gemini vector index | 1,352 records: 1,346 text chunks and 6 image descriptions |
| Keyword search | 20,346 chunks from 1,973 saved pages |

I accepted equivalent wording when reviewing answers. The source poster spells one winner **Digitoe**, while the answer key says **Digitio**. These are my reviewed scores, not official judging.

I tested during the daily quota block. One run scored 6.5/9 with incomplete retrieval logs. A request-audited run scored 5.5/9 using keyword fallback for all seven retrieval questions. After quota became available, I repeated the test with embeddings and got 7/9 with no fallback warnings. I added per-question warnings to the evaluator so future comparisons show this difference.

## What I tried and why

1. I started with five important pages. I scored 3/9 with the first version.
2. I added keyword search across the saved text. Gemini allowed 100 embedding inputs per minute, so I could search the rest of the text while building the vector index.
3. I described the workshop photos and the winner poster. I included image descriptions alongside their source pages so the model could distinguish winners from finalists. I changed the counting prompt to separate cushions from furniture supports; the three extra workshop image checks then passed.
4. I expanded the vector index from 252 to 1,352 records. The new worker saves each batch and resumes without deleting earlier work. Google stopped it at the reported 1,000-input daily limit. I scored 7/9 after this expansion.

## Still missing

The bot gives a current funding deadline for a historical question. It guesses the printer total because the pages list model names without clear unit counts. I need better evidence for both. I used the provided building photos and have not collected new physical data.

The embedding target is 20,454 records. The bulk worker is stopped; `python -u -m build.expand` resumes it. I keep keys, generated data and `LEARNING_GUIDE.md` local. Detailed test outputs are in `submission_repo/reports/`.

## Meeting questions

- **Why RAG?** I give Gemini the saved InnoWing facts needed for the question.
- **Why two searches?** Keywords match names; embeddings help match different wording with the same meaning.
- **How do you count posters?** I filter gallery records by year and count unique image URLs.
- **Why cache results?** I avoid repeating completed API work after a quota error or restart.
- **What needs work next?** Historical evidence, physical inventory and a test of the remaining vector coverage.
