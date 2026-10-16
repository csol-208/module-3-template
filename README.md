# Module 3: Text as Data

**Instructor guide**, CSOL 208

This module covers three sessions (8–10). Students use the annual environmental reports of five tech companies to ask whether the companies are backing away from the climate pledges they made in 2019–2021. They turn 37 reports into a table of paragraphs, code every paragraph with a decision model, extract a ledger of commitments with a language model, and compare the result with independent assessments.

## Pedagogical approach

Students in this course are not programmers. They direct an AI coding agent and judge its results. The student materials are guided workbooks (`session8.md`, `session9.md`, `session10.md`, plus `overview.md`). Each task says what to ask the agent, what a correct result looks like, and what to think about.

The emphasis of the module is the design of text codes. Students move from word counts, to yes/no questions, to categories, to ordered scales, and compare what each can show. They label paragraphs themselves and decide who is right when they and the model disagree.

The `notebooks/` directory holds instructor answer keys: plain Python files with `# %%` cell markers, which VS Code runs as notebook cells. They are not for distribution to students.

| Session | Title | Main skills |
|---|---|---|
| 8 | From Reports to Paragraphs | PDF text extraction, a paragraph dataset, word counts, first LLM and decision-model calls |
| 9 | Coding Text with a Decision Model | Codebook design (yes/no, choice, score), aggregation, comparing a model with your own labels |
| 10 | A Ledger of Commitments | Filter-then-extract, structured extraction with Pydantic, quote checking, reconciling with independent sources |

## Repository

```
module-3-template/
├── overview.md            # student introduction to the module
├── session8.md            # student workbooks
├── session9.md
├── session10.md
├── reports.csv            # the corpus: company, report, data year, URL
├── download_data.sh       # downloads reports and validation data into data/
├── pyproject.toml, uv.lock
├── DISCLOSURE.md
└── notebooks/             # instructor answer keys
    ├── session8-reports-to-paragraphs.py
    ├── session9-coding-with-decision-models.py
    ├── session10-commitment-ledger.py
    └── key-labels.csv     # labelled sample used by the Session 9 key
```

## Setup

```bash
bash download_data.sh        # ~500 MB; or e.g. `bash download_data.sh google microsoft`
uv sync
export OPENROUTER_API_KEY=...
uv run python notebooks/session8-reports-to-paragraphs.py   # then session9, session10
```

Each key saves its outputs in `output/` and reuses them on later runs, so API calls are made once.

### Corpus

| Company | Reports (title year) | Data years | Notes |
|---|---|---|---|
| Google | 2018–2026 (9) | 2017–2025 | 2021 and 2022 are 16–17 page summaries |
| Microsoft | 2020–2022, 2024–2026 (6) | 2020–2025 | No report titled 2023; fiscal years |
| Amazon | 2018–2025 (8) | 2018–2025 | Title year is the data year |
| Meta | 2019–2021, 2023–2025 (6) | 2019–2024 | 2019 has garbled pages; PDFs store one block per line |
| Apple | 2019–2026 (8) | 2018–2025 | Fiscal years |

`reports.csv` records the year each report's data covers. Use `data_year` for anything plotted over time.

Validation data: the Science Based Targets initiative company export (a snapshot of current status only) and NewClimate Institute's *Corporate Climate Responsibility Monitor 2025*, tech sector chapter (from the Internet Archive, because newclimate.org blocks scripted downloads).

### Models and API access

Students use their own OpenRouter key for everything.

- **Jev** (`~typesafe/jev-latest`, TypeSafe), a decision model, at `https://openrouter.ai/api/v1/systemone`. It returns probabilities for typed questions (`noul`, `choice`, `score`) and never generates text. The cost is about $0.04 per million input tokens, and output is free.
- **DeepSeek V4 Flash** (`deepseek/deepseek-v4-flash`) for structured extraction, through the OpenAI-compatible endpoint `https://openrouter.ai/api/v1`. Pass `extra_body={"reasoning": {"enabled": False}}` (about three times faster), and always set `max_tokens` and a client timeout (see Known issues).
- **Laya** (optional), the open-weights decision model on the course NRP server. It accepts the same request body as Jev, and has a batch endpoint. Session 9 Part 8 compares it with Jev if `LAYA_URL` and `LAYA_API_KEY` are set.

## Numbers from the key runs (October 2026)

| Step | Size | Time | Cost |
|---|---|---|---|
| Session 8: paragraphs from 37 PDFs | 11,858 paragraphs | 30 s | none |
| Session 9: "states a goal" over every paragraph (Jev) | 11,858 requests | 5.5 min | $0.20 |
| Session 9: 13-question codebook on candidates (Jev) | 2,337 requests | 2 min | $0.11 |
| Session 10: goal extraction (DeepSeek) | 322 paragraphs → 484 records | 3 min | about $0.01 |
| Session 10: NewClimate report in one request | 152,568 characters | 40–70 s | under $0.01 |

Peak memory for each key is under 400 MB.

## Session notes

### Session 8

- The literal search for Google's "no longer maintaining operational carbon neutrality" (2024 report, p. 41) fails because the PDF breaks the line mid-sentence. This motivates whitespace normalization.
- Meta's PDFs store each line as a separate text block. Merging blocks that share a left edge and sit less than a line apart restores paragraphs (Meta 2023 page 9: 12 blocks become 5 paragraphs).
- Keyword trends per 100 paragraphs: Google's "carbon neutral" falls from about 10 in the 2020 data year to almost none from 2022. Microsoft's "net zero" and "carbon negative" decline. "AI" rises for all companies except Apple. Microsoft's 2020 AI mentions are about its AI for Earth grants, a good example of a mention that is not a cause.

### Session 9

- About 15% of paragraphs pass P(goal) > 0.5. Jev is not perfectly deterministic: two runs produced 2,337 and 2,330 candidate paragraphs.
- Thresholded counts per report show the main pattern. Google's "goal has become harder" paragraphs rise to 12 in the 2025 data year, and its "AI is the cause" paragraphs to 7. Microsoft rises modestly. Amazon, Apple and Meta stay low. "Dropped" is rare for everyone.
- Goal-type mix: Google's carbon neutral share of goal paragraphs falls from about 0.5 (2020) to near 0, and Microsoft's net zero share falls to 0.07 by 2025.
- The four-level specificity score and the three yes/no components agree only moderately (rank correlation 0.74). 123 paragraphs score about 2 on the scale but meet none of the three components. They are good discussion material.
- On the 23-paragraph labelled sample in `key-labels.csv`, Jev agrees with the labels on 91–100% of paragraphs. Laya (optional) agrees on 83–100%, but it codes all three goal revisions as "strategy". The agreement rates are inflated by easy "no" cases, so discuss recall on the "yes" cases.
- **We tried Laya first as the main model.** Its four-level scores collapsed to the middle (mean specificity 2.0–2.1 in every report), and it missed the subtle retreats. For example, it gave Google's narrowed net-zero boundary 0.07 on "dropped or narrowed". It also scored garbled PDF text near 1.0 on several questions. Jev did much better on all of these, which is why it is the main model.
- `key-labels.csv` was labelled by Claude while writing the key. Check the labels before using them in class.

### Session 10

- The status definitions in the schema, and the instruction to include dropped and changed goals, matter a great deal. Without them, the 416 extracted records contained no "dropped" or "revised" statuses at all. With them, Google's operational carbon neutrality (2023 data) and Amazon's Shipment Zero (2022) are recorded as dropped, and Google's net-zero boundary change as revised.
- The ledger still contains errors for students to find. Amazon's 2020 net-zero announcement is marked "dropped". Microsoft's trillion-trees partnership is marked "dropped". Google's "intentionally aspirational" reframing (2026 report) is coded as progress.
- Quote checking, 484 records: 446 exact, 20 edited, 18 not found. Several of the "not found" quotes are the model describing the paragraph ("The paragraph states…").
- NewClimate whole-document extraction: an earlier version asked for each company's integrity rating. The ratings appear only in graphic tables, and the model returned plausible ratings with quotes that did not state them, so the key now asks only for criticisms with quotes. Three runs at `temperature=0` gave 2, 17 and 1 exact quotes out of 20 (most of the rest were edited rather than invented), which is useful evidence on reproducibility.
- SBTi lists Amazon (near-term) and Meta and Microsoft (net zero) as "Commitment Removed, expired commitment", while all three still state the goals in their latest reports. Discuss what each source measures.

## Known issues

| Issue | Mitigation |
|---|---|
| Runaway generation: two DeepSeek calls produced 131,072 tokens each, and one run took 53 minutes | `max_tokens=4000` and `timeout=90` on the client (in the key and the workbook) |
| Structured output not strictly enforced through OpenRouter (a status of "ongoing" was returned) | Retry, then skip; report the failures |
| pandas 3 turns a missing integer into NaN rather than the string `<NA>` | Build keys with an explicit missing-value check; otherwise goals without a target year silently disappear from `groupby` |
| PDF URLs change | `download_data.sh` checks each file is a PDF and reports failures; every exercise works on a subset |
| newclimate.org returns 403 to scripts | The download uses an Internet Archive copy |
| Student sessions have limited memory | The paragraph table is about 12 MB; nothing in the keys exceeds 400 MB |

## Assessment

Session 10 ends with a suggested assessment. Students answer a question about a body of climate text using the module's tools, and show how they know the answer is right. They can extend this module's question or bring their own documents. Edit as needed to match the course rubric.
