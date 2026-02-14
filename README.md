# Module 3: Working with LLMs & Unstructured Data

**Instructor Guide** — CSOL-208, Spring 2026

---

## Overview

This module spans **3 sessions (8–10)** across Weeks 4–5. Students move from chat-based AI use (Modules 1–2) to **programmatic LLM use**: calling APIs from Python code, extracting structured data from PDFs, and building validated, reusable extraction tools.

The unifying theme is **climate documents** — the messy PDFs that climate professionals actually work with: city climate action plans, corporate sustainability reports, and federal energy planning studies.

### What students will learn

| Session | Title | Core Skill |
|---------|-------|------------|
| 8 | Your First LLM Pipeline | OpenAI-compatible API calls, Pydantic structured outputs |
| 9 | Sustainability Report Parser | PDF parsing, chunking, LLM extraction at scale |
| 10 | Validation, Reproducibility & Tools | Multi-model comparison, confidence scoring, reusable pipelines |

---

## Repo Structure

```
module-3-template/
├── download_data.sh              # Downloads all 15 PDFs (~260MB)
├── pyproject.toml                # Python dependencies (managed by uv)
├── .gitignore                    # Excludes data/, output/, .venv/
├── notebooks/
│   ├── session8-llm-apis.py                # Session 8: LLM API intro
│   ├── session9-pdf-extraction.py          # Session 9: PDF → structured data
│   └── session10-validation-and-tools.py   # Session 10: Validation & tools
└── data/                         # Created by download_data.sh (not in git)
    ├── climate-action-plans/     # 7 city/state/international PDFs
    ├── corporate-sustainability/ # 4 tech/energy company reports
    └── utility-irps/             # 4 federal energy planning studies
```

### Files are `.py` with `# %%` cell markers

The notebooks use VS Code's "interactive Python" format — plain `.py` files with `# %%` cell delimiters. VS Code renders these as executable notebook cells. This is intentional:

- Cleaner git diffs than `.ipynb`
- AI coding agents (Copilot, Continue) work better with plain Python
- Students can run cells interactively or the whole file as a script
- No JSON metadata noise

To run: open a `.py` file in VS Code → click "Run Cell" above any `# %%` line, or use `Shift+Enter`.

---

## Setup

### 1. Download the PDFs

```bash
bash download_data.sh
```

This downloads 15 PDFs into `data/`. The script validates each file is actually a PDF (not an HTML error page) and reports page counts. Takes ~2 minutes on a fast connection.

**What's in the corpus:**

| Category | Files | Description |
|----------|-------|-------------|
| **Climate Action Plans** | 7 PDFs, 50–299 pages | Oakland ECAP, Portland CAP, Austin Climate Equity Plan, Seattle CAP, Ann Arbor A2Zero, CA Scoping Plan 2022, IPCC AR6 WG3 SPM |
| **Corporate Sustainability** | 4 PDFs, 68–113 pages | Apple Environmental Progress 2024, Google Environmental Report 2024, Amazon Sustainability 2023, BP Sustainability 2023 |
| **Energy Planning** | 4 PDFs, 20–310 pages | DOE Solar Futures Study, EIA Annual Energy Outlook 2023, NREL Electrification Futures, NREL Renewable Electricity Futures |

**If a URL breaks**: City and corporate websites frequently reorganize. The script will report failures. The exercises still work with any subset of PDFs — just update the filenames in the notebooks.

### 2. Python Environment (uv)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync
```

To run any script:

```bash
uv run python notebooks/session8-llm-apis.py
```

`uv run` activates the virtual environment automatically — no need to manually activate it.

Key packages: `pymupdf` (PDF parsing), `openai` (API client), `pydantic` (schemas), `langchain-text-splitters` (chunking), `pandas` (data tables).

### 3. API Configuration

All three notebooks use environment variables for API access:

```bash
export OPENAI_BASE_URL="https://ellm.nrp-nautilus.io/v1"
export OPENAI_API_KEY="your-nrp-token"   # Get from https://nrp.ai/llmtoken
export OPENAI_MODEL="qwen3"              # or glm-4.7, gpt-oss, gemma3, kimi
```

The code uses the standard `openai.OpenAI(base_url=..., api_key=...)` pattern, which works with any OpenAI-compatible endpoint. The NRP LLM endpoint provides access to frontier-class open-weight models.

**Available models** (see [NRP docs](https://nrp.ai/documentation/userdocs/ai/llm-managed/)):

| Model | Context | Notes |
|-------|---------|-------|
| `qwen3` | 262K | Multimodal, reasoning/thinking model, tool calling |
| `glm-4.7` | 202K | Reasoning model, tool calling |
| `gpt-oss` | 131K | OpenAI's open model, tool calling |
| `kimi` | 262K | Frontier coding/agentic performance |
| `gemma3` | 131K | Multimodal, batch-friendly |

**Reasoning models note**: `qwen3` and `glm-4.7` are "thinking" models that do internal chain-of-thought. For structured extraction (JSON mode), the notebooks disable thinking via `extra_body={"chat_template_kwargs": {"thinking": False}}` to get reliable content. For open-ended analysis, thinking is left enabled.

**Before class**: set the `OPENAI_API_KEY` environment variable. Generate tokens at https://nrp.ai/llmtoken. Never hardcode API keys in notebooks.

---

## Session-by-Session Walkthrough

### Session 8: Your First LLM Pipeline

**File**: `notebooks/session8-llm-apis.py`

**Learning arc**: Chat window → Python API call → Structured output → Batch extraction

**Parts:**

1. **API Connection** — `OpenAI()` client, send a simple climate question, get a response. Students see the raw request/response pattern for the first time.

2. **Pydantic Schema** — Define `ClimateCommitment` with typed fields (`company_name: str`, `target_year: Optional[int]`, etc.). Students understand that "structured output" means the LLM returns JSON matching a schema, not free text.

3. **Extract from Text** — Feed a passage about Apple's climate pledge, get back clean JSON. Uses `response_format={"type": "json_object"}` to force JSON mode. The schema is passed in the prompt (not as a tool/function).

4. **Model Comparison** — Same passage sent to multiple models. Key teaching moment: different models extract different things. The Google passage is deliberately ambiguous (24/7 CFE vs. net zero vs. carbon neutrality) to show how models handle ambiguity differently.

5. **Batch Extraction** — Three companies (Amazon, BP, Apple) → loop → Pandas DataFrame. This previews the Session 9 pattern of "iterate over sources, extract, compare."

**Instructor notes:**
- The text passages are real quotes from sustainability reports, lightly edited for clarity
- The Google passage has a subtle "gotcha" — Google claims both "carbon neutral since 2007" AND "net zero by 2030" which are different things. Good class discussion point.
- If only one model is available, Part 4 still works — just shows one result

**Approximate time**: 45–60 minutes active coding, 15–20 minutes discussion

---

### Session 9: Structured Data Extraction from PDFs

**File**: `notebooks/session9-pdf-extraction.py`

**Learning arc**: Open a PDF → Extract text → Chunk it → LLM extraction → Aggregate → Validate

**Parts:**

1. **Loading PDFs** — `fitz.open()` (PyMuPDF), page iteration, `page.get_text()`. Students see that PDF text extraction is imperfect — headers, footers, columns get merged. Good moment to discuss why this is hard.

2. **Chunking** — `RecursiveCharacterTextSplitter` with 4000-char chunks and 200-char overlap. Students can experiment with chunk sizes. Discussion: why not just send the whole document? (context limits, cost, noise)

3. **Extraction Schemas** — Four nested Pydantic models:
   - `EnergyMetrics` (total consumption, renewables %, PUE)
   - `EmissionsMetrics` (Scope 1/2/3, total)
   - `WaterMetrics` (withdrawal, consumption, WUE)
   - `ClimateTarget` (target type, year, scope coverage)
   - `SustainabilityReport` (wraps all of the above)

   Students see that well-defined schemas are the key to reliable extraction.

4. **LLM Extraction** — A function `extract_from_chunk()` that sends one chunk to the LLM. Uses `temperature=0.0` for deterministic results. Keyword scoring pre-filters chunks to skip irrelevant pages (table of contents, acknowledgments, etc.).

5. **Full Document Pipeline** — `extract_full_report()` orchestrates: load → chunk → score → filter → extract → collect. Processes the top 8–10 data-rich chunks per document. With gpt-oss, expect ~15–30 seconds per document.

6. **Multi-Report Comparison** — Processes all 4 corporate reports, aggregates into a Pandas DataFrame. The comparison table is the "payoff" — what would take a human analyst a full day takes the pipeline minutes.

7. **Validation** — `validate_emissions()` runs sanity checks: is Scope 3 > Scope 1? Is renewable % between 0–100? Are magnitudes reasonable? **Critical teaching point**: LLMs hallucinate numbers. Always validate.

**Instructor notes:**
- The `max_chunks` parameter controls cost/time. Use 8–10 for in-class demo, students can crank it up for the assessment
- BP's report is the most challenging — dense tables, regulatory language
- Apple's report is the cleanest — good for debugging
- If API rate limits are an issue, reduce `max_chunks` or add `time.sleep()` between calls
- The Google report shows growing data center energy use alongside "100% renewable matching" — great discussion of what "renewable" means in corporate reporting

**Approximate time**: 60–75 minutes active coding, 15 minutes discussion

---

### Session 10: Validation, Reproducibility & Building Tools

**File**: `notebooks/session10-validation-and-tools.py`

**Learning arc**: Can you trust AI-extracted data? → Multi-model comparison → Reproducibility → Confidence scoring → Reusable tool → Multi-document extraction

**Parts:**

1. **Setup** — Loads a city climate action plan (Oakland ECAP), defines a `CityClimatePlan` Pydantic schema with 13 fields (targets, timelines, strategies, equity). Different document type than Session 9's corporate reports.

2. **Multi-Model Comparison** — Sends the same text chunk to `qwen3` and `glm-4.7`. Compares field-by-field. Key teaching moment: the models disagree on fields like `ghg_reduction_target` (qwen3: "36% by 2020", glm-4.7: "83%") — both are real numbers from the document, but at different time horizons. Disagreements reveal ambiguity.

3. **Reproducibility** — Same model, same input, 3 runs. Even with `temperature=0.0`, some fields vary. Students learn that LLM extraction is fundamentally non-deterministic and needs to be treated as such.

4. **Confidence Scoring** — A `score_confidence()` function that combines three signals:
   - Multi-model agreement (0–2 points)
   - Run-to-run stability (0–2 points)
   - Source text grounding — is the value found verbatim in the PDF? (0–2 points)
   
   Fields are classified HIGH/MEDIUM/LOW. This is the key conceptual contribution of the session.

5. **Full-Context vs. Chunked Extraction** — With 262K token context windows, you can send entire PDFs in one shot. Students compare single-call extraction against the chunked approach from Session 9. Discussion: when does each approach win?

6. **Reusable `extract_document()` Tool** — Refactors all the extraction logic into a schema-agnostic function: give it any PDF path + any Pydantic BaseModel class → structured data + metadata + validation notes. This is the "build a capability, not just an analysis" moment.

7. **Multi-Document Extraction** — Applies `extract_document()` across all 7 city climate action plans. Builds a comparison DataFrame of targets, timelines, and strategies.

8. **Validation Report** — Per-document audit: coverage, missing fields, validation warnings. Students see the overall extraction quality across the corpus.

**Instructor notes:**
- The multi-model comparison produces genuinely interesting disagreements — Oakland has targets at 2020, 2030, and 2050, and the models pick different ones. This is not a bug; it reveals real ambiguity in policy documents.
- Run-to-run variability is real but usually small (~75% of fields are stable). Good discussion: is 75% acceptable for research? For policy analysis?
- The full-context extraction of the IPCC SPM (50 pages, ~56K tokens) produces remarkably good results in a single API call. This is the strongest argument against RAG for many use cases.
- The `extract_document()` function is genuinely reusable — students can use it in their assessments with custom schemas
- Processing all 7 PDFs takes several minutes; consider running 2–3 in class and letting students extend

**Approximate time**: 60–75 minutes active coding, 15–20 minutes discussion

---

## API / Performance Notes

All exercises use [NRP-managed LLMs](https://nrp.ai/documentation/userdocs/ai/llm-managed/) — frontier-class open-weight models hosted on shared infrastructure.

- **Structured output reliability**: JSON mode works well. Reasoning models (`qwen3`, `glm-4.7`) can return `content: None` if thinking consumes all tokens — the notebooks disable thinking for JSON extraction to prevent this.
- **Extraction accuracy**: Expect ~80–90% accuracy on numeric extraction with `qwen3`/`glm-4.7`. Session 10's validation exercises teach students to verify results.
- **Speed**: Response times vary with load. Expect 5–30 seconds per request. The notebooks include `time.sleep()` for rate limiting.
- **`max_tokens`**: Per [NRP docs](https://nrp.ai/documentation/userdocs/ai/llm-managed/), do NOT specify `max_tokens` unless required. If you must, keep it under half the context length. The notebooks omit `max_tokens` for best results.

---

## Assessment Alignment

The Module 3 assessment asks students to:

> Submit a working document extraction pipeline that processes real corporate sustainability PDFs and outputs structured data (JSON/CSV). The system should use LLM APIs programmatically to extract specific sustainability metrics and demonstrate understanding of structured outputs and modern document intelligence workflows.

The three sessions build directly toward this:

| Assessment Criterion | Session Coverage |
|---------------------|-----------------|
| LLM API usage | Session 8 (core pattern), used throughout |
| Structured outputs / Pydantic | Sessions 8–9 (schemas, JSON mode) |
| PDF processing | Session 9 (PyMuPDF, chunking) |
| Multi-document comparison | Sessions 9–10 (aggregation, cross-document tool) |
| Validation & reproducibility | Session 10 (multi-model, confidence scoring) |
| JSON/CSV output | Sessions 9–10 (both save to `output/`) |

Students can extend Session 9's pipeline or Session 10's reusable `extract_document()` tool for their assessment — add more companies, richer schemas, better validation, or apply to new document types.

---

## Potential Issues & Mitigations

| Issue | Mitigation |
|-------|-----------|
| PDF URLs break | `download_data.sh` validates each file; exercises work with any subset |
| API keys not set | Notebooks show clear error on first cell; env vars documented above |
| Rate limiting | `time.sleep(0.5)` between extraction calls; `max_chunks` controls volume |
| Malformed JSON from LLM | `try/except` around `json.loads()`; students learn this is expected |
| NRP endpoint slow/down | API response times vary; if calls time out, reduce scope or try later |
| Students unfamiliar with `# %%` format | First cell of Session 8 explains the pattern; just click "Run Cell" |

---

## Extending This Module

Ideas for student projects or additional exercises:

- **Add more companies**: Download Microsoft, Meta, Shell reports and run the same pipeline
- **Temporal analysis**: Compare a company's 2020 vs 2023 reports — are targets getting more or less ambitious?
- **Cross-validate**: Compare LLM-extracted numbers against manually verified databases (CDP, Net Zero Tracker)
- **MCP integration**: Wrap Session 10's `extract_document()` as an MCP tool that a chat agent can call
- **RAG exploration**: Add ChromaDB + sentence-transformers for vector search over the full corpus — useful if working with many more documents
- **Utility IRPs**: The `utility-irps` directory has federal energy reports; actual utility IRPs from PacifiCorp, Duke, etc. would be excellent hard-mode additions if stable URLs can be found
