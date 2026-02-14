# %% [markdown]
# # Session 10: Validation, Reproducibility & Building Tools
#
# ## Can You Trust What the AI Extracted?
#
# In Session 9, you built a pipeline that extracts structured data from PDFs.
# It produced numbers, filled in tables, processed hundreds of pages in minutes.
#
# But here's the uncomfortable question: **are those numbers right?**
#
# LLMs confidently produce numbers that aren't in the source document. They
# mix up units. They grab a number from a table heading instead of a data cell.
# They hallucinate plausible-sounding values for fields that don't exist in the text.
#
# This session is about learning to **trust but verify** — and then packaging
# your validated pipeline into a reusable tool.
#
# ### What you'll build:
# 1. Multi-model extraction: same document, different models — do they agree?
# 2. Reproducibility checks: same model, same document, twice — same answer?
# 3. Confidence scoring: which extracted fields should you trust?
# 4. A reusable extraction tool that works on any PDF + any schema
# 5. Test your tool on documents it hasn't seen before

# %% [markdown]
# ## Part 1: Setup — Load a Document and Define a Schema
#
# We'll work with city climate action plans — a different document type than
# the corporate sustainability reports in Session 9. This also tests whether
# our approach generalizes.

# %%
import fitz  # PyMuPDF
import os
import json
import time
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Project paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent if '__file__' in dir() else Path.cwd().parent
CAP_DIR = str(_PROJECT_ROOT / "data" / "climate-action-plans")
CORP_DIR = str(_PROJECT_ROOT / "data" / "corporate-sustainability")

# API client — credentials from environment variables, never hardcoded
client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://ellm.nrp-nautilus.io/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
)

# %%
# A schema for city climate action plans — different from Session 9's
# corporate sustainability schema
class CityClimatePlan(BaseModel):
    """Structured data extracted from a city climate action plan."""
    city_name: str = Field(description="Name of the city or jurisdiction")
    plan_title: str = Field(description="Title of the climate action plan document")
    plan_year: Optional[int] = Field(None, description="Year the plan was published")
    ghg_reduction_target: Optional[str] = Field(None, description="Primary GHG reduction target (e.g., '80% below 1990 levels by 2050')")
    target_year: Optional[int] = Field(None, description="Target year for primary GHG goal")
    baseline_year: Optional[int] = Field(None, description="Baseline year for measuring reductions")
    interim_targets: Optional[str] = Field(None, description="Intermediate milestones before the main target")
    carbon_neutrality_goal: Optional[str] = Field(None, description="Net zero or carbon neutrality commitment if any")
    renewable_energy_target: Optional[str] = Field(None, description="Renewable energy goals")
    transportation_strategy: Optional[str] = Field(None, description="Key transportation/mobility strategies")
    building_strategy: Optional[str] = Field(None, description="Building efficiency or electrification strategies")
    equity_commitment: Optional[str] = Field(None, description="Environmental justice or equity goals")
    total_current_emissions: Optional[str] = Field(None, description="Current or most recent total GHG emissions figure")

print("Schema fields:")
for name, field in CityClimatePlan.model_fields.items():
    print(f"  {name}: {field.annotation.__name__ if hasattr(field.annotation, '__name__') else field.annotation}")

# %% [markdown]
# ## Part 2: The Extraction Function
#
# Let's build the core extraction function from Session 9's patterns,
# but cleaner and more general.

# %%
def load_pdf_text(pdf_path: str) -> list[dict]:
    """Load a PDF and return page-level text."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages

def chunk_pages(pages: list[dict], chunk_size: int = 4000, overlap: int = 200) -> list[dict]:
    """Split pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for page in pages:
        for j, chunk_text in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "page_num": page["page_num"],
                "chunk_index": j,
                "text": chunk_text,
            })
    return chunks

def extract_from_chunk(chunk_text: str, schema_prompt: str, model: str,
                       disable_thinking: bool = True) -> dict:
    """Extract structured data from a text chunk using an LLM.
    
    Args:
        chunk_text: The text to extract from
        schema_prompt: System prompt describing what to extract  
        model: Which model to use
        disable_thinking: Disable reasoning/thinking mode for reliable JSON
    
    Returns:
        Parsed JSON dict, or {"no_data": True} on failure
    """
    extra_body = {}
    if disable_thinking:
        if model == "glm-4.7":
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            extra_body = {"chat_template_kwargs": {"thinking": False}}
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": schema_prompt},
                {"role": "user", "content": f"Extract data from this text:\n\n{chunk_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        if content is None:
            return {"no_data": True, "_error": "model returned no content"}
        return json.loads(content)
    except Exception as e:
        return {"no_data": True, "_error": str(e)}

print("✓ Extraction functions defined")

# %% [markdown]
# ## Part 3: Multi-Model Comparison
#
# Here's our first trust test. If we send the same text to two different 
# models, do they extract the same data?
#
# Disagreements reveal:
# - Fields that are genuinely ambiguous in the source text
# - Model-specific biases in number interpretation
# - Cases where one model hallucinates and the other doesn't

# %%
# Load a test document — Oakland's ECAP is well-structured and has clear targets
test_pdf = os.path.join(CAP_DIR, "oakland-ecap-2020.pdf")
pages = load_pdf_text(test_pdf)
chunks = chunk_pages(pages)
print(f"Loaded: {os.path.basename(test_pdf)} — {len(pages)} pages, {len(chunks)} chunks")

# Find data-rich chunks
keywords = ["emissions", "reduction", "target", "goal", "GHG", "carbon neutral",
            "renewable", "net zero", "baseline", "percent", "by 2030", "by 2050"]
scored = []
for chunk in chunks:
    score = sum(1 for kw in keywords if kw.lower() in chunk["text"].lower())
    if score >= 3:
        scored.append((score, chunk))
scored.sort(key=lambda x: x[0], reverse=True)

test_chunk = scored[0][1]
print(f"\nBest chunk (score={scored[0][0]}, page {test_chunk['page_num']}):")
print(test_chunk["text"][:400])

# %%
# Extract with two different models
SCHEMA_PROMPT = """You are a climate policy data extraction assistant.
Extract climate action plan details from the provided text as JSON.

Fields to extract:
- city_name: Name of the city or jurisdiction
- ghg_reduction_target: Primary GHG reduction target (exact quote if possible)
- target_year: Target year for the main goal
- baseline_year: Baseline year for measuring reductions
- interim_targets: Any intermediate milestones
- carbon_neutrality_goal: Net zero or carbon neutrality commitment
- renewable_energy_target: Renewable energy goals
- equity_commitment: Environmental justice or equity mentions

Use null for any field not found in the text. Do NOT guess or infer — 
only extract what is explicitly stated."""

models_to_compare = ["qwen3", "glm-4.7"]
model_results = {}

for model_name in models_to_compare:
    print(f"\n{'='*50}")
    print(f"Extracting with: {model_name}")
    print(f"{'='*50}")
    
    result = extract_from_chunk(test_chunk["text"], SCHEMA_PROMPT, model=model_name)
    model_results[model_name] = result
    print(json.dumps(result, indent=2))

# %%
# Compare the results — where do the models agree and disagree?
print("\n📊 MODEL COMPARISON")
print("=" * 70)
print(f"{'Field':<30} {'qwen3':<20} {'glm-4.7':<20}")
print("-" * 70)

all_keys = set()
for r in model_results.values():
    all_keys.update(r.keys())

for key in sorted(all_keys):
    if key.startswith("_"):
        continue
    val_a = model_results.get("qwen3", {}).get(key, "—")
    val_b = model_results.get("glm-4.7", {}).get(key, "—")
    
    # Truncate long values for display
    str_a = str(val_a)[:18] if val_a else "null"
    str_b = str(val_b)[:18] if val_b else "null"
    
    match = "✓" if str(val_a) == str(val_b) else "✗"
    print(f"{match} {key:<28} {str_a:<20} {str_b:<20}")

# %% [markdown]
# ## Part 4: Reproducibility — Same Model, Same Input, Twice
#
# Even with `temperature=0.0`, models aren't perfectly deterministic.
# (Floating-point arithmetic, batching, server-side sampling can vary.)
#
# If the same extraction gives different results on different runs,
# we can't trust single-run results for quantitative work.

# %%
# Run the same extraction 3 times with the same model
MODEL = os.environ.get("OPENAI_MODEL", "qwen3")
reproducibility_results = []

print(f"Running 3 extractions with {MODEL} on the same chunk...")
for run in range(3):
    result = extract_from_chunk(test_chunk["text"], SCHEMA_PROMPT, model=MODEL)
    reproducibility_results.append(result)
    print(f"  Run {run+1} complete")
    time.sleep(1)

# %%
# Check reproducibility across runs
print("\n🔄 REPRODUCIBILITY CHECK")
print("=" * 70)

# Count how often each field has the same value across all runs
all_keys = set()
for r in reproducibility_results:
    all_keys.update(r.keys())

stable_count = 0
total_count = 0
for key in sorted(all_keys):
    if key.startswith("_") or key == "no_data":
        continue
    values = [str(r.get(key, "MISSING")) for r in reproducibility_results]
    is_stable = len(set(values)) == 1
    status = "STABLE" if is_stable else "VARIES"
    total_count += 1
    if is_stable:
        stable_count += 1
    
    if is_stable:
        print(f"  ✓ {key}: {values[0][:60]}")
    else:
        print(f"  ✗ {key}: {' | '.join(v[:30] for v in values)}")

print(f"\nReproducibility: {stable_count}/{total_count} fields stable across 3 runs ({100*stable_count/max(total_count,1):.0f}%)")

# %% [markdown]
# ## Part 5: Confidence Scoring
#
# Not all extracted fields deserve equal trust. We can build a simple
# confidence score based on:
# 
# 1. **Multi-model agreement**: Did both models extract the same value?
# 2. **Reproducibility**: Is the value stable across multiple runs?
# 3. **Specificity**: Is it a concrete number or vague language?
# 4. **Source grounding**: Does the value appear verbatim in the text?

# %%
def score_confidence(field_name: str, value, source_text: str,
                     model_results: dict, reproducibility_results: list) -> dict:
    """Score confidence in an extracted field value."""
    if value is None:
        return {"field": field_name, "value": None, "confidence": "N/A", "score": 0, "reasons": ["null value"]}
    
    reasons = []
    score = 0
    str_val = str(value)
    
    # 1. Multi-model agreement (0-2 points)
    model_values = [str(r.get(field_name)) for r in model_results.values()]
    if len(set(model_values)) == 1 and model_values[0] != "None":
        score += 2
        reasons.append("models agree")
    elif str_val in model_values:
        score += 1
        reasons.append("partial model agreement")
    else:
        reasons.append("models disagree")
    
    # 2. Reproducibility (0-2 points) 
    repro_values = [str(r.get(field_name)) for r in reproducibility_results]
    if len(set(repro_values)) == 1:
        score += 2
        reasons.append("stable across runs")
    else:
        reasons.append("varies across runs")
    
    # 3. Source grounding (0-2 points)
    # Check if the extracted value appears in the source text
    if str_val.lower() in source_text.lower():
        score += 2
        reasons.append("found in source text")
    elif any(word in source_text.lower() for word in str_val.lower().split()[:3]):
        score += 1
        reasons.append("partial match in source")
    else:
        reasons.append("not found verbatim in source")
    
    # Classify
    if score >= 5:
        confidence = "HIGH"
    elif score >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return {
        "field": field_name,
        "value": str_val[:80],
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
    }

# %%
# Score every field from our primary extraction
primary_result = reproducibility_results[0]  # Use first run as primary

print("🎯 CONFIDENCE SCORES")
print("=" * 70)

confidence_results = []
for field_name in sorted(CityClimatePlan.model_fields.keys()):
    value = primary_result.get(field_name)
    cs = score_confidence(
        field_name, value, test_chunk["text"],
        model_results, reproducibility_results,
    )
    confidence_results.append(cs)
    
    icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴", "N/A": "⚪"}.get(cs["confidence"], "?")
    val_display = cs["value"][:50] if cs["value"] else "null"
    print(f"  {icon} {cs['confidence']:<6} {field_name:<30} {val_display}")
    if cs["confidence"] in ("LOW", "MEDIUM"):
        print(f"         Reasons: {', '.join(cs['reasons'])}")

high = sum(1 for c in confidence_results if c["confidence"] == "HIGH")
med = sum(1 for c in confidence_results if c["confidence"] == "MEDIUM")
low = sum(1 for c in confidence_results if c["confidence"] == "LOW")
print(f"\nSummary: {high} high, {med} medium, {low} low confidence fields")

# %% [markdown]
# ## Part 6: Full-Context vs. Chunked Extraction
#
# In Session 9, we broke documents into chunks because older models had 
# small context windows. But qwen3 handles 262K tokens — enough for most
# entire PDFs. Does sending the whole document at once do better?

# %%
# Full-context extraction: send all pages concatenated
# Use a smaller document for this test
small_pdf = os.path.join(CAP_DIR, "ipcc-ar6-wg3-spm.pdf")  # 56 pages
pages = load_pdf_text(small_pdf)
full_text = "\n\n".join(p["text"] for p in pages)
print(f"Document: {os.path.basename(small_pdf)}")
print(f"Pages: {len(pages)}, Total chars: {len(full_text):,}")

# Estimate tokens (~4 chars per token)
est_tokens = len(full_text) // 4
print(f"Estimated tokens: ~{est_tokens:,}")

# %%
# Extract from the full document in one shot
FULL_DOC_PROMPT = """You are a climate policy data extraction assistant.
You are given the FULL TEXT of a climate document. Extract key information as JSON.

Fields to extract:
- city_name: Name of the jurisdiction or body (e.g. "IPCC" for international reports)
- plan_title: Document title
- plan_year: Publication year
- ghg_reduction_target: Primary GHG reduction target or range discussed
- target_year: Target year(s) for climate goals
- baseline_year: Baseline year for comparisons
- interim_targets: Near-term milestones (e.g., 2030 targets)
- carbon_neutrality_goal: Net zero goal discussed
- renewable_energy_target: Renewable energy findings or targets
- transportation_strategy: Key transportation findings
- building_strategy: Building sector findings
- equity_commitment: Equity or justice dimensions
- total_current_emissions: Global or aggregate emissions figures

Extract ONLY what the document explicitly states. Use null for fields not present.
Prefer direct quotes and specific numbers over paraphrasing."""

print("Sending full document to LLM (this may take a minute)...")
full_context_result = extract_from_chunk(
    full_text, FULL_DOC_PROMPT, model=MODEL
)
print("\nFull-context extraction:")
print(json.dumps(full_context_result, indent=2))

# %%
# Now extract from just the top chunks (Session 9 approach)
chunks = chunk_pages(pages)
scored = []
for chunk in chunks:
    score = sum(1 for kw in keywords if kw.lower() in chunk["text"].lower())
    if score >= 3:
        scored.append((score, chunk))
scored.sort(key=lambda x: x[0], reverse=True)

chunked_results = []
for i, (score, chunk) in enumerate(scored[:5]):  # Top 5 chunks
    result = extract_from_chunk(chunk["text"], FULL_DOC_PROMPT, model=MODEL)
    if not result.get("no_data"):
        chunked_results.append(result)
    print(f"  Chunk {i+1}/5 (page {chunk['page_num']}): {'data found' if not result.get('no_data') else 'no data'}")
    time.sleep(0.5)

# Merge chunked results (take first non-null for each field)
merged_chunked = {}
for field in CityClimatePlan.model_fields:
    for r in chunked_results:
        val = r.get(field)
        if val is not None:
            merged_chunked[field] = val
            break

# %%
# Compare: full-context vs. chunked
print("\n📊 FULL-CONTEXT vs. CHUNKED EXTRACTION")
print("=" * 70)
print(f"{'Field':<30} {'Full-context':<20} {'Chunked (merged)':<20}")
print("-" * 70)

for field in sorted(CityClimatePlan.model_fields.keys()):
    val_full = str(full_context_result.get(field, "null"))[:18]
    val_chunk = str(merged_chunked.get(field, "null"))[:18]
    match = "✓" if val_full == val_chunk else "≠"
    print(f"{match} {field:<28} {val_full:<20} {val_chunk:<20}")

print(f"\n💡 Full-context: 1 API call. Chunked: {len(scored[:5])} API calls.")
print("   Full-context sees the whole document but may miss details in dense sections.")
print("   Chunked focuses on the best sections but may miss document-wide context.")

# %% [markdown]
# ## Part 7: Building a Reusable Extraction Tool
#
# Up until now, we've been writing one-off analysis code. Let's turn this
# into a proper **tool** — a function that takes any PDF and any Pydantic
# schema and returns validated, confidence-scored structured data.
#
# This is the difference between "I ran a notebook once" and "I built
# something useful."

# %%
import pandas as pd

def extract_document(
    pdf_path: str,
    schema: type[BaseModel],
    model: str = None,
    max_chunks: int = 10,
    validate: bool = True,
) -> dict:
    """Extract structured data from a PDF using an LLM.
    
    This is a reusable tool: give it any PDF and any Pydantic schema,
    and it returns structured, validated data.
    
    Args:
        pdf_path: Path to a PDF file
        schema: A Pydantic BaseModel class defining the fields to extract
        model: LLM model name (defaults to OPENAI_MODEL env var or 'qwen3')
        max_chunks: Maximum number of chunks to process
        validate: Whether to run validation checks
    
    Returns:
        dict with keys: 'data', 'metadata', 'validation'
    """
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "qwen3")
    
    filename = os.path.basename(pdf_path)
    
    # 1. Load and chunk
    pages = load_pdf_text(pdf_path)
    chunks = chunk_pages(pages)
    
    # 2. Build schema-aware prompt from the Pydantic model
    field_descriptions = []
    for name, field in schema.model_fields.items():
        desc = field.description or "no description"
        field_descriptions.append(f"- {name}: {desc}")
    fields_text = "\n".join(field_descriptions)
    
    prompt = f"""You are a document data extraction assistant.
Extract the following fields from the provided text as JSON.

Fields to extract:
{fields_text}

Rules:
- Use null for any field not found in the text
- Do NOT guess or infer values — only extract what is explicitly stated
- Use exact numbers and quotes from the text where possible
- If a field has multiple possible values, choose the most specific one"""
    
    # 3. Score and select best chunks
    # Use field names as keywords for relevance scoring
    score_keywords = [name.replace("_", " ") for name in schema.model_fields.keys()]
    # Add common data indicator words
    score_keywords.extend(["percent", "target", "goal", "emissions", "by 20"])
    
    scored = []
    for chunk in chunks:
        score = sum(1 for kw in score_keywords if kw.lower() in chunk["text"].lower())
        if score >= 2:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:max_chunks]
    
    # 4. Extract from each chunk
    all_results = []
    for i, (score, chunk) in enumerate(selected):
        result = extract_from_chunk(chunk["text"], prompt, model=model)
        if not result.get("no_data"):
            result["_page"] = chunk["page_num"]
            all_results.append(result)
        time.sleep(0.5)
    
    # 5. Merge results — first non-null value wins, track source pages
    merged = {}
    source_pages = {}
    for field in schema.model_fields:
        for r in all_results:
            val = r.get(field)
            if val is not None:
                merged[field] = val
                source_pages[field] = r.get("_page", "unknown")
                break
    
    # 6. Validate against schema
    validation_notes = []
    try:
        validated = schema(**merged)
        validation_notes.append("✓ Schema validation passed")
    except Exception as e:
        validation_notes.append(f"✗ Schema validation failed: {e}")
        validated = None
    
    # 7. Basic consistency checks
    if validate:
        # Check: are extracted values actually in the source text?
        full_text = " ".join(p["text"] for p in pages).lower()
        for field, value in merged.items():
            if value is not None and str(value).lower() not in full_text:
                # Don't flag short common words
                if len(str(value)) > 5:
                    validation_notes.append(
                        f"⚠️  {field}: '{str(value)[:50]}' not found verbatim in document"
                    )
    
    return {
        "data": merged,
        "validated": validated.model_dump() if validated else None,
        "metadata": {
            "source_file": filename,
            "pages": len(pages),
            "chunks_total": len(chunks),
            "chunks_with_data": len(all_results),
            "chunks_processed": len(selected),
            "model": model,
            "source_pages": source_pages,
        },
        "validation": validation_notes,
    }

print("✓ extract_document() tool defined")
print(f"  Signature: extract_document(pdf_path, schema, model='{MODEL}', max_chunks=10)")

# %% [markdown]
# ## Part 8: Test the Tool on Multiple Documents
#
# Let's run our tool across all the city climate action plans and build
# a comparison dataset.

# %%
# Process all climate action plans
cap_files = sorted([f for f in os.listdir(CAP_DIR) if f.endswith('.pdf')])
print(f"Processing {len(cap_files)} climate action plans...\n")

all_extractions = []
for pdf_file in cap_files:
    pdf_path = os.path.join(CAP_DIR, pdf_file)
    print(f"📄 {pdf_file}")
    
    result = extract_document(pdf_path, CityClimatePlan, max_chunks=8)
    all_extractions.append(result)
    
    # Show key findings
    data = result["data"]
    print(f"   City: {data.get('city_name', '?')}")
    print(f"   Target: {data.get('ghg_reduction_target', '?')}")
    print(f"   Chunks with data: {result['metadata']['chunks_with_data']}/{result['metadata']['chunks_processed']}")
    for note in result["validation"]:
        print(f"   {note}")
    print()

# %%
# Build the comparison table
rows = []
for ext in all_extractions:
    row = {"source_file": ext["metadata"]["source_file"]}
    row.update(ext["data"])
    rows.append(row)

df = pd.DataFrame(rows)

print("🏙️ CITY CLIMATE TARGETS COMPARISON")
print("=" * 80)
display_cols = ["city_name", "ghg_reduction_target", "target_year",
                "carbon_neutrality_goal", "renewable_energy_target"]
available = [c for c in display_cols if c in df.columns]
print(df[available].to_string(index=False))

# %% [markdown]
# ## Part 9: Validation Report
#
# Let's compile a validation report across all extractions.
# This is what separates rigorous data work from "I asked ChatGPT."

# %%
print("🔍 VALIDATION REPORT")
print("=" * 70)

for ext in all_extractions:
    city = ext["data"].get("city_name", ext["metadata"]["source_file"])
    print(f"\n📄 {city} ({ext['metadata']['source_file']})")
    print(f"   Model: {ext['metadata']['model']}")
    print(f"   Coverage: {ext['metadata']['chunks_with_data']}/{ext['metadata']['chunks_processed']} chunks had data")
    
    # Count non-null fields
    non_null = sum(1 for v in ext["data"].values() if v is not None)
    total = len(CityClimatePlan.model_fields)
    print(f"   Fields extracted: {non_null}/{total}")
    
    for note in ext["validation"]:
        print(f"   {note}")
    
    # Flag fields we couldn't extract
    missing = [f for f in CityClimatePlan.model_fields if ext["data"].get(f) is None]
    if missing:
        print(f"   Missing: {', '.join(missing)}")

# Count overall coverage
total_fields = len(CityClimatePlan.model_fields) * len(all_extractions)
filled_fields = sum(
    sum(1 for v in ext["data"].values() if v is not None)
    for ext in all_extractions
)
print(f"\n{'='*70}")
print(f"Overall: {filled_fields}/{total_fields} fields extracted ({100*filled_fields/total_fields:.0f}%)")

# %% [markdown]
# ## Part 10: Save Everything

# %%
output_dir = str(_PROJECT_ROOT / "output")
os.makedirs(output_dir, exist_ok=True)

# Save the full extraction results (including metadata and validation)
with open(os.path.join(output_dir, "city_climate_extractions.json"), "w") as f:
    json.dump(all_extractions, f, indent=2, default=str)

# Save the comparison table
df.to_csv(os.path.join(output_dir, "city_climate_comparison.csv"), index=False)

print(f"✓ Saved to {output_dir}/")
print(f"  - city_climate_extractions.json (full results with validation)")
print(f"  - city_climate_comparison.csv (comparison table)")

# %% [markdown]
# ## 🎯 Session 10 Takeaways
#
# 1. **Don't trust single-model extraction.** Different models extract 
#    different values from the same text. Multi-model comparison reveals 
#    which fields are ambiguous.
#
# 2. **Reproducibility matters.** Even with temperature=0, results can vary
#    across runs. Stable fields are more trustworthy than unstable ones.
#
# 3. **Confidence scoring** combines multiple signals (model agreement, 
#    reproducibility, source grounding) into actionable trust levels.
#
# 4. **Full-context vs. chunked** is a real tradeoff. Large context windows
#    let you send entire documents, but chunked extraction can focus on 
#    the most relevant sections.
#
# 5. **Reusable tools > one-off scripts.** The `extract_document()` function
#    works on any PDF with any schema. That's the difference between an 
#    analysis and a capability.
#
# 6. **Validation is the hard part.** Building the extraction pipeline is 
#    straightforward. Knowing which results to trust is the real skill.
#
# ### What you built:
# A validated, multi-document extraction system with confidence scoring —  
# the kind of pipeline that turns hundreds of pages of climate PDFs into
# a trustworthy dataset for policy analysis.
#
# ### Module 3 Assessment:
# Submit a working document extraction pipeline that processes real climate
# PDFs and outputs structured data (JSON/CSV). Your system should demonstrate:
# - PDF loading and intelligent chunking
# - Pydantic schema design for your chosen domain
# - LLM-based structured extraction
# - Validation and confidence assessment
# - Results saved as JSON and/or CSV
