# %% [markdown]
# # Session 9: Structured Data Extraction from PDFs
#
# ## Sustainability Report Parser
#
# In Session 8, we extracted structured data from text passages. Now we'll 
# work with **real PDF documents** — the messy, complex files that climate 
# professionals actually encounter.
#
# ### The Challenge
# Corporate sustainability reports contain valuable data buried in prose,
# tables, infographics, and mixed formatting. Manual extraction is slow
# and error-prone. We'll build an automated pipeline.
#
# ### What you'll build:
# 1. Load and parse PDFs into text chunks
# 2. Define Pydantic schemas for sustainability metrics
# 3. Use LLMs to extract structured data from each chunk
# 4. Aggregate and validate results across a full document
# 5. Compare metrics across multiple companies

# %% [markdown]
# ## Part 1: Loading PDFs
#
# We'll use PyMuPDF (fitz) to extract text from PDFs. This handles most
# PDF formats including those with complex layouts.

# %%
import fitz  # PyMuPDF
import os
from pathlib import Path

# Point to our downloaded PDFs
# Works whether run as a script or interactively from the notebooks/ folder
_PROJECT_ROOT = Path(__file__).resolve().parent.parent if '__file__' in dir() else Path.cwd().parent
DATA_DIR = str(_PROJECT_ROOT / "data" / "corporate-sustainability")

# List available PDFs
pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
print("Available sustainability reports:")
for pdf in sorted(pdfs):
    doc = fitz.open(os.path.join(DATA_DIR, pdf))
    print(f"  📄 {pdf} — {doc.page_count} pages")
    doc.close()

# %%
# Load a single PDF and examine its structure
pdf_path = os.path.join(DATA_DIR, "google-env-2024.pdf")
doc = fitz.open(pdf_path)

print(f"Document: {os.path.basename(pdf_path)}")
print(f"Pages: {doc.page_count}")
print(f"Metadata: {doc.metadata}")

# Extract text from first few pages to understand the structure
print(f"\n{'='*60}")
print("FIRST 3 PAGES (truncated):")
print(f"{'='*60}")
for page_num in range(min(3, doc.page_count)):
    page = doc[page_num]
    text = page.get_text()
    print(f"\n--- Page {page_num + 1} ---")
    print(text[:500] + "..." if len(text) > 500 else text)

doc.close()

# %% [markdown]
# ## Part 2: Chunking Strategy
#
# PDFs can be hundreds of pages. LLMs have context limits. We need to 
# split documents into manageable **chunks** while preserving meaning.
#
# Key decisions:
# - **Chunk size**: How many characters per chunk?
# - **Overlap**: How much context to share between adjacent chunks?
# - **Boundaries**: Should we split on pages, paragraphs, or sections?

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_pages(pdf_path: str) -> list[dict]:
    """Load a PDF and return a list of {page_num, text} dicts."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # Skip blank pages
            pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages

def chunk_document(pages: list[dict], chunk_size: int = 4000, overlap: int = 200) -> list[dict]:
    """Split document pages into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    
    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for j, chunk_text in enumerate(page_chunks):
            chunks.append({
                "page_num": page["page_num"],
                "chunk_index": j,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })
    return chunks

# Load and chunk the Google report
pages = load_pdf_pages(pdf_path)
chunks = chunk_document(pages)

print(f"Pages with text: {len(pages)}")
print(f"Total chunks: {len(chunks)}")
print(f"Avg chunk size: {sum(c['char_count'] for c in chunks) / len(chunks):.0f} chars")
print(f"\nSample chunk (page {chunks[10]['page_num']}):")
print(chunks[10]["text"][:300])

# %% [markdown]
# ## Part 3: Define Extraction Schemas
#
# What exactly do we want to pull out of these reports? Let's define 
# precise Pydantic schemas for the sustainability metrics we care about.

# %%
from pydantic import BaseModel, Field
from typing import Optional

class EnergyMetrics(BaseModel):
    """Energy consumption and renewable energy data."""
    total_energy_consumption_mwh: Optional[float] = Field(None, description="Total energy consumed in MWh or equivalent")
    renewable_energy_percentage: Optional[float] = Field(None, description="Percentage of energy from renewable sources")
    renewable_energy_mwh: Optional[float] = Field(None, description="Renewable energy consumed in MWh")
    data_center_energy_mwh: Optional[float] = Field(None, description="Energy consumed by data centers in MWh")
    pue: Optional[float] = Field(None, description="Power Usage Effectiveness ratio for data centers")
    year_reported: Optional[int] = Field(None, description="Year the data refers to")

class EmissionsMetrics(BaseModel):
    """Greenhouse gas emissions data."""
    scope_1_mtco2e: Optional[float] = Field(None, description="Scope 1 emissions in metric tons CO2e")
    scope_2_market_mtco2e: Optional[float] = Field(None, description="Scope 2 market-based emissions in metric tons CO2e")
    scope_2_location_mtco2e: Optional[float] = Field(None, description="Scope 2 location-based emissions in metric tons CO2e") 
    scope_3_mtco2e: Optional[float] = Field(None, description="Scope 3 emissions in metric tons CO2e")
    total_emissions_mtco2e: Optional[float] = Field(None, description="Total emissions in metric tons CO2e")
    year_reported: Optional[int] = Field(None, description="Year the data refers to")

class WaterMetrics(BaseModel):
    """Water usage data."""
    total_water_withdrawal_megaliters: Optional[float] = Field(None, description="Total water withdrawn in megaliters")
    water_consumption_megaliters: Optional[float] = Field(None, description="Total water consumed in megaliters")
    data_center_water_megaliters: Optional[float] = Field(None, description="Water used by data centers in megaliters")
    wue: Optional[float] = Field(None, description="Water Usage Effectiveness ratio")
    year_reported: Optional[int] = Field(None, description="Year the data refers to")

class ClimateTarget(BaseModel):
    """Net zero or emissions reduction targets."""
    target_type: Optional[str] = Field(None, description="Type: 'net zero', 'carbon neutral', 'emissions reduction', etc.")
    target_year: Optional[int] = Field(None, description="Year to achieve the target")
    target_description: str = Field(description="Description of what the target entails")
    baseline_year: Optional[int] = Field(None, description="Baseline year for comparison")
    scope_coverage: Optional[str] = Field(None, description="Which scopes: 'Scope 1', 'Scope 1,2', 'Scope 1,2,3', etc.")
    interim_targets: Optional[str] = Field(None, description="Any intermediate milestones")

class SustainabilityReport(BaseModel):
    """Complete structured extraction from a sustainability report."""
    company_name: str
    report_year: Optional[int] = None
    energy: Optional[EnergyMetrics] = None
    emissions: Optional[EmissionsMetrics] = None
    water: Optional[WaterMetrics] = None
    climate_targets: list[ClimateTarget] = Field(default_factory=list)
    
print("✓ Schemas defined")
print(f"\nSustainabilityReport has {len(SustainabilityReport.model_fields)} top-level fields")
print(f"EnergyMetrics has {len(EnergyMetrics.model_fields)} fields")
print(f"EmissionsMetrics has {len(EmissionsMetrics.model_fields)} fields")

# %% [markdown]
# ## Part 4: LLM-Powered Extraction
#
# Now we'll send each relevant chunk to the LLM and ask it to extract
# any sustainability metrics it finds.

# %%
from openai import OpenAI
import json

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://ellm.nrp-nautilus.io/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
)
MODEL = os.environ.get("OPENAI_MODEL", "qwen3")

EXTRACTION_PROMPT = """You are a sustainability data extraction assistant. 
Analyze the following text chunk from a corporate sustainability report.

Extract any of these metrics you can find. Use EXACT numbers from the text.
Do NOT estimate or infer numbers that aren't explicitly stated.
Return null for any field where data is not present in this chunk.

Return a JSON object with these optional sections:
- "energy": {total_energy_consumption_mwh, renewable_energy_percentage, renewable_energy_mwh, data_center_energy_mwh, pue, year_reported}
- "emissions": {scope_1_mtco2e, scope_2_market_mtco2e, scope_2_location_mtco2e, scope_3_mtco2e, total_emissions_mtco2e, year_reported}
- "water": {total_water_withdrawal_megaliters, water_consumption_megaliters, data_center_water_megaliters, wue, year_reported}
- "climate_targets": [{target_type, target_year, target_description, baseline_year, scope_coverage, interim_targets}]

Only include sections where you find actual data. If the chunk contains no 
relevant metrics at all, return: {"no_data": true}
"""

def extract_from_chunk(chunk_text: str, model: str = MODEL) -> dict:
    """Send a text chunk to the LLM and extract sustainability metrics."""
    # Disable thinking for structured extraction — avoids content=None issue
    # and lets the model use full context for the actual answer.
    # See https://nrp.ai/documentation/userdocs/ai/llm-managed/
    if model == "glm-4.7":
        thinking_kwargs = {"enable_thinking": False}
    else:
        thinking_kwargs = {"thinking": False}
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"Extract sustainability metrics from this text:\n\n{chunk_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,  # We want deterministic extraction
        extra_body={"chat_template_kwargs": thinking_kwargs},
    )
    
    content = response.choices[0].message.content
    if content is None:
        # Fallback in case thinking still consumed all tokens
        print("  ⚠️  Model returned no content. Returning no_data.")
        return {"no_data": True}
    result = json.loads(content)
    return result

# Test on a single chunk — pick one likely to have data
# We'll scan chunks for keywords first
data_chunks = []
keywords = ["emissions", "scope 1", "scope 2", "MWh", "renewable", "carbon", "water", "CO2"]
for chunk in chunks:
    score = sum(1 for kw in keywords if kw.lower() in chunk["text"].lower())
    if score >= 2:
        data_chunks.append((score, chunk))

data_chunks.sort(key=lambda x: x[0], reverse=True)
print(f"Found {len(data_chunks)} chunks with sustainability keywords")
print(f"Top chunk (score={data_chunks[0][0]}) from page {data_chunks[0][1]['page_num']}:")
print(data_chunks[0][1]["text"][:300])

# %%
# Extract from the top chunk
test_chunk = data_chunks[0][1]
print(f"Extracting from page {test_chunk['page_num']}...\n")

result = extract_from_chunk(test_chunk["text"])
print("Extracted:")
print(json.dumps(result, indent=2))

# %% [markdown]
# ## Part 5: Full Document Extraction
#
# Now let's process all data-rich chunks and aggregate the results.
# This is where the pipeline really shines — automating what would take
# hours of manual work.

# %%
import time

def extract_full_report(pdf_path: str, max_chunks: int = 30) -> dict:
    """Extract sustainability metrics from an entire PDF report."""
    
    filename = os.path.basename(pdf_path)
    print(f"📄 Processing: {filename}")
    
    # Load and chunk
    pages = load_pdf_pages(pdf_path)
    chunks = chunk_document(pages)
    print(f"   {len(pages)} pages → {len(chunks)} chunks")
    
    # Filter to data-rich chunks
    keywords = ["emissions", "scope 1", "scope 2", "scope 3", "MWh", "TWh", "GWh",
                 "renewable", "carbon", "CO2", "water", "net zero", "target", 
                 "megaliters", "metric tons", "mtco2e"]
    
    scored = []
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw.lower() in chunk["text"].lower())
        if score >= 2:
            scored.append((score, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:max_chunks]
    print(f"   Selected top {len(selected)} data-rich chunks (of {len(scored)} candidates)")
    
    # Extract from each chunk
    all_results = []
    for i, (score, chunk) in enumerate(selected):
        try:
            result = extract_from_chunk(chunk["text"])
            if not result.get("no_data"):
                result["_source_page"] = chunk["page_num"]
                all_results.append(result)
                print(f"   ✓ Chunk {i+1}/{len(selected)} (page {chunk['page_num']}): data found")
            else:
                print(f"   · Chunk {i+1}/{len(selected)} (page {chunk['page_num']}): no data")
        except Exception as e:
            print(f"   ✗ Chunk {i+1}/{len(selected)}: error - {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    print(f"   📊 Extracted data from {len(all_results)} chunks")
    return all_results

# Process one report as a demonstration
results = extract_full_report(pdf_path, max_chunks=10)  # Limit for demo
print(f"\n{'='*60}")
print("RESULTS:")
print(json.dumps(results[:3], indent=2))  # Show first 3

# %% [markdown]
# ## Part 6: Multi-Report Comparison
# 
# The real power is comparing across companies. Let's process all four
# reports and build a comparison table.

# %%
import pandas as pd

def aggregate_report_results(results: list[dict], company_name: str) -> dict:
    """Aggregate extraction results from multiple chunks into a single record."""
    
    aggregated = {"company": company_name}
    
    # Collect all non-null values for each metric
    for result in results:
        if "emissions" in result and result["emissions"]:
            emissions = result["emissions"]
            for key, val in emissions.items():
                if val is not None and key != "year_reported":
                    field = f"emissions_{key}"
                    if field not in aggregated or aggregated[field] is None:
                        aggregated[field] = val
                if key == "year_reported" and val:
                    aggregated["emissions_year"] = val
        
        if "energy" in result and result["energy"]:
            energy = result["energy"]
            for key, val in energy.items():
                if val is not None and key != "year_reported":
                    field = f"energy_{key}"
                    if field not in aggregated or aggregated[field] is None:
                        aggregated[field] = val
                if key == "year_reported" and val:
                    aggregated["energy_year"] = val
        
        if "water" in result and result["water"]:
            water = result["water"]
            for key, val in water.items():
                if val is not None and key != "year_reported":
                    field = f"water_{key}"
                    if field not in aggregated or aggregated[field] is None:
                        aggregated[field] = val
        
        if "climate_targets" in result:
            targets = result["climate_targets"]
            if isinstance(targets, list):
                if "targets" not in aggregated:
                    aggregated["targets"] = []
                aggregated["targets"].extend(targets)
    
    return aggregated

# Process all available reports (limit chunks for class time)
all_company_results = {}

report_files = {
    "Google": "google-env-2024.pdf",
    "Apple": "apple-env-2024.pdf",
    "Amazon": "amazon-sustainability-2023.pdf",
    "BP": "bp-sustainability-2023.pdf",
}

for company, filename in report_files.items():
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        results = extract_full_report(path, max_chunks=8)
        all_company_results[company] = aggregate_report_results(results, company)
        print()

# Build comparison DataFrame
df = pd.DataFrame(all_company_results.values())
print("\n📊 SUSTAINABILITY METRICS COMPARISON")
print("=" * 60)

# Show key columns
key_cols = ["company", "emissions_scope_1_mtco2e", "emissions_scope_2_market_mtco2e", 
            "emissions_scope_3_mtco2e", "energy_renewable_energy_percentage"]
available_cols = [c for c in key_cols if c in df.columns]
print(df[available_cols].to_string(index=False))

# %% [markdown]
# ## Part 7: Validation — Checking for Hallucinations
#
# LLMs can hallucinate numbers. Always validate extracted data:
# 1. **Source tracing**: Which page did this number come from?
# 2. **Cross-reference**: Do numbers from different chunks agree?
# 3. **Sanity checks**: Are the values in reasonable ranges?

# %%
def validate_emissions(aggregated: dict) -> list[str]:
    """Run basic sanity checks on extracted emissions data."""
    warnings = []
    company = aggregated.get("company", "Unknown")
    
    # Check: Scope 1 should typically be < Scope 2 for tech companies
    s1 = aggregated.get("emissions_scope_1_mtco2e")
    s2 = aggregated.get("emissions_scope_2_market_mtco2e")
    s3 = aggregated.get("emissions_scope_3_mtco2e")
    
    if s1 and s2 and s1 > s2 * 10:
        warnings.append(f"⚠️  {company}: Scope 1 ({s1:,.0f}) is 10x larger than Scope 2 ({s2:,.0f}) — unusual for tech companies")
    
    if s3 and s1 and s3 < s1:
        warnings.append(f"⚠️  {company}: Scope 3 ({s3:,.0f}) is less than Scope 1 ({s1:,.0f}) — Scope 3 is typically largest")
    
    # Check: Renewable energy percentage should be 0-100
    re_pct = aggregated.get("energy_renewable_energy_percentage")
    if re_pct and (re_pct < 0 or re_pct > 100):
        warnings.append(f"⚠️  {company}: Renewable energy {re_pct}% is outside 0-100 range")
    
    if not warnings:
        warnings.append(f"✓ {company}: All basic checks passed")
    
    return warnings

# Validate all companies
print("🔍 VALIDATION RESULTS")
print("=" * 60)
for company, data in all_company_results.items():
    for warning in validate_emissions(data):
        print(warning)

# %% [markdown]
# ## 💾 Save Results
# 
# Export the structured data to JSON and CSV for further analysis.

# %%
# Save as JSON (preserving full structure)
output_dir = str(_PROJECT_ROOT / "output")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "sustainability_extraction.json"), "w") as f:
    json.dump(all_company_results, f, indent=2, default=str)

# Save comparison table as CSV
df.to_csv(os.path.join(output_dir, "sustainability_comparison.csv"), index=False)

print(f"✓ Results saved to {output_dir}/")
print(f"  - sustainability_extraction.json (full structured data)")
print(f"  - sustainability_comparison.csv (comparison table)")

# %% [markdown]
# ## 🎯 Session 9 Takeaways
#
# 1. **PDF → Text → Chunks → LLM → JSON**: The core extraction pipeline
# 2. **Pydantic schemas** define exactly what you want to extract
# 3. **Keyword filtering** reduces API calls by skipping irrelevant chunks
# 4. **Aggregation** combines results from multiple chunks into one record
# 5. **Validation** catches hallucinations and extraction errors
# 6. The same pipeline works across companies, enabling **comparison at scale**
#
# ### What you built:
# A pipeline that transforms 400+ pages of corporate PDFs into a clean
# comparison table in minutes — work that would take a human analyst days.
#
# ### Next Session:
# In Session 10, we'll use **RAG (Retrieval-Augmented Generation)** and 
# **Model Context Protocol** to ask questions across multiple documents
# simultaneously, without processing every page.
