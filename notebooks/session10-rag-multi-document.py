# %% [markdown]
# # Session 10: RAG & Multi-Document Climate Policy Analysis
#
# ## From Single Documents to Cross-Document Intelligence
#
# In Session 9, we extracted metrics from individual PDFs one at a time.
# But real climate analysis often requires **cross-document questions**:
#
# - "Which city has the most ambitious GHG reduction target?"
# - "How do renewable energy projections differ between the DOE and EIA?"
# - "What building electrification strategies appear across multiple city plans?"
#
# Processing every page of every document for each question is expensive
# and slow. **RAG (Retrieval-Augmented Generation)** solves this by:
#
# 1. **Embedding** all document chunks into a vector database (once)
# 2. **Retrieving** only the most relevant chunks for each question
# 3. **Generating** answers grounded in the retrieved context
#
# ### What you'll build:
# 1. A vector database of city climate action plans
# 2. A RAG pipeline that answers questions across all documents
# 3. Structured extraction of policy targets across multiple cities
# 4. A comparison analysis of climate ambition across jurisdictions

# %% [markdown]
# ## Part 1: Building the Vector Store
#
# First, we'll load all our climate action plan PDFs, chunk them, and 
# embed them into a ChromaDB vector database.

# %%
import fitz  # PyMuPDF
import os
import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Our document collections
_PROJECT_ROOT = Path(__file__).resolve().parent.parent if '__file__' in dir() else Path.cwd().parent
CAP_DIR = str(_PROJECT_ROOT / "data" / "climate-action-plans")
ENERGY_DIR = str(_PROJECT_ROOT / "data" / "utility-irps")

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 2000, overlap: int = 200) -> list[dict]:
    """Load a PDF and return chunked text with metadata."""
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    
    all_chunks = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        if not text.strip():
            continue
            
        page_chunks = splitter.split_text(text)
        for j, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "page": page_num + 1,
                    "chunk_index": j,
                }
            })
    
    doc.close()
    return all_chunks

# Load all climate action plans
print("Loading climate action plans...")
all_chunks = []
for pdf_file in sorted(os.listdir(CAP_DIR)):
    if pdf_file.endswith('.pdf'):
        path = os.path.join(CAP_DIR, pdf_file)
        chunks = load_and_chunk_pdf(path)
        all_chunks.extend(chunks)
        print(f"  📄 {pdf_file}: {len(chunks)} chunks")

# Also load energy planning docs
print("\nLoading energy planning reports...")
for pdf_file in sorted(os.listdir(ENERGY_DIR)):
    if pdf_file.endswith('.pdf'):
        path = os.path.join(ENERGY_DIR, pdf_file)
        chunks = load_and_chunk_pdf(path)
        all_chunks.extend(chunks)
        print(f"  📄 {pdf_file}: {len(chunks)} chunks")

print(f"\nTotal: {len(all_chunks)} chunks from {len(os.listdir(CAP_DIR)) + len(os.listdir(ENERGY_DIR))} documents")

# %% [markdown]
# ## Part 2: Create Embeddings & Vector Database
#
# We'll use a local embedding model to convert text chunks into vectors,
# then store them in ChromaDB for fast similarity search.

# %%
from sentence_transformers import SentenceTransformer

# Use a compact, fast embedding model
# This runs locally — no API calls needed!
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Test it
test_embedding = embed_model.encode("climate change mitigation targets")
print(f"Embedding dimension: {len(test_embedding)}")
print(f"First 5 values: {test_embedding[:5]}")

# %%
# Create ChromaDB collection
db_path = str(_PROJECT_ROOT / "output" / "vectordb")
os.makedirs(db_path, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=db_path)

# Delete existing collection if re-running
try:
    chroma_client.delete_collection("climate_docs")
except:
    pass

collection = chroma_client.create_collection(
    name="climate_docs",
    metadata={"description": "Climate action plans and energy planning documents"}
)

# Batch embed and add to collection
BATCH_SIZE = 100
print(f"Embedding {len(all_chunks)} chunks in batches of {BATCH_SIZE}...")

for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i:i + BATCH_SIZE]
    texts = [c["text"] for c in batch]
    metadatas = [c["metadata"] for c in batch]
    ids = [f"chunk_{i+j}" for j in range(len(batch))]
    
    # Embed locally
    embeddings = embed_model.encode(texts).tolist()
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Processed {i + len(batch)}/{len(all_chunks)} chunks")

print(f"✓ Vector database created with {collection.count()} chunks")

# %% [markdown]
# ## Part 3: RAG Queries
#
# Now we can ask questions and get answers grounded in the actual documents.
# The pipeline:
# 1. **Embed** the question  
# 2. **Retrieve** the most similar chunks
# 3. **Generate** an answer using the retrieved context

# %%
from openai import OpenAI
import json

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://ellm.nrp-nautilus.io/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
)
MODEL = os.environ.get("OPENAI_MODEL", "qwen3")

def rag_query(question: str, n_results: int = 8, show_sources: bool = True) -> str:
    """Answer a question using RAG over the climate documents."""
    
    # Step 1: Embed the question
    query_embedding = embed_model.encode(question).tolist()
    
    # Step 2: Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    
    # Build context from retrieved chunks
    context_parts = []
    sources = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        source = metadata["source"]
        page = metadata["page"]
        context_parts.append(f"[Source: {source}, Page {page}]\n{doc}")
        sources.append(f"{source} (p.{page})")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Step 3: Generate answer with context
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a climate policy analyst. Answer questions using ONLY the provided context from climate documents. 
                
Rules:
- Cite specific documents and page numbers when stating facts
- If the context doesn't contain enough information, say so
- Never make up numbers or policies not in the context
- Compare across documents when relevant"""
            },
            {
                "role": "user",
                "content": f"Context from climate documents:\n\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.1,
    )
    
    answer = response.choices[0].message.content
    if answer is None:
        answer = "⚠️ Model returned no content. This can happen with reasoning models — try a different model or disable thinking."
    
    if show_sources:
        unique_sources = list(dict.fromkeys(sources))  # Deduplicate preserving order
        answer += f"\n\n📚 Sources consulted: {', '.join(unique_sources)}"
    
    return answer

# %%
# Test with a cross-document question
question = "What GHG reduction targets have cities set for 2030? Compare the targets across different cities."
print(f"❓ {question}\n")
print(rag_query(question))

# %%
# Ask about renewable energy
question = "What renewable energy goals or solar deployment targets appear across these documents?"
print(f"❓ {question}\n")
print(rag_query(question))

# %%
# Ask about building electrification
question = "What building electrification strategies or natural gas phaseout policies are discussed?"
print(f"❓ {question}\n")
print(rag_query(question))

# %% [markdown]
# ## Part 4: Structured Cross-Document Extraction
#
# RAG gives us free-text answers. But we can combine RAG + structured 
# outputs to extract comparable data across documents.

# %%
from pydantic import BaseModel, Field
from typing import Optional

class CityClimateTarget(BaseModel):
    """Structured climate target from a city plan."""
    city_or_jurisdiction: str = Field(description="Name of the city, county, or state")
    document_name: str = Field(description="Source document filename")
    ghg_reduction_target: Optional[str] = Field(None, description="GHG reduction target (e.g., '40% below 1990 by 2030')")
    target_year: Optional[int] = Field(None, description="Year to achieve the target")
    baseline_year: Optional[int] = Field(None, description="Baseline year for measuring reduction")
    carbon_neutrality_target: Optional[str] = Field(None, description="Carbon neutrality or net zero goal if any")
    renewable_energy_target: Optional[str] = Field(None, description="Renewable energy percentage or capacity goal")
    building_electrification: Optional[str] = Field(None, description="Building electrification goals or policies")
    transportation_target: Optional[str] = Field(None, description="EV or transportation emissions goals")
    equity_focus: Optional[str] = Field(None, description="Environmental justice or equity commitments")

def extract_city_targets(source_filter: str = None) -> list[dict]:
    """Extract structured climate targets from each document."""
    
    # Get all unique source documents
    all_sources = set()
    sample = collection.get(limit=collection.count(), include=["metadatas"])
    for meta in sample["metadatas"]:
        all_sources.add(meta["source"])
    
    # Filter to climate action plans (not energy reports)
    cap_sources = sorted([s for s in all_sources 
                          if any(kw in s.lower() for kw in ["cap", "ecap", "climate", "scoping", "ipcc", "austin", "portland", "seattle", "annarbor", "oakland"])])
    
    print(f"Climate plan documents: {cap_sources}")
    
    results = []
    for source in cap_sources:
        print(f"\n📄 Extracting targets from: {source}")
        
        # Use RAG to find target-related content from this specific document
        query_embedding = embed_model.encode(
            f"GHG emissions reduction target goal net zero renewable energy building electrification {source}"
        ).tolist()
        
        doc_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"source": source},
        )
        
        if not doc_results["documents"][0]:
            print(f"   No results found")
            continue
        
        context = "\n\n".join(doc_results["documents"][0])
        
        # Disable thinking for structured extraction to avoid content=None
        if MODEL == "glm-4.7":
            thinking_kwargs = {"enable_thinking": False}
        else:
            thinking_kwargs = {"thinking": False}
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""Extract climate targets from this document ({source}). 
Return JSON with: city_or_jurisdiction, document_name, ghg_reduction_target, target_year, baseline_year, 
carbon_neutrality_target, renewable_energy_target, building_electrification, transportation_target, equity_focus.
Use null for fields not found in the text. Use exact quotes where possible."""
                },
                {"role": "user", "content": context}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            extra_body={"chat_template_kwargs": thinking_kwargs},
        )
        
        try:
            content = response.choices[0].message.content
            if content is None:
                print(f"   ⚠️ Model returned no content (reasoning tokens exhausted)")
                continue
            extracted = json.loads(content)
            extracted["document_name"] = source  # Ensure correct source
            results.append(extracted)
            print(f"   ✓ Target: {extracted.get('ghg_reduction_target', 'N/A')}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    return results

# Run extraction
city_targets = extract_city_targets()

# %%
import pandas as pd

# Build comparison table
df = pd.DataFrame(city_targets)

print("\n🏙️ CITY CLIMATE TARGETS COMPARISON")
print("=" * 80)

display_cols = ["city_or_jurisdiction", "ghg_reduction_target", "target_year", 
                "carbon_neutrality_target", "renewable_energy_target"]
available = [c for c in display_cols if c in df.columns]
print(df[available].to_string(index=False))

# %% [markdown]
# ## Part 5: Ambition Analysis
#
# Let's use the LLM to analyze the extracted targets and rank cities
# by climate ambition.

# %%
# Send all extracted targets to the LLM for comparative analysis
targets_summary = json.dumps(city_targets, indent=2, default=str)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": """You are a climate policy analyst. Analyze these extracted climate targets 
from multiple city and state plans. Provide:
1. A ranking of jurisdictions by ambition (most to least ambitious)
2. Common strategies that appear across multiple plans
3. Notable gaps or weaknesses in the targets
4. Which targets are most/least likely to be achieved and why

Be specific — cite actual numbers and policies from the data."""
        },
        {"role": "user", "content": f"Analyze these climate targets:\n\n{targets_summary}"}
    ],
    temperature=0.3,
)

print("📊 CLIMATE AMBITION ANALYSIS")
print("=" * 60)
analysis_content = response.choices[0].message.content
if analysis_content is None:
    print("⚠️ Model returned no content. Try a different model or disable thinking.")
else:
    print(analysis_content)

# %% [markdown]
# ## Part 6: Energy Futures — Cross-Document Questions
#
# Let's also use RAG to compare the federal energy planning reports.

# %%
energy_questions = [
    "What does the DOE Solar Futures Study project for solar capacity by 2035 and 2050?",
    "What are the main barriers to electrification identified in the NREL Electrification Futures Study?",
    "What does the EIA Annual Energy Outlook project for natural gas consumption through 2050?",
    "How do the renewable energy projections in these reports compare to each other?",
]

print("⚡ ENERGY FUTURES ANALYSIS")
print("=" * 60)

for q in energy_questions:
    print(f"\n❓ {q}\n")
    answer = rag_query(q, n_results=6)
    print(answer)
    print()

# %% [markdown]
# ## 💾 Save Everything

# %%
output_dir = str(_PROJECT_ROOT / "output")
os.makedirs(output_dir, exist_ok=True)

# Save city targets
with open(os.path.join(output_dir, "city_climate_targets.json"), "w") as f:
    json.dump(city_targets, f, indent=2, default=str)

df.to_csv(os.path.join(output_dir, "city_climate_targets.csv"), index=False)

print(f"✓ Saved to {output_dir}/")
print(f"  - city_climate_targets.json")
print(f"  - city_climate_targets.csv")
print(f"  - vectordb/ (ChromaDB persistent store)")

# %% [markdown]
# ## 🎯 Session 10 Takeaways
#
# 1. **RAG** = Embed chunks → Retrieve relevant ones → Generate grounded answers
# 2. **Vector databases** (ChromaDB) make retrieval fast over thousands of chunks
# 3. **Local embeddings** (sentence-transformers) avoid API costs for indexing
# 4. RAG + **structured outputs** enables cross-document comparison tables
# 5. **Source attribution** is critical — always know which document and page
# 6. Different documents may use different units, baselines, and definitions —
#    the LLM can help normalize, but **human review is essential**
#
# ### What you built:
# A system that can answer arbitrary questions across 11+ climate documents,
# extract comparable structured data, and produce ranked analysis — 
# capabilities that real climate analysts need every day.
#
# ### Module 3 Assessment:
# Submit a working document extraction pipeline that processes real climate
# PDFs and outputs structured data (JSON/CSV). Your system should demonstrate:
# - PDF loading and intelligent chunking
# - LLM-powered structured extraction with Pydantic schemas
# - RAG for cross-document questions
# - Validation and source attribution
