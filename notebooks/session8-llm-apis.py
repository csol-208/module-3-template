# %% [markdown]
# # Session 8: Your First LLM Pipeline
# 
# ## Introduction to LLM APIs & Structured Extraction
#
# In Modules 1 and 2, we worked with **structured data** — CSV files, parquet 
# databases, and GeoJSON. The data came pre-organized in rows and columns.
#
# But most of the world's climate-relevant information lives in **unstructured
# documents**: sustainability reports, climate action plans, policy briefs, 
# and regulatory filings. These are PDFs full of prose, tables, charts, and
# mixed formatting.
#
# In this session, you'll learn to use LLM APIs **programmatically** — not 
# through a chat window, but through Python code that can be automated, 
# reproduced, and scaled.
#
# ### What you'll build:
# 1. Connect to an LLM API using the OpenAI-compatible interface
# 2. Use **structured outputs** (Pydantic schemas) to extract specific fields
# 3. Compare responses across different models
# 4. Extract climate commitment data from a text passage

# %% [markdown]
# ## Part 1: Connecting to an LLM API
#
# The OpenAI Python library works with any API that follows the OpenAI format.
# This includes OpenRouter, local models (GLM-4, Qwen-3), and cloud providers.
#
# The key configuration is:
# - `base_url`: Where the API lives  
# - `api_key`: Your authentication token
# - `model`: Which model to use

# %%
from openai import OpenAI
import os

# Configure the client to point at our API endpoint
# The base_url and api_key will be set as environment variables
client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://ellm.nrp-nautilus.io/v1"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Pick a model — we have access to several open-source models
# Available: qwen3, glm-4.7, gpt-oss, gemma3, kimi, llama3-sdsc
MODEL = os.environ.get("OPENAI_MODEL", "qwen3")

# Test the connection with a simple prompt
# Note: NRP docs recommend NOT specifying max_tokens at all.
# See https://nrp.ai/documentation/userdocs/ai/llm-managed/
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "What is the Paris Agreement temperature target? Answer in one sentence."}
    ],
)

print(f"Model: {MODEL}")
print(f"Response: {response.choices[0].message.content}")

# %% [markdown]
# ## Part 2: Structured Outputs with Pydantic
#
# Chat responses give us free-form text. But for data extraction, we need
# **structured outputs** — responses that follow a specific schema.
#
# We define what we want using Pydantic models, and the API returns JSON
# that exactly matches our schema.

# %%
from pydantic import BaseModel, Field
from typing import Optional

# Define what we want to extract
# We use model_config to allow coercion (e.g., "2030" → 2030)
# because LLMs sometimes return numbers as strings
class ClimateCommitment(BaseModel):
    """Structured representation of a corporate climate commitment."""
    model_config = {"coerce_numbers_to_str": False, "strict": False}

    company_name: str = Field(description="Name of the company")
    target_year: Optional[int] = Field(None, description="Target year for the commitment (e.g., 2030, 2050)")
    target_description: str = Field(description="What the company committed to (e.g., 'net zero emissions')")
    baseline_year: Optional[int] = Field(None, description="Baseline year for measuring progress")
    scope_coverage: Optional[str] = Field(None, description="Which emission scopes are covered (e.g., 'Scope 1 and 2', 'Scope 1, 2, and 3')")
    interim_target: Optional[str] = Field(None, description="Any intermediate target before the main goal")

print("Schema defined! Fields:")
for name, field in ClimateCommitment.model_fields.items():
    print(f"  {name}: {field.annotation} — {field.description}")

# %% [markdown]
# ## Part 3: Extract from a Text Passage
#
# Here's a real passage from a corporate sustainability report. Let's 
# extract structured data from it using our schema.

# %%
# A real passage about corporate climate commitments
sample_text = """
Apple has committed to becoming carbon neutral across its entire supply chain 
and product life cycle by 2030. This includes Scope 1, Scope 2, and Scope 3 
emissions. Using a 2015 baseline, Apple has already reduced its comprehensive 
carbon footprint by more than 55 percent since 2015. As an interim milestone, 
Apple achieved carbon neutrality for its global corporate operations in 2020. 
The company's approach prioritizes direct emissions reductions of 75 percent 
from 2015 levels, with the remaining 25 percent addressed through high-quality 
carbon removal projects.
"""

# Use the LLM to extract structured data
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a data extraction assistant. Extract the requested fields from the provided text. Return valid JSON matching the schema exactly."
        },
        {
            "role": "user", 
            "content": f"Extract the climate commitment details from this text:\n\n{sample_text}\n\nReturn JSON with these fields: company_name, target_year, target_description, baseline_year, scope_coverage, interim_target"
        }
    ],
    response_format={"type": "json_object"},
    # Disable thinking for structured extraction — avoids content=None issue
    # qwen3 uses {"thinking": False}, glm-4.7 uses {"enable_thinking": False}
    extra_body={"chat_template_kwargs": {"thinking": False}},
)

import json
extracted = json.loads(response.choices[0].message.content)
print("Extracted data:")
print(json.dumps(extracted, indent=2))

# Validate against our schema
commitment = ClimateCommitment(**extracted)
print(f"\n✓ Valid! {commitment.company_name} → {commitment.target_description} by {commitment.target_year}")

# %% [markdown]
# ## Part 4: Compare Across Models
#
# Different models may extract different information or make different 
# mistakes. Let's test the same extraction with multiple models.

# %%
# A more ambiguous passage to test model differences
ambiguous_text = """
Google has set ambitious sustainability goals. The company aims to run on 
24/7 carbon-free energy on every grid where it operates by 2030. Google 
achieved carbon neutrality in 2007 and has been purchasing enough renewable 
energy to match 100% of its annual electricity consumption since 2017. 
However, the company's total energy consumption has grown significantly — 
its data centers used approximately 25.3 TWh of electricity in 2023, 
a 17% increase over the previous year. Google has also committed to 
achieving net-zero emissions across all of its operations and value chain 
by 2030, covering Scopes 1, 2, and 3.
"""

# Try different models to compare extraction results
# Available models at our endpoint: qwen3, glm-4.7, gpt-oss, gemma3, kimi
models_to_test = ["qwen3", "glm-4.7"]

for model_name in models_to_test:
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    
    try:
        # Disable "thinking" mode for structured extraction (avoids content=None)
        # Different models use different param names:
        if model_name == "glm-4.7":
            thinking_kwargs = {"enable_thinking": False}
        else:
            thinking_kwargs = {"thinking": False}
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Extract climate commitment details as JSON. Fields: company_name, target_year, target_description, baseline_year, scope_coverage, interim_target"
                },
                {"role": "user", "content": ambiguous_text}
            ],
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": thinking_kwargs},
        )
        
        content = response.choices[0].message.content
        if content is None:
            print("  ⚠️ Model returned no content")
            continue
        result = json.loads(content)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error with {model_name}: {e}")

# %% [markdown]
# ## Part 5: Batch Extraction
#
# Now let's scale up — extract commitments from multiple text passages,
# building toward the PDF pipeline we'll create in Session 9.

# %%
# Multiple companies' commitment texts
company_texts = {
    "Amazon": """
    Amazon's Climate Pledge commits the company to reaching net-zero carbon 
    emissions by 2040 — 10 years ahead of the Paris Agreement. The pledge 
    covers all three emission scopes. As of 2023, Amazon has reduced the 
    carbon intensity of its shipments by 11.5% compared to a 2019 baseline. 
    The company is the world's largest corporate purchaser of renewable energy, 
    with 500+ solar and wind projects globally.
    """,
    
    "BP": """
    BP aims to become a net zero company by 2050 or sooner, and to help the 
    world get to net zero. This ambition covers Scope 1, 2, and 3 emissions. 
    BP's interim target is to reduce Scope 1 and 2 operational emissions by 
    50% by 2030, compared to a 2019 baseline. The company also aims to reduce 
    the carbon intensity of the products it sells by 15-20% by 2030. BP has 
    faced criticism for potentially weakening its climate targets in 2023.
    """,
    
    "Apple": sample_text,
}

import pandas as pd

results = []
for company, text in company_texts.items():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Extract climate commitment as JSON. Fields: company_name, target_year, target_description, baseline_year, scope_coverage, interim_target. Use null for missing fields."
            },
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
    
    content = response.choices[0].message.content
    if content is None:
        print(f"  ⚠️ {company}: Model returned no content, skipping")
        continue
    extracted = json.loads(content)
    results.append(extracted)
    print(f"✓ Extracted: {company}")

# Create a comparison table
df = pd.DataFrame(results)
print("\n📊 Climate Commitments Comparison:")
print(df.to_string(index=False))

# %% [markdown]
# ## 🎯 Session 8 Takeaways
#
# 1. **LLM APIs** follow the OpenAI chat completion pattern — messages in, text out
# 2. **Structured outputs** let you define a schema and get JSON back
# 3. **Pydantic models** validate that extracted data matches your expectations
# 4. Different models may extract different information — **always validate**
# 5. This pattern scales: same code works for 3 companies or 3,000
#
# ### Next Session: 
# In Session 9, we'll apply these same techniques to **real PDF documents** — 
# loading, chunking, and extracting structured data from corporate sustainability 
# reports and city climate action plans.
