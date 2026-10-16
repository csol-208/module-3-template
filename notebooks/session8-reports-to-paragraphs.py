# %% [markdown]
# # Session 8: From reports to paragraphs
#
# **Instructor answer key.** Students build this with their coding agent by following `session8.md`.
#
# Between 2019 and 2021, Google, Microsoft, Amazon, Meta and Apple each made headline climate
# pledges: carbon neutral, carbon negative, net zero, 24/7 carbon-free energy. Since then, the
# energy demands of AI have grown quickly. This module asks whether these companies are backing away
# from their pledges, as measured by what they say in their own annual environmental reports.
#
# This session turns 37 reports into one table of paragraphs, looks at how word use changes over
# time, and makes the first calls to a language model and a decision model.
#
# Run `bash download_data.sh` first. API calls need `OPENROUTER_API_KEY` set in the environment.

# %%
import os
import re
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pymupdf
import requests
import seaborn as sns

matplotlib.use("Agg")
Path("output/figures").mkdir(parents=True, exist_ok=True)
reports = pd.read_csv("reports.csv")
reports

# %% [markdown]
# ## Part 1: What the computer sees in a PDF
#
# Google's 2024 report says, on page 41, that it is no longer maintaining operational carbon
# neutrality. Find that sentence in the raw text of the page.

# %%
doc = pymupdf.open("data/reports/google/google-2024.pdf")
page_text = doc[40].get_text()
print(page_text[:1500])

# %%
# A literal search for the sentence fails: the PDF breaks lines in the middle of it.
phrase = "no longer maintaining operational carbon neutrality"
print("literal match:", phrase in page_text)
print("after collapsing whitespace:", phrase in re.sub(r"\s+", " ", page_text))

# %% [markdown]
# ## Part 2: One row per paragraph
#
# PyMuPDF returns text in *blocks* with their position on the page. In most reports a block is a
# paragraph, but some PDFs (Meta's especially) store every line as its own block. We merge a block
# into the one above when it starts at the same left edge and the gap between them is smaller than
# a line. Fragments shorter than 200 characters (headings, captions, page furniture) are dropped.

# %%
def paragraphs(page):
    """Return the paragraphs on a PDF page, merging blocks that continue the one above."""
    paras, cur = [], None
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        lines = text.count("\n") or 1
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if cur and abs(x0 - cur["x0"]) < 4 and 0 <= y0 - cur["y1"] < 0.8 * cur["line_height"]:
            cur["text"] += " " + text
            cur["y1"] = y1
        else:
            if cur:
                paras.append(cur["text"])
            cur = {"x0": x0, "y1": y1, "line_height": (y1 - y0) / lines, "text": text}
    if cur:
        paras.append(cur["text"])
    return paras


# Meta 2023: count raw blocks against merged paragraphs on one page
page = pymupdf.open("data/reports/meta/meta-2023.pdf")[8]
print(len(page.get_text("blocks")), "blocks ->", len(paragraphs(page)), "paragraphs")

# %%
rows = []
for r in reports.itertuples():
    path = Path(f"data/reports/{r.company}/{r.company}-{r.report}.pdf")
    if not path.exists():
        continue
    for page_number, page in enumerate(pymupdf.open(path), start=1):
        for text in paragraphs(page):
            if len(text) >= 200:
                rows.append((r.company, r.report, r.data_year, page_number, text))

paras = pd.DataFrame(rows, columns=["company", "report", "data_year", "page", "text"])
paras.to_parquet("output/paragraphs.parquet")
print(len(paras), "paragraphs")
paras.groupby(["company", "report"]).size().unstack("company")

# %% [markdown]
# Two things to notice. Google's 2021 and 2022 reports are short summaries (16–17 pages), so any
# raw count will dip in those years. Meta's 2019 report has pages whose text is garbled: run
# `paras[paras.company.eq("meta") & paras.report.eq(2019)].text.head()` and look.

# %% [markdown]
# ## Part 3: Keyword trends
#
# Count paragraphs mentioning each pledge term, per report. Reports differ a lot in length, so the
# figure shows mentions per 100 paragraphs.

# %%
terms = {
    "carbon neutral": r"carbon[- ]neutral",
    "net zero": r"net[- ]zero",
    "carbon negative": r"carbon[- ]negative",
    "24/7 carbon-free": r"24/7",
    "offsets": r"\boffsets?\b",
    "AI": r"\bAI\b|artificial intelligence",
}
for name, pattern in terms.items():
    paras[name] = paras.text.str.contains(pattern, case=False if name != "AI" else True, regex=True)

trend = (paras.groupby(["company", "data_year"])[list(terms)].mean().mul(100)
         .reset_index().melt(["company", "data_year"], var_name="term", value_name="per_100"))

g = sns.relplot(trend, x="data_year", y="per_100", hue="company", col="term", col_wrap=3,
                kind="line", marker="o", height=2.6, aspect=1.3, facet_kws={"sharey": False})
g.set_axis_labels("data year", "paragraphs per 100")
g.savefig("output/figures/s8-keyword-trends.png", dpi=120)
trend.pivot_table(index=["term", "company"], columns="data_year", values="per_100").round(1)

# %% [markdown]
# Google's "carbon neutral" drops to almost nothing after the 2022 data year, while "AI" climbs
# for every company. But a mention is not a claim. Read the Google mentions of "carbon neutral"
# that remain after 2022:

# %%
late = paras[paras.company.eq("google") & paras.data_year.ge(2023) & paras["carbon neutral"]]
for r in late.itertuples():
    print(f"[{r.report} p{r.page}]", r.text[:300], "\n")

# %% [markdown]
# ## Part 4: A first call to a language model
#
# OpenRouter gives one API key access to hundreds of models through the OpenAI-compatible
# interface. We use `deepseek/deepseek-v4-flash`, which is cheap and reliable.

# %%
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
MODEL = "deepseek/deepseek-v4-flash"

paragraph = paras[paras.text.str.contains("no longer maintaining operational")].text.iloc[0]
response = client.chat.completions.create(
    model=MODEL, temperature=0,
    messages=[{"role": "user", "content": f"In one sentence, what does this paragraph say about Google's climate goals?\n\n{paragraph}"}],
)
print(response.choices[0].message.content)

# %% [markdown]
# Free text is hard to put in a table. A Pydantic schema asks for named, typed fields instead.

# %%
from typing import Literal, Optional
from pydantic import BaseModel, Field


class GoalStatement(BaseModel):
    goal: str = Field(description="the climate goal discussed, in a few words")
    target_year: Optional[int] = Field(None, description="year the goal should be met, if stated")
    status: Literal["new", "reaffirmed", "progress", "harder", "revised", "dropped"] = Field(
        description="what the paragraph says is happening to the goal")


t0 = time.time()
response = client.chat.completions.parse(
    model=MODEL, temperature=0, response_format=GoalStatement,
    messages=[{"role": "system", "content": "Extract the main climate goal discussed in the paragraph. Use only the paragraph."},
              {"role": "user", "content": paragraph}],
)
llm_seconds = time.time() - t0
print(response.choices[0].message.parsed, f"({llm_seconds:.1f} s)")

# %% [markdown]
# ## Part 5: A first call to a decision model
#
# [Jev](https://openrouter.ai/docs/guides/community/jev) is a *decision model*. It does not write
# text. You send it a text (the `state`) and typed questions, and it returns a probability for
# each possible answer. It is served through OpenRouter with the same key.

# %%
JEV_URL = "https://openrouter.ai/api/v1/systemone"
JEV_HEADERS = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


def decide(text, questions):
    """Ask Jev typed questions about a text. Returns the answers and the cost in USD."""
    r = requests.post(JEV_URL, headers=JEV_HEADERS, timeout=120,
                      json={"model": "~typesafe/jev-latest", "state": text, "questions": questions})
    r.raise_for_status()
    body = r.json()
    return body["answers"], body["usage"]["cost"]


t0 = time.time()
answers, cost = decide(paragraph, {
    "dropped": {"type": "noul", "instructions": "Does this paragraph say the company has dropped, narrowed, delayed or stopped a climate goal or claim it previously made?"},
})
print(answers, f"({time.time() - t0:.2f} s, ${cost:.6f})")

# %% [markdown]
# Run the same question on a paragraph that restates a goal without changing it. A good question
# gives a low probability here.

# %%
restated = paras[paras.text.str.contains("most ambitious goal yet: Reach net zero", regex=False)].text.iloc[0]
print(restated[:300])
print(decide(restated, {"dropped": {"type": "noul", "instructions": "Does this paragraph say the company has dropped, narrowed, delayed or stopped a climate goal or claim it previously made?"}}))
