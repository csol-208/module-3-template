# %% [markdown]
# # Session 10: A ledger of commitments
#
# **Instructor answer key.** Students build this with their coding agent by following `session10.md`.
#
# Session 9 coded paragraphs. This session turns the most important ones into records, one per
# goal statement: which goal, what target year, what the report says is happening to it, and a
# quote from the text as evidence. Linked across years, the records form a ledger of each
# company's commitments. We then compare the ledger with two independent sources.
#
# The pattern is *filter, then extract*: a cheap decision model reads everything, and a language
# model reads only what the decision model picks out.
#
# Needs `output/coded.parquet` from Session 9 and `OPENROUTER_API_KEY`.

# %%
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pymupdf
import seaborn as sns
from openai import OpenAI
from pydantic import BaseModel, Field

matplotlib.use("Agg")
# A timeout and a cap on output length: now and then a model starts repeating itself and would
# otherwise generate for many minutes (we saw 131,000-token answers to one paragraph).
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                timeout=90, max_retries=0)
MODEL = "deepseek/deepseek-v4-flash"
FAST = {"reasoning": {"enabled": False}, "usage": {"include": True}}

coded = pd.read_parquet("output/coded.parquet")

# %% [markdown]
# ## Part 1: Filter
#
# Keep paragraphs that state a goal and that Jev says announce, revise or report a setback on one,
# or that it says drop or make harder a goal.

# %%
statements = coded[(coded.p_goal > 0.5) & (
    coded.role.isin(["announce", "setback", "revise"]) | (coded.dropped > 0.5) | (coded.harder > 0.5)
)].reset_index(drop=True)
print(len(statements), "of", len(coded), "coded paragraphs")
statements.groupby(["company", "data_year"]).size().unstack("company")

# %% [markdown]
# ## Part 2: Extract
#
# One paragraph can mention several goals, so the schema is a list of goal records. The `evidence`
# field asks for a verbatim quote, which we can check.

# %%
class Goal(BaseModel):
    goal: str = Field(description="the goal in a few words, e.g. 'net zero across the value chain'")
    goal_type: Literal["net zero", "carbon neutral", "carbon negative", "clean energy", "emissions reduction",
                       "water", "waste", "other"]
    target_year: Optional[int] = Field(None, description="year the goal should be met, if stated")
    baseline_year: Optional[int] = Field(None, description="baseline year the goal is measured against, if stated")
    scope: Optional[str] = Field(None, description="what the goal covers, e.g. 'Scope 1 and 2', 'operations', 'value chain'")
    status: Literal["new", "reaffirmed", "progress", "harder", "revised", "dropped"] = Field(description=(
        "new: the paragraph announces the goal for the first time; "
        "reaffirmed: restates an existing goal without news; "
        "progress: reports measured progress toward the goal; "
        "harder: says the goal has become harder to reach or emissions are moving the wrong way; "
        "revised: changes, narrows, delays or reframes the goal; "
        "dropped: says the company has stopped pursuing the goal or making the claim"))
    evidence: str = Field(description="an exact quote from the paragraph, copied word for word, supporting this record")


class Goals(BaseModel):
    goals: list[Goal]


# Without the second sentence, the model records Google's replacement goal but not the carbon
# neutrality claim Google gave up.
SYSTEM = ("You extract climate goals from one paragraph of a company's environmental report. "
          "Return one record per distinct goal or claim the paragraph discusses, including goals or claims the "
          "paragraph says were dropped, changed or are becoming harder to reach. Use only the paragraph. "
          "If the paragraph contains no climate goal, return an empty list.")


def extract(text):
    for attempt in range(3):
        try:
            r = client.chat.completions.parse(
                model=MODEL, temperature=0, max_tokens=4000, response_format=Goals, extra_body=FAST,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
            return r.choices[0].message.parsed, getattr(r.usage, "cost", 0) or 0
        except Exception as e:
            err = e
            time.sleep(2 ** attempt)
    print("failed:", err)
    return None, 0


path = Path("output/goals.parquet")
if not path.exists():
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(extract, statements.text))
    rows = []
    for i, (parsed, _) in enumerate(results):
        for g in (parsed.goals if parsed else []):
            rows.append({"para": i, **g.model_dump()})
    goals = pd.DataFrame(rows).merge(statements[["company", "report", "data_year", "page", "text"]],
                                     left_on="para", right_index=True)
    goals.to_parquet(path)
    print(f"{len(goals)} goal records from {len(statements)} paragraphs in {time.time() - t0:.0f} s "
          f"for ${sum(c for _, c in results):.2f}; {sum(p is None for p, _ in results)} failures")
goals = pd.read_parquet(path)
goals.status.value_counts()

# %% [markdown]
# ## Part 3: Check the evidence
#
# A quote the model wrote itself is not evidence. Comparing only letters and digits avoids false
# alarms from spacing, punctuation, curly quotes and PDF ligatures ("ﬁ"). An exact test still can't
# tell a quote with a citation left out, or two sentences stitched together, from an invented one,
# so we also measure *coverage*: the share of the quote found in the source in runs of 20 or more
# characters.

# %%
def normalize(s):
    return re.sub(r"[^a-z0-9]", "", s.replace("ﬁ", "fi").replace("ﬂ", "fl").lower())


def coverage(quote, source):
    q, s = normalize(quote), normalize(source)
    i = covered = 0
    while i < len(q):
        lo, hi = 0, len(q) - i          # longest prefix of q[i:] that occurs in s
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if q[i:i + mid] in s else (lo, mid - 1)
        if lo >= 20:
            covered += lo
            i += lo
        else:
            i += 1
    return covered / max(len(q), 1)


def quote_check(quote, source):
    if normalize(quote) in normalize(source):
        return "exact"
    return "edited" if coverage(quote, source) >= 0.8 else "not found"


goals["quote_check"] = [quote_check(e, t) for e, t in zip(goals.evidence, goals.text)]
print(goals.quote_check.value_counts())
for r in goals[goals.quote_check != "exact"].head(6).itertuples():
    print(f"\n[{r.company} {r.report} p{r.page}] {r.quote_check}: {r.evidence!r}")

# %% [markdown]
# ## Part 4: The ledger
#
# A goal is identified by company, type and target year. For each goal and report, record the
# strongest change the report signals (dropped > revised > harder > progress > reaffirmed > new).

# %%
severity = ["new", "reaffirmed", "progress", "harder", "revised", "dropped"]
goals["status"] = pd.Categorical(goals.status, categories=severity, ordered=True)
goals["goal_key"] = goals.goal_type + " " + goals.target_year.map(lambda y: "(no year)" if pd.isna(y) else str(int(y)))

ledger = (goals[goals.quote_check != "not found"].groupby(["company", "goal_key", "data_year"], observed=True)
          .agg(status=("status", "max"), statements=("status", "size"), example=("evidence", "first"))
          .reset_index())
ledger.to_csv("output/ledger.csv", index=False)

# To keep the figure readable: goals stated in at least two reports, plus any goal ever marked
# revised or dropped (a dropped goal often appears only once, in the report that drops it)
by_goal = ledger.groupby(["company", "goal_key"])
keep = (by_goal.data_year.transform("nunique") >= 2) | by_goal.status.transform(lambda s: s.isin(["revised", "dropped"]).any())
plot = ledger[keep]
colors = dict(zip(severity, ["#bdbdbd", "#4daf4a", "#377eb8", "#ff7f00", "#984ea3", "#e41a1c"]))
companies = sorted(plot.company.unique())
fig, axes = plt.subplots(len(companies), 1, figsize=(9, 2.2 * len(companies)), sharex=True)
for ax, company in zip(axes, companies):
    df = plot[plot.company == company]
    keys = sorted(df.goal_key.unique())
    ax.scatter(df.data_year, df.goal_key.map({k: i for i, k in enumerate(keys)}),
               c=df.status.astype(str).map(colors), s=20 + 15 * df.statements, edgecolors="black", linewidths=0.3)
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.set_title(company, loc="left", fontsize=10)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in colors.items()]
fig.legend(handles=handles, loc="upper right", fontsize=8, title="status")
plt.tight_layout(rect=(0, 0, 0.88, 1))
plt.savefig("output/figures/s10-ledger.png", dpi=120)
plt.close()

# %% [markdown]
# Which goals stop appearing? For each company, list goals stated in an earlier report but not in
# the latest one. Absence is weak evidence on its own (a report may simply not repeat a goal), so
# read the last statement of each.

# %%
last_seen = ledger.groupby(["company", "goal_key"]).agg(last=("data_year", "max"), reports=("data_year", "nunique"),
                                                         last_status=("status", "last"))
last_seen["company_latest"] = ledger.groupby("company").data_year.max().reindex(last_seen.index.get_level_values(0)).values
gone = last_seen[(last_seen["last"] < last_seen.company_latest) & (last_seen.reports >= 2)]
gone

# %% [markdown]
# ## Part 5: Compare with the Science Based Targets initiative
#
# SBTi validates corporate targets and removes commitments that lapse. Its export is a snapshot of
# today's status only.

# %%
sbti = pd.read_excel("data/validation/sbti-companies.xlsx")
names = {"google": "Alphabet Inc.", "amazon": "Amazon", "apple": "Apple, Inc.",
         "meta": "Meta Platforms, Inc.", "microsoft": "Microsoft Corporation"}
sbti = sbti[sbti.company_name.isin(names.values())][
    ["company_name", "near_term_status", "near_term_target_year", "net_zero_status", "net_zero_year", "status_reason"]]
sbti

# %%
# What does each company's latest report still say about net zero / carbon negative?
latest_report = goals[goals.data_year == goals.groupby("company").data_year.transform("max")]
latest_report[latest_report.goal_type.isin(["net zero", "carbon negative", "carbon neutral"])][
    ["company", "data_year", "goal", "target_year", "status", "evidence"]]

# %% [markdown]
# Amazon, Meta and Microsoft still state net-zero or carbon-negative goals in their latest reports,
# while SBTi lists their net-zero (or, for Amazon, near-term) commitments as removed. These measure
# different things: SBTi records whether a target was submitted and validated on time, not whether
# the company still claims it.

# %% [markdown]
# ## Part 6: Whole-document extraction — the NewClimate Institute assessment
#
# NewClimate Institute's *Corporate Climate Responsibility Monitor 2025* assessed the same five
# companies. The tech chapter is 34 pages, short enough to send to the model in one request. This
# is a different pattern from Parts 1–2: no filtering, the model reads everything at once.

# %%
# An earlier version of this key asked for each company's "integrity rating". The report shows its
# ratings in graphic tables, and the model returned plausible ratings backed by quotes that did not
# state them. Ask only for what the text states, with a quote that can be checked.
class Finding(BaseModel):
    company: Literal["Amazon", "Apple", "Google", "Meta", "Microsoft"]
    goal_type: Literal["net zero", "carbon neutral", "carbon negative", "clean energy", "emissions reduction",
                       "water", "waste", "other"]
    criticism: str = Field(description="the criticism, briefly, in your own words")
    quote: str = Field(description="the exact sentence from the report making this criticism, copied word for word")


class Findings(BaseModel):
    findings: list[Finding]


report_text = "\n".join(page.get_text() for page in pymupdf.open("data/validation/newclimate-ccrm2025-tech.pdf"))
t0 = time.time()
r = client.chat.completions.parse(
    model=MODEL, temperature=0, max_tokens=8000, response_format=Findings, extra_body=FAST,
    messages=[{"role": "system", "content": "List NewClimate Institute's main criticisms of each of the five tech companies' climate "
                                            "goals and claims, two to four per company. Use only the report."},
              {"role": "user", "content": report_text}])
assessments = pd.DataFrame([f.model_dump() for f in r.choices[0].message.parsed.findings])
assessments["quote_check"] = [quote_check(q, report_text) for q in assessments.quote]
print(f"{len(report_text):,} characters, {time.time() - t0:.0f} s, ${getattr(r.usage, 'cost', 0) or 0:.3f}")
print(assessments.quote_check.value_counts())
assessments[["company", "goal_type", "criticism", "quote_check"]]

# %% [markdown]
# ## Part 7: Save

# %%
goals.drop(columns=["text"]).to_json("output/goals.json", orient="records", indent=1)
assessments.to_csv("output/newclimate-assessments.csv", index=False)
print("saved output/ledger.csv, output/goals.json, output/newclimate-assessments.csv")
