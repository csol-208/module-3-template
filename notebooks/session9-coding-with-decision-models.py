# %% [markdown]
# # Session 9: Coding text with a decision model
#
# **Instructor answer key.** Students build this with their coding agent by following `session9.md`.
#
# Session 8 counted words. A word count cannot tell a pledge from a retreat: Google still says
# "net zero" in 2025, but now in a section that narrows what the goal covers. This session
# *codes* paragraphs, the way a social scientist codes interview transcripts: we define questions,
# a model answers them for every paragraph, and we analyse the answers as data.
#
# The design of the questions is the main subject. We move from yes/no codes, to categories, to
# ordered scales, and look at what each kind of code can and cannot show.
#
# Needs `output/paragraphs.parquet` from Session 8 and `OPENROUTER_API_KEY`.

# %%
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

matplotlib.use("Agg")
paras = pd.read_parquet("output/paragraphs.parquet")

JEV_URL = "https://openrouter.ai/api/v1/systemone"
JEV_HEADERS = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


def decide(text, questions):
    """Ask Jev typed questions about one text. Retries on transient errors."""
    for attempt in range(5):
        try:
            r = requests.post(JEV_URL, headers=JEV_HEADERS, timeout=120,
                              json={"model": "~typesafe/jev-latest", "state": text, "questions": questions})
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("Jev request failed five times")


def decide_all(texts, questions, workers=8):
    """Ask the same questions about many texts, eight at a time. Returns answers and total cost."""
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda t: decide(t, questions), texts))
    return [r["answers"] for r in results], sum(r["usage"]["cost"] for r in results)


def value(answer):
    """The single number or label from one answer: P(yes), the score, or the chosen option."""
    return answer.get("noul", answer.get("score", answer.get("choice")))

# %% [markdown]
# ## Part 1: The simplest code — does this paragraph state a goal?
#
# One yes/no (`noul`) question over every paragraph. Jev returns the probability of "yes".

# %%
GOAL = {"type": "noul", "instructions": "Does this paragraph state a climate or energy goal that the company has set for itself?"}

path = Path("output/goal-pass.parquet")
if not path.exists():
    t0 = time.time()
    answers, cost = decide_all(paras.text.tolist(), {"goal": GOAL})
    paras["p_goal"] = [a["goal"]["noul"] for a in answers]
    paras.to_parquet(path)
    print(f"{len(paras)} paragraphs in {time.time() - t0:.0f} s for ${cost:.2f}")
paras = pd.read_parquet(path)

ax = paras.p_goal.plot.hist(bins=40)
ax.set_xlabel("P(states a goal)")
plt.savefig("output/figures/s9-goal-histogram.png", dpi=120)
plt.close()
print((paras.p_goal > 0.5).mean().round(3), "of paragraphs pass 0.5")

# %% [markdown]
# Most paragraphs are near 0 or near 1, but not all. Choosing a threshold is a decision: read some
# paragraphs near 0.5 before choosing.

# %%
for r in paras[paras.p_goal.between(0.45, 0.55)].sample(4, random_state=1).itertuples():
    print(f"[{r.company} {r.report} p{r.page}  p={r.p_goal:.2f}]", r.text[:300], "\n")

# %% [markdown]
# ## Part 2: A richer codebook
#
# For paragraphs that state a goal (or mention one of the pledge terms), ask more questions. Each
# question type gives a different kind of variable:
#
# - **yes/no** (`noul`): a probability. Several narrow questions, each about one thing.
# - **choice**: one label from a list you define, plus a probability for every label.
# - **score**: a position on an ordered scale you describe, from 0 to the number of levels minus one.

# %%
CODEBOOK = {
    # yes/no
    "dropped": {"type": "noul", "instructions": "Does this paragraph say the company has dropped, narrowed, delayed or stopped a climate goal or claim it previously made?"},
    "harder": {"type": "noul", "instructions": "Does this paragraph say a climate goal has become harder to reach?"},
    "increase": {"type": "noul", "instructions": "Does this paragraph say the company's own total greenhouse gas emissions went up?"},
    "ai_cause": {"type": "noul", "instructions": "Does this paragraph say that AI is causing the company's energy use or emissions to grow?"},
    "offsets": {"type": "noul", "instructions": "Does this paragraph rely on carbon offsets, credits or carbon removal to meet a goal?"},
    # choice
    "role": {"type": "choice", "instructions": "What is this paragraph mainly doing with respect to the company's climate goals?",
             "criteria": {"announce": "announces or restates a goal or commitment",
                          "progress": "reports measured progress toward a goal, with numbers",
                          "setback": "admits emissions or energy use rose, or that a goal is harder to reach",
                          "revise": "changes, narrows, delays, reframes or drops a goal or claim",
                          "strategy": "describes actions, projects or investments to cut emissions",
                          "other": "background, methodology, or unrelated to climate goals"}},
    "goal_type": {"type": "choice", "instructions": "Which kind of climate goal is this paragraph mainly about?",
                  "criteria": {"net zero": "net-zero emissions",
                               "carbon neutral": "carbon neutrality or offsetting emissions",
                               "carbon negative": "removing more carbon than emitted",
                               "clean energy": "100% renewable or 24/7 carbon-free electricity",
                               "water": "water positive, replenishment or water use",
                               "waste": "zero waste, packaging, circularity",
                               "none": "no specific climate goal"}},
    # score
    "specificity": {"type": "score", "instructions": "How specific and measurable is the climate commitment in this paragraph?",
                    "criteria": ["vague aspiration with no numbers or dates",
                                 "a goal with a target year but no baseline or scope",
                                 "a quantified goal with target year and baseline or scope",
                                 "a quantified goal with target year, baseline, scope and interim milestones"]},
    "tone": {"type": "score", "instructions": "How confident does the company sound about reaching its climate goals?",
             "criteria": ["says the goal is at risk, unlikely or being abandoned",
                          "acknowledges serious challenges or uncertainty",
                          "neutral or matter of fact",
                          "confident and on track"]},
    # the same idea as `specificity`, asked as three separate yes/no questions
    "has_year": {"type": "noul", "instructions": "Does this paragraph give a year by which a climate goal should be reached?"},
    "has_baseline": {"type": "noul", "instructions": "Does this paragraph give a baseline year that a climate goal is measured against?"},
    "has_number": {"type": "noul", "instructions": "Does this paragraph state a numerical climate target, such as a percentage reduction?"},
}

pledge_terms = r"net[- ]zero|carbon[- ]neutral|carbon[- ]negative|24/7|water positive|zero waste|100% renewable|carbon[- ]free"
candidates = paras[(paras.p_goal > 0.5) | paras.text.str.contains(pledge_terms, case=False)].copy()
print(len(candidates), "candidate paragraphs")

path = Path("output/coded.parquet")
if not path.exists():
    t0 = time.time()
    answers, cost = decide_all(candidates.text.tolist(), CODEBOOK)
    for q in CODEBOOK:
        candidates[q] = [value(a[q]) for a in answers]
    candidates["raw_answers"] = [json.dumps(a) for a in answers]
    candidates.to_parquet(path)
    print(f"coded in {time.time() - t0:.0f} s for ${cost:.2f}")
coded = pd.read_parquet(path)
coded["goal"] = coded.p_goal > 0.5

# %% [markdown]
# ## Part 3: Yes/no codes, counted
#
# The simplest summary: how many paragraphs per report does Jev say "yes" to (probability > 0.5)?

# %%
binary = ["dropped", "harder", "increase", "ai_cause"]
counts = (coded.assign(**{q: coded[q] > 0.5 for q in binary})
          .groupby(["company", "data_year"])[binary].sum().reset_index()
          .melt(["company", "data_year"], var_name="question", value_name="paragraphs"))
g = sns.relplot(counts, x="data_year", y="paragraphs", hue="company", col="question", col_wrap=2,
                kind="line", marker="o", height=2.6, aspect=1.5)
g.savefig("output/figures/s9-binary-counts.png", dpi=120)
counts.pivot_table(index=["question", "company"], columns="data_year", values="paragraphs")

# %% [markdown]
# ## Part 4: Categories — what goals do the companies talk about?
#
# A choice question puts every goal paragraph in exactly one category. The share of each category
# per report shows how a company's emphasis shifts.

# %%
mix = (coded[coded.goal].groupby(["company", "data_year"]).goal_type.value_counts(normalize=True)
       .rename("share").reset_index())
g = sns.relplot(mix[mix.goal_type.isin(["net zero", "carbon neutral", "carbon negative", "clean energy"])],
                x="data_year", y="share", hue="goal_type", col="company", col_wrap=3,
                kind="line", marker="o", height=2.6, aspect=1.2)
g.savefig("output/figures/s9-goal-mix.png", dpi=120)

# %% [markdown]
# Roles are a second categorical variable. A stacked bar per report shows the balance between
# announcing, reporting progress, admitting setbacks and revising goals.

# %%
roles = pd.crosstab([coded.company, coded.data_year], coded.role, normalize="index")
fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
for ax, (company, df) in zip(axes, roles.groupby(level="company")):
    df.droplevel("company")[["announce", "progress", "setback", "revise", "strategy", "other"]].plot.bar(
        stacked=True, ax=ax, legend=ax is axes[-1], width=0.85, title=company)
axes[-1].legend(bbox_to_anchor=(1, 1), fontsize=8)
plt.tight_layout()
plt.savefig("output/figures/s9-roles.png", dpi=120)
plt.close()

# %% [markdown]
# ## Part 5: Ordered scales, and how to ask them
#
# `specificity` asks for a judgement on a four-level scale. The same idea, asked as three narrow
# yes/no questions (`has_year`, `has_baseline`, `has_number`) and added up, gives a 0–3 index.
# Do the two agree? Where they disagree, read the paragraph.

# %%
goals = coded[coded.goal].copy()
goals["spec_index"] = (goals[["has_year", "has_baseline", "has_number"]] > 0.5).sum(axis=1)
print("rank correlation:", goals.specificity.rank().corr(goals.spec_index.rank()).round(2))
print(pd.crosstab(goals.specificity.round(), goals.spec_index))

for r in goals[(goals.specificity > 1.5) & (goals.spec_index == 0)].sample(3, random_state=0).itertuples():
    print(f"\n[{r.company} {r.report} p{r.page} score={r.specificity:.2f} index={r.spec_index}]", r.text[:350])

# %% [markdown]
# Score answers are probability-weighted averages over the levels. Looking at the full distribution
# matters: a mean of 1.5 can come from "half 0, half 3" or "all 1.5".

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
sns.boxplot(goals, x="company", y="specificity", ax=axes[0])
sns.boxplot(goals, x="company", y="tone", ax=axes[1])
plt.tight_layout()
plt.savefig("output/figures/s9-scores.png", dpi=120)
plt.close()

tone = goals.groupby(["company", "data_year"]).tone.mean().unstack("company").round(2)
tone

# %% [markdown]
# ## Part 6: One question, three summaries
#
# "How much does a report talk about setbacks?" can be measured as
#
# 1. the number of paragraphs with P(harder) > 0.5,
# 2. the sum of P(harder) over all paragraphs (an expected count, no threshold), or
# 3. the share of goal paragraphs with P(harder) > 0.5 (adjusts for report length).

# %%
summ = coded.groupby(["company", "data_year"]).agg(
    count=("harder", lambda p: (p > 0.5).sum()),
    expected=("harder", "sum"),
    share=("harder", lambda p: (p[coded.loc[p.index, "goal"]] > 0.5).mean()),
).reset_index()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, col in zip(axes, ["count", "expected", "share"]):
    sns.lineplot(summ, x="data_year", y=col, hue="company", marker="o", ax=ax, legend=ax is axes[0])
    ax.set_title(col)
plt.tight_layout()
plt.savefig("output/figures/s9-three-summaries.png", dpi=120)
plt.close()
summ.pivot_table(index="company", columns="data_year", values=["count", "expected"]).round(1)

# %% [markdown]
# The expected count rises with report length even when nothing is said about setbacks, because
# hundreds of small probabilities add up. Thresholded counts ignore that noise but throw away
# information. Which one to plot depends on the question.

# %% [markdown]
# ## Part 7: Checking the codes against a careful reading
#
# Students label 20–30 paragraphs themselves before looking at Jev's answers, then compare. The
# key uses `notebooks/key-labels.csv`: a pinned sample of 23 paragraphs (11 that document known
# events, 12 random goal paragraphs) labelled while the key was written. These labels were made by
# Claude, not by the instructor: check them before relying on them.

# %%
labels = pd.read_csv("notebooks/key-labels.csv")
coded["start"] = coded.text.str[:60]
merged = labels.merge(coded, on=["company", "report", "page", "start"], how="inner")
# Jev's answers vary slightly between runs, so a labelled paragraph near the 0.5 cut-off may not
# be among this run's candidates. Report any that are missing.
print(len(merged), "of", len(labels), "labelled paragraphs were coded in this run")

for q in ["dropped", "harder", "increase", "ai_cause"]:
    jev_yes = merged[q] > 0.5
    print(f"{q:9s} agree {(jev_yes == merged['label_' + q].astype(bool)).mean():.0%}  "
          f"jev yes {jev_yes.sum():2d}  label yes {merged['label_' + q].sum():2d}")

# Agreement is inflated by the many easy "no" paragraphs. Look at the "yes" cases separately:
# how many of the paragraphs you labelled "yes" did Jev also say "yes" to, and vice versa?
disagree = merged[(merged.dropped > 0.5) != merged.label_dropped.astype(bool)]
for r in disagree.itertuples():
    print(f"\n[{r.company} {r.report} p{r.page}] P(dropped)={r.dropped:.2f}  label={r.label_dropped}  note: {r.note}")
    print("  ", r.text[:400])

# %% [markdown]
# ## Part 8 (optional): An open-weights decision model
#
# [Laya](https://huggingface.co/convaiinnovations/laya) accepts the same request format as Jev. If
# the course Laya server is configured (`LAYA_URL`, `LAYA_API_KEY`), send it the labelled sample
# and compare.

# %%
if os.environ.get("LAYA_URL"):
    laya_headers = {"Authorization": f"Bearer {os.environ['LAYA_API_KEY']}"}
    qs = {q: CODEBOOK[q] for q in ["dropped", "harder", "increase", "ai_cause", "role"]}
    r = requests.post(f"{os.environ['LAYA_URL']}/v1/systemone/batch", headers=laya_headers, timeout=600,
                      json={"states": merged.text.tolist(), "questions": qs})
    r.raise_for_status()
    laya = [res["answers"] for res in r.json()["results"]]
    for q in ["dropped", "harder", "increase", "ai_cause"]:
        merged["laya_" + q] = [a[q]["noul"] for a in laya]
        agree = ((merged["laya_" + q] > 0.5) == merged["label_" + q].astype(bool)).mean()
        print(f"{q:9s} Laya agrees with labels {agree:.0%}")
    merged["laya_role"] = [a["role"]["choice"] for a in laya]
    print(merged[["company", "report", "page", "role", "laya_role"]].to_string())
