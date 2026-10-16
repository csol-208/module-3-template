# Session 10: A Ledger of Commitments

## The big idea

Last session you coded thousands of paragraphs and saw patterns: Google talking more about setbacks, "carbon neutral" disappearing, AI showing up as a cause. The codes show where to look. This session records what each company committed to and what happened to each commitment.

Today you'll build a **ledger**: one record per goal statement, with the goal, its target year, what the report says is happening to it, and a quote from the report as evidence. Linked across years, the records show which goals persist, which change and which disappear. Then you'll compare your ledger with two independent sources and ask why they differ.

This session uses a pattern that matters well beyond this module: **filter, then extract**. A cheap, fast decision model reads everything. A slower, more expensive language model reads only what the decision model picks out.

---

## Before you start

- `output/coded.parquet` (or whatever you saved) from Session 9.
- `OPENROUTER_API_KEY` set in your terminal.
- `data/validation/` from `download_data.sh`: the Science Based Targets initiative export (`sbti-companies.xlsx`) and NewClimate Institute's assessment of these five companies (`newclimate-ccrm2025-tech.pdf`).

---

## Task 1: Filter

**What you're doing:** Choosing which paragraphs are worth sending to a language model.

**Ask your coding agent to:**

From your Session 9 codes, keep paragraphs that state a goal *and* whose role is "announce", "setback" or "revise", or that Jev says drop a goal or make one harder. Show how many paragraphs this keeps per company and year.

**Checkpoint:** A few hundred paragraphs, out of the 11,000+ you started with.

**Think about:** What might this filter miss? Look at your Session 9 disagreements: would any of the paragraphs you labelled "yes" be dropped here?

---

## Task 2: Extract a record for every goal

**What you're doing:** Asking a language model to fill in a schema for each goal in each paragraph.

**Ask your coding agent to:**

Define a schema for one goal, with these fields:
- `goal`: the goal in a few words
- `goal_type`: one of net zero, carbon neutral, carbon negative, clean energy, emissions reduction, water, waste, other
- `target_year` and `baseline_year`: numbers, may be missing
- `scope`: what the goal covers, such as "Scope 1 and 2" or "value chain", may be missing
- `status`: one of new, reaffirmed, progress, harder, revised, dropped, **with a one-line definition of each** in the field description (for example, "dropped: says the company has stopped pursuing the goal or making the claim")
- `evidence`: an exact quote from the paragraph supporting the record

A paragraph can mention several goals, so the model should return a *list* of these. Tell the model, in its instructions, to include goals or claims the paragraph says were **dropped, changed or are becoming harder to reach**. Without that sentence, the model records Google's new goal on page 41 of its 2024 report but not the carbon neutrality claim Google gave up. Send each filtered paragraph to `deepseek/deepseek-v4-flash` through OpenRouter, 8 at a time, and save all the records as a table with the company, report, data year and page of each.

**Important details to mention:**
- Use `client.chat.completions.parse` with the schema, and `temperature=0`.
- Pass `extra_body={"reasoning": {"enabled": False}}`. This model can "think" before answering, which is slower and doesn't help here.
- Set `max_tokens=4000` and a timeout of 90 seconds on the client. Once in a while a model gets stuck repeating itself; when we tested this, two paragraphs produced 131,000-token answers and the run took 53 minutes instead of 3.
- Expect a few failures: the model sometimes returns a value that isn't in your list (we saw `"ongoing"` as a status). Retry, then skip.
- Save the results to a file and reuse it if it exists.

**Checkpoint:** Several hundred goal records, in a few minutes, for well under a dollar.

---

## Task 3: Check the evidence

**What you're doing:** Language models sometimes produce quotes that sound right but aren't in the text. A made-up quote is not evidence.

**Ask your coding agent to:**

For each record, check whether the `evidence` quote appears word for word in its paragraph. Compare only letters and digits, so that spacing, punctuation, curly quotes and PDF quirks (like "ﬁ" stored as one character) don't count as differences. For quotes that fail, also measure how much of the quote *does* appear in the paragraph, in runs of 20 characters or more. Classify each quote as **exact**, **edited** (most of it is there) or **not found**.

**Checkpoint:** About 9 in 10 quotes should be exact. Read every quote that isn't.

**Think about:** Look at the "not found" quotes. Some are the model *describing* the paragraph ("The paragraph states…") instead of quoting it. For the "edited" ones, was a citation dropped, a sentence trimmed, or two sentences stitched together? Which would be acceptable in a report you'd sign? Use only exact and edited records from here on.

---

## Task 4: The ledger

**What you're doing:** Linking records across years so you can follow each goal over time.

**Ask your coding agent to:**

1. Identify each goal by company, goal type and target year (for example "microsoft · carbon negative · 2030").
2. For each goal and data year, keep the most serious status that year's report gives it, in the order new < reaffirmed < progress < harder < revised < dropped. Use only records whose quote was exact or edited.
3. Save the ledger as `output/ledger.csv`.
4. Plot it as a timeline: one panel per company, one row per goal, one dot per year, coloured by status.
5. List goals that appear in at least two earlier reports but not in the company's latest one.

**Important detail:** many goals have no target year. Make sure those still get a key (such as "carbon neutral · no year") rather than being silently dropped. In the plot, include goals stated in at least two reports *plus* any goal ever marked revised or dropped (a dropped goal often appears only once).

**Checkpoint:** The ledger should include Google's carbon neutrality claim marked dropped (2023 data), Amazon's Shipment Zero marked dropped (2022), and Google goals marked revised in recent years. It will also contain mistakes. Find at least two records whose status you disagree with, and read the paragraph.

**Think about:** A goal that stops appearing may have been dropped, or the report may simply not repeat it. How would you tell the difference? Check one goal on your list in the company's latest report.

---

## Task 5: Compare with the Science Based Targets initiative

**What you're doing:** The Science Based Targets initiative (SBTi) checks whether companies' targets are consistent with climate science. It publishes each company's status, including commitments it has removed.

**Ask your coding agent to:**

Load `data/validation/sbti-companies.xlsx`, keep the five companies (Alphabet is Google's parent company), and show their near-term and net-zero status, target years and status reason. Then show what each company's *latest report* still says about net zero, carbon neutral and carbon negative goals, from your ledger.

**Checkpoint:** SBTi lists several of these commitments as "Commitment Removed", while the companies' latest reports still state the goals.

**Think about:** Is that a contradiction? What does SBTi's status measure, and what does your ledger measure? Note that the SBTi file is a snapshot of *today's* status only: it can't tell you when each change happened.

---

## Task 6: Read a whole report at once

**What you're doing:** NewClimate Institute assessed these same five companies in 2025. Their report is 34 pages, short enough to send to the language model in a single request, with no filtering or chunking.

**Ask your coding agent to:**

Extract the text of `data/validation/newclimate-ccrm2025-tech.pdf` and send all of it to the model in one request, with a schema that asks for NewClimate's main criticisms of each company's goals and claims: two to four per company, each with the kind of goal it concerns and the exact sentence from the report making the criticism. Check the quotes as in Task 3.

**Checkpoint:** About 20 criticisms. Most quotes should be exact or edited.

**A warning from testing:** NewClimate also gives each company ratings, but they're shown in graphic tables. When we asked the model for the ratings, it returned plausible ones backed by quotes that didn't state them. Ask only for what the text states, and check it.

**Think about:**
- How do NewClimate's criticisms compare with what your ledger shows for each company?
- When is it better to send a whole document at once, and when is the filter-then-extract pattern better? Think about length, cost, and how much you need to see where each answer came from.

---

## Task 7: Save and summarize

**Ask your coding agent to:**

Save the goal records as `output/goals.json` and the NewClimate assessments as a CSV. Then write, in your own words, a paragraph answering the module's question for one company of your choice: **is it backing away from its climate pledges?** Use your ledger, your Session 9 plots and the two independent sources, and cite paragraphs by report and page.

---

## Wrapping up: what you built this module

Across three sessions you went from 37 PDFs to:

1. **A paragraph dataset**, and an understanding of what PDF text extraction gets wrong.
2. **Word counts**, and why a mention isn't a claim.
3. **A codebook** for a decision model, with yes/no, category and scale questions, and evidence about which kinds of question work.
4. **Your own labels**, and a record of where you and the model disagreed.
5. **A ledger of commitments** with verified quotes, built by filtering with one model and extracting with another.
6. **A comparison** with independent sources, and an account of why they differ.

---

## For your assessment

Your Module 3 assessment asks you to answer a question about a body of climate text using the tools from this module, and to show how you know your answer is right. You can extend this module's question (more companies, more years, a different kind of commitment) or bring a new set of documents. Your submission should include:

- the documents and how you turned them into data,
- the questions you coded, and why you designed them as you did,
- at least one comparison between your own reading and a model's,
- figures or tables that answer your question,
- a check against at least one independent source, and
- outputs saved as CSV or JSON.
