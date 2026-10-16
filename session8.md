# Session 8: From Reports to Paragraphs

## The big idea

Between 2019 and 2021, Google, Microsoft, Amazon, Meta and Apple each made headline climate pledges. Google said it had been carbon neutral since 2007 and would run on carbon-free energy around the clock by 2030. Microsoft promised to be carbon negative by 2030. Amazon and Meta promised net zero. Since then, the energy demands of AI have grown very quickly.

**Are these companies backing away from their pledges?** In this module you'll answer that from the companies' own words: 37 annual environmental reports, about 3,200 pages in all.

Nobody reads 3,200 pages to answer one question. Today you'll turn the reports into a table with one row per paragraph, see what simple word counts can and cannot tell you, and make your first requests to two kinds of AI model from code.

You won't write the code yourself — your coding agent will. Your job is to tell it what you want, check that it's doing the right thing, and think critically about the results.

---

## Before you start

1. Open the `module-3-template` folder in VS Code.
2. In the terminal, run `bash download_data.sh`. This downloads the reports (about 500 MB) into `data/reports/`. The list of reports, and the year each one covers, is in `reports.csv`.
3. Run `uv sync` to install the Python packages.
4. Set your OpenRouter key in the terminal. This is the same key you use for your coding agent:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

(This is an "environment variable": a setting your code can read without the key ever being written into a file. Never paste the key into your code.)

**A note on years.** Companies name their reports differently. Google's "2024 Environmental Report" describes 2023. Microsoft skipped a "2023" report. `reports.csv` has a `data_year` column with the year each report actually describes. Use it for any plot over time.

---

## Task 1: What the computer sees in a PDF

**What you're doing:** Google's 2024 report says, on page 41, that the company is "no longer maintaining operational carbon neutrality." You'll find that sentence the way a program would.

**Ask your coding agent to:**

Open `data/reports/google/google-2024.pdf` with the PyMuPDF library, print the text of page 41, and then check whether the exact phrase "no longer maintaining operational carbon neutrality" appears in that text.

**Checkpoint:** You should see the sentence on the page, but the search should say **it isn't there**. Ask your agent why. (The PDF breaks lines in the middle of the sentence. Collapsing line breaks and extra spaces into single spaces fixes it.)

**Think about:** If an exact search fails on a sentence you can see with your own eyes, what else might a simple search miss?

---

## Task 2: One row per paragraph

**What you're doing:** Building the dataset for the whole module: a table where each row is one paragraph, with columns for company, report, data year, page number and text.

**Ask your coding agent to:**

Loop over every report listed in `reports.csv`, split each page into paragraphs, and save all paragraphs as `output/paragraphs.parquet`. Drop fragments shorter than 200 characters (headings, captions, page numbers). Then show how many paragraphs each report produced, as a table with companies as columns and years as rows.

**Important details to mention:**
- PyMuPDF can return a page's text as "blocks" with their position on the page. Most blocks are paragraphs.
- Some PDFs store *every line* as its own block. Ask your agent to merge a block into the one above it when they line up on the left and the gap between them is less than a line.

**Checkpoint:** About 11,000–12,000 paragraphs. Meta's reports should produce over a hundred paragraphs each. If Meta has only a handful, the line-merging step isn't working: show your agent page 9 of `meta-2023.pdf`.

**Think about:** Google's 2021 and 2022 reports produce far fewer paragraphs than other years. Open one. Why? What will that do to any count you make over time?

---

## Task 3: Word counts over time

**What you're doing:** The simplest kind of text analysis: counting words. You'll track how often each company uses the language of its pledges.

**Ask your coding agent to:**

For each company and data year, compute the share of paragraphs (per 100 paragraphs) that mention each of these terms: "carbon neutral", "net zero", "carbon negative", "24/7" (as in 24/7 carbon-free energy), "offset", and "AI". Plot one small panel per term with a line per company, and save the figure to `output/figures/`.

**Important detail:** Ask for counts *per 100 paragraphs*, not raw counts. Reports differ a lot in length.

**Checkpoint:** Google's "carbon neutral" line should fall to almost nothing in its later reports, and "AI" should climb for every company except Apple.

**Think about:** Print the Google paragraphs that still mention "carbon neutral" after 2022 and read them. Is the company *claiming* carbon neutrality, or *talking about* it? Microsoft mentions "AI" a lot in 2020, years before the AI boom. Why? (Read some of those paragraphs.) What can a word count not tell you?

---

## Task 4: A first request to a language model

**What you're doing:** Sending a paragraph to a large language model from code, rather than pasting it into a chat window.

A quick vocabulary note: an **API** is a way for code to talk to a service. **OpenRouter** is a service that gives one key access to hundreds of AI models. Your code uses the standard "OpenAI-compatible" interface, pointed at OpenRouter's address.

**Ask your coding agent to:**

Connect to OpenRouter with the OpenAI Python library (base URL `https://openrouter.ai/api/v1`, key from `OPENROUTER_API_KEY`) and ask the model `deepseek/deepseek-v4-flash` to summarize, in one sentence, what the Google page 41 paragraph says about Google's climate goals.

Then ask for **structured output** instead: define a schema with the fields `goal` (text), `target_year` (a number, may be missing) and `status` (one of: new, reaffirmed, progress, harder, revised, dropped), and get the model to fill it in for the same paragraph.

A **schema** is a template that names the fields you want and their types: the columns of your future table. Your agent will use a Python library called Pydantic to define it.

**Important details to mention:**
- Use `temperature=0` to make answers as consistent as possible.
- Use the `client.chat.completions.parse` method with the schema as the `response_format`.

**Checkpoint:** The one-sentence summary is free text. The structured version is a set of labelled fields, and `status` should be "revised" or "dropped".

**Think about:** Which version could you put in a spreadsheet? Which one tells you more?

---

## Task 5: A first request to a decision model

**What you're doing:** Meeting a different kind of model. A **decision model** never writes text. You give it a text and a typed question, and it gives back a *probability*. We'll use Jev, served through OpenRouter with the same key.

**Ask your coding agent to:**

Write a function `decide(text, questions)` that sends a request to `https://openrouter.ai/api/v1/systemone` with your OpenRouter key, the model `~typesafe/jev-latest`, the text as `state`, and a dictionary of questions. Ask Jev this yes/no question (Jev calls yes/no questions `noul`) about the Google paragraph:

> Does this paragraph say the company has dropped, narrowed, delayed or stopped a climate goal or claim it previously made?

Then ask the same question about Meta's 2020 paragraph announcing its net zero goal (search the table for "most ambitious goal yet").

Print the probabilities, the time each request took, and the cost (Jev reports it in `usage.cost`).

**Checkpoint:** A high probability for Google (around 0.9) and a low one for Meta (near 0). Each request should take well under a second and cost a tiny fraction of a cent.

**Think about:** The language model took a few seconds and gave you words to read. Jev took a fraction of a second and gave you a number. Next session you'll ask Jev questions about all 11,000+ paragraphs. Why would that be impractical with a language model? What do you give up by getting only a number?

---

## What you built today

- A dataset of every paragraph in 37 reports, with company, year and page.
- Word-count trends, and a sense of what word counts miss.
- A request to a language model, returning free text and then structured fields.
- A request to a decision model, returning a probability.

**Save your work.** Next session builds on `output/paragraphs.parquet`.
