# Module 3: Text as Data

## The question

Between 2019 and 2021, the largest tech companies made some of the most ambitious corporate climate pledges ever announced:

- **Google** said it had been carbon neutral since 2007, and would run every data center on carbon-free energy, every hour of every day, by 2030.
- **Microsoft** promised to be carbon negative by 2030: removing more carbon than it emits.
- **Amazon** co-founded The Climate Pledge: net zero by 2040.
- **Meta** promised net zero across its value chain by 2030.
- **Apple** promised a carbon neutral business and products by 2030.

Since then, the energy demands of AI have grown faster than almost anyone expected. Data centers need power, and building them needs steel and concrete.

**Are these companies backing away from their climate pledges?**

Journalists have reported pieces of the answer. In this module you'll build your own answer from the companies' own words: 37 annual environmental reports, about 3,200 pages, published between 2018 and 2026.

## Why this is a data problem

You could read one report carefully. You can't read 37 carefully, and you can't compare them consistently by hand. Changes are often quiet: a claim stops appearing, a boundary is redrawn in a footnote, "goal" becomes "moonshot".

So we treat the text as data. Every paragraph becomes a row in a table. Then we ask questions of every row and analyse the answers like any other dataset.

## What you'll learn

### Turn documents into data

You'll turn PDFs into a table of paragraphs and see what goes wrong along the way: sentences broken across lines, pages stored line by line, garbled text. You'll count words over time, and see why a word count can't tell a pledge from a retreat.

### Code text with a decision model

A **decision model** reads a text and answers a typed question with a probability, in a fraction of a second, for a fraction of a cent. You'll design the questions:

- **yes/no:** Does this paragraph say a goal has become harder to reach?
- **category:** Is this paragraph announcing a goal, reporting progress, admitting a setback or revising a goal?
- **scale:** How specific is this commitment, from vague aspiration to quantified target with milestones?

Each kind of question gives a different kind of data and a different kind of plot. How you word a question changes the answer, and finding out how is a large part of the module.

### Extract records with a language model

A **language model** can fill in a detailed form (a schema) for each goal: what it is, its target year, what's happening to it, and a quote as evidence. It's slower and more expensive than a decision model, so you'll use the decision model to decide which paragraphs the language model should read.

### Know when a model is wrong

Every tool here is sometimes wrong. You'll label paragraphs yourself and compare your reading with the model's. You'll check that quotes are real. And you'll compare your results with two independent sources: the Science Based Targets initiative and NewClimate Institute. When sources disagree, you'll work out what each one is actually measuring.

---

## Sessions

### Session 8: From Reports to Paragraphs

Turn 37 reports into one table of paragraphs. Track how often each company uses the language of its pledges, and find out what word counts miss. Make your first requests to a language model and to a decision model from code.

**By the end you can:** extract text from PDFs, build a paragraph dataset, and call two kinds of AI model from Python.

### Session 9: Coding Text with a Decision Model

Design a codebook of yes/no, category and scale questions, and ask a decision model to answer them for thousands of paragraphs. Plot how each company's language changes. Label a sample yourself and decide who's right when you and the model disagree.

**By the end you can:** design questions for text coding, explain the trade-offs between kinds of question, summarize coded text in figures, and measure a model against your own reading.

### Session 10: A Ledger of Commitments

Use the decision model to pick out the paragraphs that matter, and a language model to turn each into a structured record of a goal. Check every quote. Link the records across years into a ledger, and compare it with independent assessments.

**By the end you can:** build a filter-then-extract pipeline, verify extracted evidence, and reconcile your results with independent sources.

---

## Tools

- **Your coding agent** writes the code. You direct it and judge the results.
- **OpenRouter** gives one key access to both models. It's the same key you use for your coding agent.
- **Jev** (`~typesafe/jev-latest`), a decision model. Coding the whole corpus costs about 30 cents.
- **DeepSeek V4 Flash** (`deepseek/deepseek-v4-flash`), a language model, for structured extraction.

## After this module

The same approach works on any large body of text where the question is "what does it say, and how has that changed?" Examples include climate action plans from hundreds of cities, comment letters on a proposed regulation, utility filings, news coverage and court decisions. The tools are cheap enough that what limits you is how well you design your questions and how carefully you check the answers.
