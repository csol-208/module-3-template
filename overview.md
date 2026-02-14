# Module 3: Getting Real Data Out of Real Documents

## The Problem

Most of the world's climate information lives in PDFs. City climate action plans. Corporate sustainability reports. Federal energy studies. Regulatory filings. 

You've probably already used ChatGPT or Claude to ask questions about a document — upload a PDF, ask "what are Google's Scope 1 emissions?", get an answer. That works. But it doesn't scale, it's not reproducible, and you can't build on it.

What if you need to:

- Compare emissions targets across **50 cities** to find which have the most ambitious plans?
- Track whether Apple, Google, Amazon, and BP are actually making progress on their climate commitments — every quarter, automatically?
- Check whether the numbers a company puts in their sustainability report actually add up?
- Build a tool that anyone in your organization can run on the next batch of reports?

You can't do that by uploading PDFs to a chat window one at a time and copy-pasting answers into a spreadsheet.

## What You'll Learn

### Write code that reads documents for you

By the end of this module, you'll be able to point a Python script at a folder of PDFs and get back a clean CSV comparing every company's emissions, targets, and renewable energy claims. The same pipeline that takes you an hour per document will process a hundred documents overnight.

### Get structured data, not paragraphs

When you ask ChatGPT about a sustainability report, you get a paragraph. When you write code to do it, you get a number in a column. You define exactly what fields you want — Scope 1 emissions, target year, baseline year, renewable energy percentage — and the AI fills in the table. If a field isn't in the document, you get a null, not a hallucinated guess buried in prose.

### Know when the AI is wrong

This is the part most people skip. LLMs confidently produce numbers that aren't in the source document. Your pipeline will include validation: Do the Scope 1 + Scope 2 numbers add up to the reported total? Is Scope 3 really smaller than Scope 1? (It shouldn't be.) Did two different chunks of the same report give you contradictory numbers? You'll build checks that flag problems automatically, so you know which extracted data to trust and which to verify by hand.

### Build tools, not just run scripts

There's a difference between writing a one-off script and building something reusable. By the end of the module, you'll have built a **document analysis tool** — a function you can call on any new PDF that returns structured data in a consistent format. You'll understand how to wrap AI capabilities into tools that other people (or other AI agents) can use.

### Understand what's happening under the hood

You won't just use AI — you'll understand the mechanics well enough to debug it when it fails. Why does the same prompt give different results on different models? Why does breaking a document into chunks sometimes help and sometimes hurt? Why do some PDFs extract cleanly and others come out as garbage? These aren't academic questions — they determine whether your pipeline works or doesn't.

---

## Session Overview

### Session 8: From Chat to Code

You already know how to ask an AI a question. Now you'll learn to do it from Python — which means you can do it a thousand times, with the same question, and collect the answers in a spreadsheet.

You'll start with text passages about corporate climate commitments (Apple, Google, Amazon, BP) and extract structured data: company name, target year, what they committed to, which emission scopes are covered. You'll see how different AI models interpret the same text differently — and start to understand why "just trust the AI" isn't a strategy.

**By the end of this session you can**: call an AI model from Python, get structured JSON instead of prose, and process multiple sources in a loop.

### Session 9: Real PDFs, Real Problems

PDFs are messy. Headers and footers get mixed into the text. Tables become scrambled. Charts disappear entirely. This session is where you hit reality.

You'll load corporate sustainability reports (Google, Apple, Amazon, BP), split them into pieces the AI can handle, and extract emissions data, energy metrics, and climate targets. Then you'll aggregate results across the whole document and validate them — checking whether the extracted numbers are internally consistent and in plausible ranges.

**By the end of this session you can**: extract text from PDFs, build an extraction pipeline that processes multi-hundred-page documents, and validate AI-extracted data for reliability.

### Session 10: Can You Trust What the AI Extracted?

The hardest question in AI-assisted data extraction isn't "can I get the numbers out?" — it's "should I believe them?"

In this session you'll run the same extraction with two different AI models and see where they disagree. You'll run the same model twice and see where it gives different answers. You'll build a confidence scoring system that tells you which results to trust and which to double-check. Then you'll take everything you've built and package it into a reusable tool — a function that takes any PDF and any schema and returns structured, validated data.

You'll also test your tool on a new set of documents: city climate action plans from Oakland, Portland, Seattle, Austin, and Ann Arbor. These are fundamentally different from the corporate reports in Session 9 — different formats, different language, different challenges. If your tool works on both, you've built something genuinely useful.

**By the end of this session you can**: assess the reliability of AI-extracted data, score confidence in individual fields, compare extraction across models, and build reusable extraction tools.

---

## What This Gets You

After this module, you'll have the skills to take on projects like:

- **Policy comparison**: Download 20 city climate action plans and build a comparison table of targets, timelines, and strategies — in an afternoon, not a month.
- **Corporate accountability**: Track whether companies are meeting the commitments in their sustainability reports by extracting and comparing year-over-year data.
- **Literature synthesis**: Process a set of energy studies and identify where projections agree and disagree.
- **Automated monitoring**: Build a tool that processes new reports as they're published, extracts key metrics, and flags changes.

The goal isn't to replace careful reading — it's to make careful reading possible at scale. You read the 5 reports that matter. The AI reads the other 95 and tells you what's in them.
