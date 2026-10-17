# Module 3: Text as Data

## Team members



## Learning objectives

- Turn a collection of documents into data: from PDFs to a table of paragraphs
- Design text codes: yes/no questions, categories and scales, and how their wording changes the answer
- Use a decision model and a language model from code, and choose between them
- Summarize coded text in figures and tables
- Check a model's answers against your own reading and against independent sources
- Describe, precisely and in your own words, what you want a coding agent to do

## Content

[Notebook](tech-pledges.ipynb)  
[Rubric](rubric.md)

Between 2019 and 2021, Google, Microsoft, Amazon, Meta and Apple made some of the most ambitious corporate
climate pledges ever announced. Since then the energy demands of AI have grown very quickly. In this module
we read 37 of the companies' own annual environmental reports with code, and ask whether they are backing
away from those pledges.

The notebook gives the background, the data, and working examples of the two kinds of model we use. It does
not give step-by-step instructions. Deciding what the question means in practice, writing down the codes
you will apply to the text, and judging when a model is wrong is the work of this module.

## What to hand in

- `tech-pledges.ipynb`, run from top to bottom.
- `prompts/`: your conversations with your coding agent, exported as files.
- This README, with your team members and a short description of what you found.

## Data

- `reports.csv` lists the 37 reports, the year each describes (`data_year`), and their URLs.
- `bash download_data.sh` downloads the reports and the validation data into `data/` (about 500 MB).
- Validation: the [Science Based Targets initiative](https://sciencebasedtargets.org/target-dashboard)
  company export, and NewClimate Institute's *Corporate Climate Responsibility Monitor 2025* (tech sector).

## Models

Both models are reached through [OpenRouter](https://openrouter.ai), with the same key you use for your
coding agent.

- [Jev](https://openrouter.ai/docs/guides/community/jev) (`~typesafe/jev-latest`), a decision model that
  answers typed questions with probabilities.
- DeepSeek V4 Flash (`deepseek/deepseek-v4-flash`), a language model, for structured extraction.

## Setup

```bash
bash download_data.sh
uv sync
export OPENROUTER_API_KEY="your-key-here"
```

The notebook reads `OPENROUTER_API_KEY` from the environment. Never put the key in the notebook or commit it.
