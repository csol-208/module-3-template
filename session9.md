# Session 9: Coding Text with a Decision Model

## The big idea

Last session's word counts showed Google's "carbon neutral" disappearing. But a word count cannot tell a pledge from a retreat. Google still writes "net zero" in 2025, in a section that narrows what the goal covers.

Today you'll **code** the paragraphs, the way a social scientist codes interview transcripts: you write questions, a decision model answers them for every paragraph, and you analyse the answers as data.

The questions are yours to design, and the design is the main subject of this session. You'll move from yes/no codes, to categories, to ordered scales, and see what each kind of code can and cannot show. Along the way you'll read paragraphs yourself and decide where the model is wrong.

---

## Before you start

- `output/paragraphs.parquet` from Session 8.
- `OPENROUTER_API_KEY` set in your terminal.
- Your `decide()` function from Session 8.

**Cost and time.** Jev charges about 4 cents per million tokens of input (a token is roughly three quarters of a word). Coding the whole corpus as below cost about 30 cents when we tested it. It takes a few minutes if your agent sends several requests at once. Ask your agent to **save every result to a file** and to reuse the file if it exists, so you only pay once.

---

## Task 1: The simplest code — does this paragraph state a goal?

**What you're doing:** One yes/no question, asked of every paragraph.

**Ask your coding agent to:**

Ask Jev this yes/no (`noul`) question about every paragraph in `output/paragraphs.parquet`:

> Does this paragraph state a climate or energy goal that the company has set for itself?

Send 8 requests at a time (your agent will know how to do this with a "thread pool"). Store the probability in a new column, save the table, and print the total cost and time. Then plot a histogram of the probabilities.

**Checkpoint:** Most probabilities should be near 0 or near 1. Somewhere around 15% of paragraphs should be above 0.5.

**Think about:** You need a cut-off to turn a probability into yes or no. Ask your agent to show you a few paragraphs with probabilities between 0.45 and 0.55, and read them. Would you call them goals? Where would you put the cut-off, and why?

---

## Task 2: Design a codebook

**What you're doing:** A **codebook** is the list of questions you ask of every paragraph, with careful wording. Jev offers three types of question, and each gives a different kind of variable:

| Type | You define | Jev returns | Example |
|---|---|---|---|
| `noul` (yes/no) | a question | probability of "yes" | Does this paragraph say emissions went up? |
| `choice` | a question and a list of labels, each with a description | one label, plus a probability for every label | What kind of goal is this about: net zero, carbon neutral, clean energy, …? |
| `score` | a question and an ordered list of levels | a number from 0 to (levels − 1) | How confident does the company sound, from "abandoning the goal" to "on track"? |

**Ask your coding agent to:**

Ask Jev a set of questions about the paragraphs that state a goal (probability above your cut-off) or that mention a pledge term ("net zero", "carbon neutral", "carbon negative", "24/7", "100% renewable", "carbon-free", "water positive", "zero waste"). Start with these, and change or add your own:

- **yes/no:**
  - Does this paragraph say the company has dropped, narrowed, delayed or stopped a climate goal or claim it previously made?
  - Does it say a climate goal has become harder to reach?
  - Does it say the company's own total emissions went up?
  - Does it say AI is causing the company's energy use or emissions to grow?
  - Does it rely on carbon offsets, credits or carbon removal to meet a goal?
- **choice — role:** What is this paragraph mainly doing? *announce* a goal, report *progress* with numbers, admit a *setback*, *revise* a goal, describe *strategy*, or *other*.
- **choice — goal type:** Which kind of goal is it mainly about? net zero, carbon neutral, carbon negative, clean energy, water, waste, or none.
- **score — specificity:** How specific is the commitment? From "vague aspiration with no numbers or dates", through "target year only" and "target year and baseline or scope", to "target year, baseline, scope and interim milestones".
- **score — tone:** How confident does the company sound about reaching its goals? From "says the goal is at risk or being abandoned" to "confident and on track".
- **The same idea as specificity, asked differently:** three yes/no questions — does the paragraph give a target year? a baseline year? a numerical target?

All the questions go in one request per paragraph. Save the answers.

**Important details to mention:**
- A `choice` question needs `criteria`: a dictionary from each label to a short description.
- A `score` question needs `criteria`: a list of level descriptions, in order.
- The question names you choose (the dictionary keys) are not sent to the model. Only the instructions and criteria matter.

**Checkpoint:** About 2,000–2,500 paragraphs coded, one column per question.

**Think about:** Read your yes/no questions again. Is each one about exactly one thing? "Did emissions go up *and* is AI to blame?" is two questions, and Jev reads instructions literally.

---

## Task 3: Yes/no codes, counted

**Ask your coding agent to:**

For the four yes/no questions about dropping goals, goals getting harder, emissions rising and AI, count the paragraphs per company and data year where Jev says yes (probability above 0.5). Plot one panel per question, a line per company.

**Checkpoint:** Google should stand out after 2022 on "harder" and "AI". "Dropped" stays rare for everyone.

**Think about:** "Dropped" is rare. Is that because companies rarely drop goals, or because they rarely *say* so in one paragraph? Ask your agent to print every paragraph Jev says yes to on "dropped". Do you agree with each one?

---

## Task 4: Categories

**Ask your coding agent to:**

1. Among goal paragraphs, compute the share of each goal type per company and data year, and plot it (one panel per company, a line per goal type).
2. Make a stacked bar chart of the *role* categories for each company and year.

**Checkpoint:** Google's "carbon neutral" share should fall sharply. Look at Microsoft's "net zero" share too.

**Think about:** How is this different from Session 8's word count? A category forces each paragraph into exactly one box. When might that be wrong, for example a paragraph about both net zero and clean energy? Jev also returns the probability of *every* label. How could you use those instead of the single winning label?

---

## Task 5: Scales, and how to ask them

**What you're doing:** Comparing two ways of measuring the same idea: one four-level `score` question about specificity, and three narrow yes/no questions (target year? baseline? number?) added up into a 0–3 index.

**Ask your coding agent to:**

1. Make a table comparing the specificity score (rounded) against the 0–3 index.
2. Show a few paragraphs where the two disagree most.
3. Make box plots of specificity and of tone, by company.

**Checkpoint:** The two measures should broadly agree, but not perfectly.

**Think about:** Read the disagreements. Which measure was right? Which version would you trust in a report, and which would you rather explain to a reader? A `score` comes back as an average over the levels (for example 1.6). What does 1.6 mean on a scale whose levels are sentences?

---

## Task 6: One question, three summaries

**What you're doing:** Turning paragraph-level codes into one number per report forces another choice.

**Ask your coding agent to:**

For the "goal has become harder" question, compute three summaries per company and year, and plot them side by side:
1. the **count** of paragraphs with probability above 0.5,
2. the **expected count**: the sum of all the probabilities, with no cut-off,
3. the **share** of goal paragraphs above 0.5.

**Checkpoint:** The three plots tell somewhat different stories.

**Think about:** Long reports have hundreds of paragraphs with small probabilities, and the expected count adds them all up. Which summary would you put in a report, and how would you explain it in one sentence?

---

## Task 7: You versus the model

**What you're doing:** Checking the codes against your own careful reading. This is the most important task of the session.

**Ask your coding agent to:**

Draw a random sample of 20 goal paragraphs, and add these documented paragraphs to it (search the text for the quoted phrase):
- Google 2024: "no longer maintaining operational carbon neutrality"
- Google 2025: "adjusted the boundary"
- Google 2025: "now more complex and challenging"
- Microsoft 2024: "emissions are up 29.1%"
- Amazon 2022: "no longer made sense to have a separate"

Save them to a spreadsheet **without Jev's answers**, with empty columns for the four yes/no questions.

**Then label them yourself.** Read each paragraph and mark 1 or 0 for each question. Write a short note wherever you hesitated. Don't let your agent do this part. Your reading is what's being tested.

**Then ask your coding agent to** compare your labels with Jev's answers, and list every disagreement next to its paragraph.

**Think about:**
- For each disagreement, read the paragraph again. Who was right, you or Jev? Was the question itself unclear?
- Agreement is easy when most paragraphs are obviously "no". Of the paragraphs *you* said yes to, how many did Jev catch? Of the ones Jev said yes to, how many did you agree with?
- Rewrite one question to fix a problem you found, re-run it on your sample, and see whether it helps.

---

## Going further (optional)

- **An open model.** Laya is an open-weights decision model that accepts the same requests as Jev. If your instructor gives you a Laya address and key, send it your labelled sample and compare it with Jev and with you.
- **Your own question.** Design a question about something else you noticed: water, supply-chain emissions, carbon removal, or how companies describe their critics. Code the corpus with it and plot the result.

---

## What you built today

- Codes for every paragraph: yes/no, categories and scales.
- Plots of how each company's language changes over time.
- A comparison of two ways to measure the same idea.
- Your own labels, and a careful look at where the model disagrees with you.

Next session you'll turn the most important paragraphs into a ledger of each company's commitments, and compare it with independent assessments.
