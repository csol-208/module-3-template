# Text as Data Module: Detailed Rubric

Total: 20 points (of 28 available -- this allows multiple paths through the module)

No single tool or method is required. A notebook can earn full marks with yes/no codes and careful reading,
with categories and scales, with extracted records, or with any mix of them. You may automate any step that a
model or your coding agent can do well. You are graded on the decisions you make, how clearly you describe
them, how you check them, and how well you tell the story.

## 1. Defining the question and the corpus (**4 points**)

**4 points**: The notebook turns the broad question into one a computer can answer, and says how.

  - It says what counts as "backing away" from a pledge, with real examples from the reports, and what
    does not.
  - It states which reports and which paragraphs are included and why, and how you checked that the text
    extraction worked.
  - Where a different reasonable choice would change the answer, the notebook says so.

## 2. The codebook (**6 points**)

**6 points**: The notebook contains a codebook written in plain English, precise enough that a coding agent
(or a classmate) could apply it without asking you anything.

  - Each question has a name, a type (yes/no, choice or score), its exact wording, and its labels or levels
    with descriptions.
  - The notebook explains why each question has the type it has, and uses more than one type.
  - At least one idea is asked in two different ways, and the notebook compares the results.
  - The codebook shows how it changed: earlier versions are kept, with a note on what went wrong and how you
    fixed it.

## 3. Checking the models (**5 points**)

**5 points**: The student can answer the question _how do I know these results are right?_

  - Model outputs (codes, extracted records, quotes) are compared against the source text, not taken on
    trust.
  - You labelled a sample of paragraphs yourself before looking at the model's answers. The notebook
    documents specific disagreements, says who was right after reading again, and explains what each
    disagreement says about the question or the model.
  - At least one result is compared with an independent source, such as the Science Based Targets initiative
    or NewClimate Institute. Before any difference is explained, the notebook establishes what each source
    measures.
  - The notebook is honest about who made each judgment. Labels or text produced by a model are described as
    such, and your own reading is described as yours.

## 4. Results (**4 points**)

**4 points**: The notebook gives a clear answer to the question for each company, supported by well-labelled
figures or tables.

  - It says how paragraph-level codes were turned into one number per report (for example a count above a
    threshold, a sum of probabilities, or a share of paragraphs), and why.
  - The text says what each figure shows and how much confidence it deserves.
  - Claims about speed, cost or accuracy are supported by measurements from this work.

## 5. Narrative and context (**5 points**)

**5 points**: The notebook tells a clear story that a reader outside the class could follow.

  - It explains the stakes: what these companies pledged, why the growth of AI matters to those pledges, and
    why the way a company describes its goals matters.
  - Instructor headings are replaced with topical headings, and the text connects the code and figures to
    the argument.
  - It ends with a conclusion in your own words: what you found, how confident you are in it, and what you
    would do next.

## 6. Process and GitHub (**4 points**)

**4 points**:

  - The repository includes your conversations with your coding agent, exported into `prompts/`. These show
    the instructions you gave the agent, including your codebook, corrections, and times you redirected it.
  - The README is updated with team members and a short description of what you found.
  - The notebook runs from top to bottom without errors, and expensive results are saved and reused rather
    than recomputed.
  - Commits show how the work developed, with clear messages. No API keys are committed.

## Failure conditions

The notebook fabricates data, or presents made-up or "example" numbers, quotations or citations as if they
came from the sources. (Automatic fail)

This applies without change to output from a language model. Models sometimes produce plausible quotations
or numbers that are not in the documents. Passing those along as real is the same failure whether you typed
them or accepted them.

The notebook presents a model's judgment as your own: for example, labels or reflections written by a model
and described as your reading. Using a model for these is allowed. Misrepresenting who made the judgment is
not.
