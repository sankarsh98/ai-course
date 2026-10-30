# Writing guide for course modules

You are writing one or more modules of a multi-level AI course. Read this whole file and `SYLLABUS.md` before you start. Your module assignment names the module IDs you own.

## Audience and goal

A motivated self-learner who writes Python and can read code, has an engineering background, and wants real understanding, not hype. They learn by running things. Each module should leave them able to **explain the idea, implement the core of it, and judge claims about it**.

## Deliverables per module

For module `L2-05-transformers` you produce:

1. `docs/<level-dir>/<NN>-<slug>.md`, the lesson page (see "Page structure").
2. `code/<module-id>/*.py`, the runnable scripts that the page uses.

Nothing else. Do not edit `mkdocs.yml`, `SYLLABUS.md`, index pages, or other people's modules.

## Length and depth

- **3,500 to 6,000 words of prose and code per module.** Depth beats breadth. Derive things, show the math where it earns its place, and explain intuition before formalism.
- Foundations modules can lean on intuition and worked numeric examples. Level 3 and 4 modules should reach the level of a good graduate course introduction, with the real design decisions and trade-offs.
- Do not pad. Cut anything the reader wouldn't miss.

## Page structure (use these headings, in this order)

```
# <Module title>

> **Level X · Module N** · Prerequisites: <module IDs or "none"> · Time: <hours> · Code: `code/<module-id>/`

## What you will be able to do
<4-6 outcomes, each starting with a verb>

## 1. <Concept sections, as many as needed>
...

## Hands-on
<scripts the reader runs, each shown in full, followed by the REAL output you got running it>

## Common mistakes
<what goes wrong in practice and how to notice>

## Exercises
<8-12 exercises ordered easy to hard. Mark each [warm-up], [core], or [stretch].
 After the list, a "Hints and answers" section with short answers or pointers for every exercise.>

## Check your understanding
<5 short questions with one-paragraph answers>

## Further reading
<annotated, see "Sources">
```

## Sources: the most important rule

**Never cite anything you have not verified in this session.** Your memory of papers, authors, years, and URLs is not good enough. For every source:

1. Find it with WebSearch, then confirm with WebFetch (or a search result that shows the title, authors, and year on an authoritative page such as arxiv.org, a publisher, a conference site, or the author's own page).
2. Record exactly what you verified: title, authors, year, venue or arXiv ID, URL.
3. If you can't verify it, **leave it out**. A shorter, true list beats a longer one with a single error.

Format each entry in "Further reading" like this:

```
- **Author(s), "Title"** (Year, Venue/arXiv:ID). <URL>
  *Read for:* one sentence on what this source gives the reader. *Level:* intro / intermediate / advanced.
```

Prefer, in this order: primary sources (the original paper), well-regarded textbooks with free official versions, official documentation, and then high-quality tutorials. Include **6 to 12 sources per module**, with at least two that a beginner can read. Include free/open-access versions where they exist, and say when something is paywalled.

Rules for claims in the text:

- **Facts that come from a source** must be traceable: name the source in the sentence or put a bracketed short cite like `[Vaswani et al., 2017]` that matches a "Further reading" entry.
- **Numbers** (benchmark scores, parameter counts, dates, prices, compute) must come from a verified source. If you can't verify a number, don't state it, or say clearly that it's approximate and why.
- **Time-sensitive claims** (current models, laws, state of the art) must say "as of <month year>" and be verified by search. The current date is October 2026. Don't rely on training knowledge for anything recent.
- **Be honest about uncertainty and disagreement.** Where experts disagree, say so and give both views. Where evidence is weak, say that.
- Do not copy text from sources. Paraphrase. Quote at most a short phrase, with attribution.

## Code rules

- Python 3.12. Use the shared environment at `/home/sankarsh/projects/ai-course/.venv` (`/home/sankarsh/projects/ai-course/.venv/bin/python`). Installed: numpy, scipy, scikit-learn, matplotlib, pandas, and torch (CPU) if its install has finished. Check with `python -c "import torch"`. If you need another package, install it with `uv pip install --python /home/sankarsh/projects/ai-course/.venv/bin/python <pkg>` and say so in your report.
- No GPU, no API keys, no large downloads. Keep every script under about 60 seconds on a laptop CPU and under about 2 GB of RAM. Use small or synthetic data, or datasets bundled with scikit-learn or PyTorch-free sources. If a real dataset would add a lot and is small and openly licensed, you may use it, but say where it comes from and prove it downloads.
- **Run every script.** Paste the actual output into the page, not output you expect. If a result is surprising, investigate rather than hide it, and explain what happened.
- Set random seeds so results reproduce. If numbers vary with the platform, say so.
- Save plots only if they add real understanding. If you make one, save it to `docs/assets/<module-id>/` and reference it with a relative Markdown image link.
- Code should be clear and commented where the *why* is not obvious. Prefer short functions. The page contains the full code of each script (copy it, don't abbreviate), and the file in `code/` must be identical.

## Writing style

- Plain, direct language. Define a term the first time you use it. Avoid hype words and filler.
- Explain the intuition first, then the math, then the code, then what to watch for in practice.
- Use concrete numbers and small worked examples before general statements.
- Math: use `$...$` and `$$...$$` (standard LaTeX). Keep notation consistent within a module, and define every symbol.
- **Portability:** use only plain Markdown (headings, lists, tables, fenced code, images, blockquotes, math). **No** MkDocs admonitions (`!!!`), no HTML components, no tabs, no site-specific syntax. Use blockquotes for callouts: `> **Note:** ...` or `> **Warning:** ...`.
- Link to other modules by module ID in plain text (for example, "see L2-05"), not by file path.
- YAML front matter: none. The first line of the file is the `# Title` heading.
- Don't use the em dash character. Use commas, colons, or parentheses.

## Honesty about limits

- If something in the module can't be verified or you couldn't run it, say so in the page and in your report.
- Don't claim a result you didn't see. Don't describe behavior of a tool you didn't test.

## Your final report (returned to the coordinator)

Keep it under 300 words. Include:

- Files written (paths) and approximate word counts.
- How many sources you verified and how many you dropped as unverifiable.
- Any code that failed or behaved unexpectedly, and what you did.
- Any claims in the text that you are less than fully sure about.
- Any extra packages you installed.
- Suggested cross-links to other modules.
