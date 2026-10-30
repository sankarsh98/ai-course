# Capstone projects

Five projects that combine modules. Each one is open-ended on purpose: the module code gives you the building blocks, and the project is to put them together on data you care about and to **evaluate the result honestly**. Aim for a short written report with every project, covering what you built, how you measured it, and what failed.

A good report states the question, shows a baseline, reports uncertainty (not just a point estimate), and says what you would try next.

---

## Capstone 1: A tabular predictor with honest evaluation

**Uses:** Levels 0 and 1 (especially L1-01, L1-03, L1-05).

**Task.** Pick a tabular dataset you care about (public or your own) and build a predictor you can defend.

**Steps**
1. State the prediction problem and what a wrong prediction costs.
2. Build trivial baselines first (majority class or mean, a simple linear model).
3. Split the data so that leakage can't happen (time order, groups, duplicates), and write down why.
4. Compare a regularized linear model with a tree ensemble. Tune with cross-validation, touching the test set once.
5. Report metrics with confidence intervals (bootstrap), calibration, and results by subgroup.
6. Check for dataset shift between your training and test periods or sources.

**Deliverables.** Code, a one-page report, and a table of baseline versus model with uncertainty.

**You are done when** a skeptical reader could not find leakage and your model beats the baseline by more than the noise in your estimate.

---

## Capstone 2: A GPT from scratch on your own text

**Uses:** Level 2 (L2-01, L2-02, L2-05), plus L3-01 for tokenization.

**Task.** Train a small GPT on a corpus you choose (your own writing, public-domain books, code), starting from `code/L2-05-transformers/gpt.py` and `train_gpt.py`.

**Steps**
1. Build the corpus and a held-out validation split. Decide the tokenization (characters, or the BPE from L3-01).
2. Train the baseline and record validation loss in bits per character or token.
3. Run three controlled experiments (for example: context length, model width, layer-norm placement, learning-rate schedule). Change one thing at a time and use several seeds.
4. Generate samples at several temperatures and write down what the model has and has not learned.
5. Use `code/L4-02-efficient-training-and-inference/calculator.py` to predict the model's memory and compute, and compare with what you measure.

**Deliverables.** Training code, loss curves, the experiment table with seeds, and sample outputs with commentary.

**You are done when** you can explain each experiment's result and say which ones are within seed noise.

---

## Capstone 3: RAG over your own notes, with an evaluation set

**Uses:** L3-06, L4-03, L2-06.

**Task.** Build retrieval-augmented question answering over a set of your documents, and measure it. Start from `code/L3-06-retrieval-and-rag/rag_lib.py` and the evaluation harness in `code/L4-03-evaluation-and-benchmarks/evalkit.py`.

**Steps**
1. Collect 10 to 50 documents and write at least 40 questions with known answers, including some that the documents do not answer.
2. Build BM25 and a dense retriever; compare them and a hybrid on recall at k.
3. Experiment with chunk size and overlap, and add a reranker if it helps.
4. Measure answer quality with a fixed rubric, and check the judge (human or model) against your own labels on a sample.
5. Test failure modes: stale documents, near-duplicates, unanswerable questions, and a document containing instructions (prompt injection).

**Deliverables.** The pipeline, the question set, a table of configurations with confidence intervals, and a list of failures you found.

**You are done when** your numbers separate real differences from noise, and your eval set could catch a regression.

---

## Capstone 4: Reproduce a small paper result

**Uses:** L4-06, plus whichever modules match the paper.

**Task.** Choose a short, well-scoped paper whose main result can be reproduced on a CPU or small dataset, and try to reproduce one claim.

**Steps**
1. Use the reading and reproduction checklist in L4-06. Pick a paper with code or enough detail, and say in advance what result would count as reproduced.
2. Implement it, logging every run with `code/L4-06-research-skills-and-careers/tracker.py`.
3. Compare to the paper's numbers with several seeds, and say how far apart is "close".
4. Run at least one ablation the paper did not, and report it.
5. Write what was unclear in the paper, and what you had to guess.

**Deliverables.** Code, a run log, and a two-page report that is clear about what did and didn't reproduce.

**You are done when** a reader could rerun your code and get your numbers, and your report separates what you confirmed from what you assumed.

---

## Capstone 5: An agent with guardrails and an eval harness

**Uses:** L3-07, L4-03, L4-05, and the "AI Systems" course (https://sankarsh98.github.io/ai-systems/).

**Task.** Build a small tool-using agent for a narrow task, and an evaluation harness that decides whether it is good enough to trust. The agent can be rule-based while you develop the harness, and a real model once you have API access.

**Steps**
1. Define 20 or more tasks with checkable outcomes, including adversarial ones. Study `code/L3-07-agents-and-tool-use/shop.py` and `01_eval_harness.py` for a template.
2. Give the agent a minimal tool set and permission rules (what runs automatically, what needs a human).
3. Measure success over repeated runs, and report reliability (success in all of k runs) as well as average success.
4. Add attacks: tool output that contains instructions, malformed inputs, and runaway loops. Measure how often they succeed.
5. Add guardrails and show, with numbers and uncertainty (see `code/L4-05-safety-alignment-and-governance/05_safety_eval_statistics.py`), whether they helped and what they cost.
6. Write a go/no-go recommendation for deploying it, with the evidence.

**Deliverables.** The agent, the harness, results with confidence intervals, and the recommendation.

**You are done when** your recommendation would survive a skeptical reviewer.
