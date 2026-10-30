---
title: "Research Skills and Careers"
---

> **Level 4 · Module 6** · Prerequisites: L0-04, L1-01, L2-02, L4-03 (L4-01 helps for section 4) · Time: 6 to 8 hours, plus the capstone · Code: `code/L4-06-research-skills-and-careers/`

## What you will be able to do

- Read a machine learning paper in passes, extract its central claim and the evidence for it, and spot the most common weaknesses in that evidence.
- Plan and carry out a reproduction of a published result, and write down where and why your numbers differ.
- Design an ablation study with several seeds, paired comparisons and a held-out test set, and report it with honest uncertainty.
- Track experiments so that every number in a report can be traced to a configuration, a code version and an environment.
- Write a short research report, choose problems worth working on, and make a first useful open-source contribution.
- Describe the main kinds of AI jobs and what each one rewards, and find your way around the field's venues, textbooks and communities.

## 1. Reading a paper efficiently

Most papers you open you will not read to the end, and that is correct. Reading is a search problem: spend a little time on many papers and a lot of time on a few.

**Three passes.** Keshav's short note proposes reading in up to three passes [Keshav, 2007]. The first pass, five to ten minutes long, covers the title, abstract, introduction, section headings and conclusion, and lets you decide whether to continue; he suggests you should then be able to say what kind of paper it is, what it builds on, whether its assumptions look valid, what it contributes and whether it is clearly written. The second pass, up to about an hour, reads the paper with care but skips proofs, and pays attention to figures and tables. The third pass, for papers you need to master, amounts to virtually re-implementing the work, challenging every assumption; he estimates four or five hours for a beginner.

For machine learning papers, adapt the passes:

1. **Find the claim.** Write one sentence of the form "method M improves metric Y on task T compared with baseline B, under conditions C". If you cannot, the paper may not have one claim, which is itself information.
2. **Find the evidence for exactly that claim.** Usually one table or figure. Check what is compared with what: same data, same compute budget, same tuning effort? How many seeds, and is there any measure of spread? Is the test set the one everyone uses, and could it have leaked into training (L4-03)?
3. **Find the ablation.** Which part of the method produces the gain? If the paper changes five things at once and ablates none, you do not know which matter.
4. **Read the limitations and the appendix.** Hyperparameter tables, compute budgets and data-filtering details often decide whether you can reproduce anything.
5. **Write a five-line note** with the claim, the evidence, your doubts, what you would try next, and the citation. Notes are what you will have in six months.

**Red flags.** Lipton and Steinhardt describe four troubling patterns in ML writing: not separating explanation from speculation, not identifying where empirical gains come from, "mathiness" (math that decorates rather than clarifies), and misuse of language such as suggestive names and overloaded terms [Lipton and Steinhardt, 2018]. In practice also watch for: baselines copied from older papers instead of rerun, a single run per configuration, gains smaller than the seed-to-seed spread, results only on benchmarks the authors created, and missing compute numbers.

**Tuning matters as much as ideas.** Melis, Dyer and Blunsom re-evaluated several recurrent language-model architectures with large-scale automatic hyperparameter tuning and found that standard LSTMs, properly regularized, outperformed more recent models [Melis et al., 2018]. When you read "new method beats baseline", ask how much tuning each side received.

## 2. Reproducing results

There are three distinct things people call reproduction, and you should say which one you did:

- **Rerun:** the authors' code and data, your hardware. Tests whether the artifact works and the reported numbers come out.
- **Reimplement:** your code, from the paper's description. Tests whether the paper describes the method fully.
- **Replicate the claim:** a new dataset, scale or setting. Tests whether the finding is general.

Each level catches different problems. A rerun that fails usually means environment or data differences. A reimplementation that fails often reveals an undocumented detail (a learning-rate warmup, a data filter, an evaluation script that differs from the description). A failed replication is a scientific result in its own right.

The NeurIPS 2019 reproducibility program introduced a code submission policy, a community reproducibility challenge and a checklist for authors, and its organizers report on what each changed [Pineau et al., 2021]. As of October 2026 the NeurIPS paper checklist has 16 items, from claims and limitations through statistical significance, compute, licenses and safeguards to a declaration of LLM usage [NeurIPS Paper Checklist]; it doubles as a list of what a reproducer needs. The ML Reproducibility Challenge (MLRC) runs every year; reports are submitted to TMLR, and MLRC 2026 is a track at NeurIPS 2026 [MLRC]. Papers with Code, long the standard place to find code for a paper, now redirects to Hugging Face's trending-papers page (I checked the redirect; news reports dated its shutdown to July 2025, which I could not confirm from a primary source).

### A paper-reproduction checklist

Use this before, during and after a reproduction. It is also the plan for capstone 4 ("Reproduce a small paper result").

**Before you start**

- [ ] Write the exact claim you will test (the one-sentence form from section 1) and the number you expect, with the paper's table or figure reference.
- [ ] Choose the level (rerun, reimplement, replicate) and a compute budget in hours. Pick a claim you can test at small scale; most papers have one.
- [ ] Collect every detail: model and data sizes, preprocessing, tokenizer, optimizer and schedule, batch size, number of steps, regularization, evaluation script, number of seeds, hardware. List what is missing.
- [ ] Check for official code, issues on its repository, errata, later versions of the paper, and existing reproductions.
- [ ] Decide in advance what counts as "reproduced": for example, within the paper's reported spread, or within your own 95% interval over seeds.

**While running**

- [ ] Record every run with a tracker (section 4): config, seed, code version, environment, metrics.
- [ ] First reproduce the *baseline*. If you cannot match the baseline, the method comparison means nothing.
- [ ] Overfit a tiny subset first, then scale up (L2-02).
- [ ] Change one thing at a time and keep a written log of decisions and surprises, with dates.
- [ ] Run several seeds of the final configuration before comparing.

**After**

- [ ] Report your number with its uncertainty next to the paper's number, and say whether your success criterion was met.
- [ ] List every deviation from the paper and every guess you had to make.
- [ ] Separate "did not reproduce" from "could not test" (for example because of compute).
- [ ] Release code and configs, and contact the authors politely if a detail was missing; many reply.

## 3. Ablations and experimental hygiene

An **ablation** removes or replaces one component of a method to measure its contribution. Two designs:

- **Leave-one-out:** start from the full method and remove one component at a time. Measures what each part adds *given the others*.
- **Add-one-in:** start from a baseline and add one component at a time. Measures what each part adds alone.

They disagree when components interact. Regularizers are the usual example: dropout and weight decay may each help alone and be redundant together. If the two designs disagree, say so; that is a finding.

**Hygiene rules that matter most:**

1. **Three splits, used for three purposes.** Train on the training set, make every choice (hyperparameters, early stopping, which ablation to keep) on the validation set, and touch the test set once at the end (L1-01). Every look at the test set that influences a decision makes the test number optimistic.
2. **Seeds.** Henderson et al. showed that in deep reinforcement learning, results can differ significantly just from changing random seeds, so that comparisons on a handful of runs can be misleading [Henderson et al., 2018]. Bouthillier et al. measured the sources of variance in ML benchmarks, including data sampling, initialization and hyperparameter search, and recommend randomizing as many of them as possible when comparing methods [Bouthillier et al., 2021]. Agarwal et al. recommend reporting interval estimates and robust aggregates rather than point estimates from a few runs [Agarwal et al., 2021].
3. **Pair your comparisons.** If each seed fixes data order and augmentation, then the same seed for two configs shares those draws, and the difference per seed has less noise than the two configs' spreads suggest. Use a paired test.
4. **Equal tuning effort.** Give the baseline the same hyperparameter budget as your method (section 1).
5. **Count your comparisons.** With six ablations each tested at $p < 0.05$, the chance of at least one false alarm when nothing is real is up to $1 - 0.95^6 \approx 0.26$. Report all comparisons you made, not only the significant ones.
6. **Report compute.** Wall-clock time, hardware and number of runs, including failed and exploratory ones if you can.

The confidence interval for a mean over $n$ seeds uses the $t$ distribution: $\bar x \pm t_{0.975,\,n-1}\, s/\sqrt{n}$, where $s$ is the sample standard deviation. With $n = 5$, $t_{0.975,4} = 2.78$, so the half-width is $1.24\,s$: five seeds give a wide interval. This interval covers variation from seeds only. A fixed test set of $m$ examples adds sampling error of about $\sqrt{a(1-a)/m}$ for accuracy $a$, which is 0.006 for $a = 0.96$ and $m = 997$, the test set in our scripts. It is shared by all configs and largely cancels in paired comparisons, but not in claims about the absolute number.

Scripts 01 and 02 run a leave-one-out ablation over five components of a small digit classifier, five seeds each. The results, discussed in the hands-on section, include two common surprises: a component everyone "knows" helps (data augmentation) turns out to hurt in this setting, and one ablation would have looked harmless from a single seed.

## 4. Tracking experiments

Every number you report should be traceable to: the full configuration (including the seed), the code version, the data version, the environment (package versions, hardware), and the logs. Without this you cannot answer the most common question in research, "which run was that?"

The tracker in this module (`tracker.py`) is about 100 lines and shows the essentials: each run gets a directory named by a hash of its configuration, containing `config.json`, `env.json` (Python and library versions, platform, git commit with a "-dirty" flag for uncommitted changes, and a hash of the script), `metrics.jsonl` (one line per logged step), and `summary.json`, written only if the run finished. Crashed runs are therefore visible and never mixed into results.

Production tools add a UI, a server, artifact storage and comparison views. MLflow Tracking logs parameters, metrics and artifacts per run [MLflow docs]. DVC (dvc.org) versions data and models alongside git. Hosted services such as Weights & Biases do similar things. Pick one and use it from the first experiment; retrofitting tracking is painful. MLOps practice, including pipelines and model registries, is covered in L4-01.

**Determinism.** Same seed should mean same result, but only under the same conditions. Script 03 shows what breaks bit-for-bit equality on CPU: changing the number of threads changed the final weights by about $10^{-7}$, because parallel sums add numbers in a different order and floating-point addition is not associative (L0-01). Changing data order, batch size or seed changes them by $10^{-2}$ to $10^{-1}$. PyTorch's reproducibility notes say that completely reproducible results are not guaranteed across releases, commits or platforms, and describe `torch.use_deterministic_algorithms` and the cuDNN settings that trade speed for determinism on GPUs [PyTorch, Reproducibility]. The practical rule: aim for determinism to debug, but make claims that hold across seeds, because a result that depends on bit-level details is not a result.

**A lab notebook.** Keep a dated, plain-text log: what you tried, why, what happened, and what you concluded. It takes two minutes a day and is one of the most useful research habits.

## 5. Writing a short research report

A two- to four-page report is the right format for a reproduction, a project or an internal study. A structure that works:

1. **Title and one-sentence result.** The finding, not the topic: "Removing augmentation improves a small digit MLP by 1.7 points" rather than "A study of augmentation".
2. **Question and motivation** (one paragraph). What you asked and why anyone should care.
3. **Setup.** Data, splits, model, training details, number of seeds, compute. Enough for a reader to rerun it.
4. **Results.** One main table or figure that answers the question, with uncertainty (intervals, or mean and standard deviation over seeds, and say which). Then the ablations.
5. **What the results do and do not show.** Separate the measured finding from your explanation of it, and label speculation as speculation (the first of Lipton and Steinhardt's trends).
6. **Limitations.** Scale, datasets, seeds, what you did not test.
7. **Reproducibility.** Link to code and configs, and the environment.

Write the results table first, then the sentence that states what the table shows, then everything else. Every number in the text should appear in a table or log. Script 02 prints a "report line" with the interval already in it; get used to never writing a bare number.

## 6. Finding problems

Hamming's talk "You and Your Research" is the classic on choosing problems. He argues that if you do not work on an important problem you are unlikely to do important work, that great scientists keep a number of important problems in mind and watch for ways to attack them, and that a problem is important not because of its consequences alone but because you have a reasonable attack on it; he recalls asking colleagues what the important problems of their field were, and why they were not working on them [Hamming, 1986]. Add one more test: a good problem is one where *you* have some advantage (a skill, a dataset, a vantage point).

Practical sources of problems, especially with modest compute:

- **Failures you hit.** A reproduction that does not reproduce, a method that breaks on your data, a benchmark that rewards the wrong thing (L4-03). Your confusion is often the field's.
- **Missing ablations and baselines.** Many papers leave "which part matters?" unanswered. Answering it carefully is publishable and useful.
- **Scaling down.** Find the smallest setting where a phenomenon appears (a toy model, a synthetic task) and study it there. Much interpretability work (L4-04) and many scripts in this course follow this pattern.
- **Evaluation and measurement.** Better ways to measure something often matter more than better models.
- **Transfer to a domain you know.** Domain knowledge plus ML is a real advantage (L3-10).
- **Tools.** A library, dataset or harness that others need.

Before committing weeks, write a one-page plan: the question, why it matters, the first experiment, what result would change your mind, and what you will do if the first idea fails. Then run the cheapest experiment that could kill the idea.

## 7. Contributing to open source

Contributions are how many people enter the field, and they are public evidence of your skill. The "How to Contribute to Open Source" guide covers the mechanics and etiquette: read the contributing guide and code of conduct, look at open issues, and communicate before large changes [Open Source Guides].

A path that works for ML libraries:

1. **Use the library seriously** and note what confused you.
2. **Start small:** documentation fixes, a missing example, a clearer error message, a test that reproduces a reported bug. Issues labeled "good first issue" exist for this.
3. **Open an issue before a large pull request**, describing the problem and your proposed fix; maintainers may already have a plan.
4. **Make reviewable pull requests:** one change, with tests, following the project's style. Respond to review without taking it personally.
5. **Reproductions as contributions.** A clean reimplementation of a paper with matching numbers, or an issue documenting a discrepancy, is valuable to everyone.

Check the license before reusing code or weights, and do not paste code from a project with an incompatible license into another.

## 8. The shape of careers in AI

Job titles vary across companies and change often; the underlying kinds of work are more stable. As of October 2026 I could not find primary-source data on hiring or pay that I trust for a general statement, so this section describes the work, not the market.

| Role family | Typical work | What it rewards | Evidence that you can do it |
|---|---|---|---|
| Research scientist | Choosing questions, designing experiments, writing papers | Taste in problems, rigor, clear writing | Papers, reproductions, a research report people cite or use |
| Research engineer | Building training and evaluation infrastructure, running large experiments, implementing ideas fast | Strong software engineering plus ML depth | Clean reimplementations, contributions to ML libraries |
| ML engineer / applied scientist | Turning models into products: data pipelines, training, evaluation, serving, monitoring | Engineering judgment, measurement, reliability (L4-01, L4-02) | End-to-end projects with honest evaluation |
| Data scientist / data engineer | Framing questions with data, analysis, experiments (A/B tests), data infrastructure | Statistics, communication, domain knowledge | Analyses that changed a decision; robust pipelines |
| Safety, evaluation and governance | Technical alignment and interpretability research, evaluations and red teaming, policy and standards work | Rigor about failure, threat modeling, ability to work across technical and policy worlds (L4-03, L4-04, L4-05) | Evals, red-team reports, careful written analysis |
| Product and AI engineering | Building applications on top of models: retrieval, agents, guardrails, UX; product management | Judgment about what to build, evaluation of systems (L3-06, L3-07) | Shipped tools with users and an eval harness |

Some honest observations:

- **The boundaries are porous.** Research engineers publish, applied scientists do research, and many people move between these roles.
- **Credentials versus public work.** Research scientist postings often ask for a publication record or a PhD, while engineering roles tend to weigh public code and shipped work more. This is a general pattern, not a measured statistic, and it varies by organization; read the postings of the places you are interested in.
- **Safety work spans several roles.** 80,000 Hours, a career-advice organization that treats AI risk as a priority, publishes a career review of technical AI safety research describing paths into it [80,000 Hours]. Read it as one organization's perspective.
- **The work is changing.** AI coding and research assistants are changing how much of each job is routine implementation. Claims about which jobs will grow or shrink are speculation. My own view, which is an opinion, is that measurement and judgment (deciding what is true and what matters) are the skills least likely to lose value.

## 9. How to keep learning, and a map of the field

**Habits.** Skim new papers in your area weekly (arXiv listings, conference proceedings), read one paper a week in depth with a five-line note, reproduce one result a quarter, and write publicly about what you learn. Implement things; reading alone produces the illusion of understanding.

**Venues.** As of October 2026:

| Venue | What it is | Where |
|---|---|---|
| NeurIPS, ICML, ICLR | The main general machine learning conferences; ICLR hosts papers on OpenReview with public review and discussion | neurips.cc, icml.cc, iclr.cc, openreview.net |
| ACL, EMNLP, NAACL | Natural language processing; papers free in the ACL Anthology (CC BY 4.0 from 2016 on) | aclweb.org, aclanthology.org |
| CVPR, ICCV | Computer vision | thecvf.com |
| AAAI | Broad artificial intelligence | aaai.org |
| FAccT | ACM Conference on Fairness, Accountability, and Transparency | facctconference.org |
| JMLR | Journal of Machine Learning Research; all published papers free online | jmlr.org |
| TMLR | Transactions on Machine Learning Research: rolling submissions, double-blind review hosted on OpenReview, judged on correctness rather than perceived significance; also the route into the ML Reproducibility Challenge | jmlr.org/tmlr |
| arXiv | Preprint server; many ML papers appear here before or alongside peer review, and an arXiv posting is not itself peer-reviewed | arxiv.org (cs.LG, cs.CL, cs.CV, stat.ML) |

**Textbooks with free official versions** (all checked; I cite no chapter numbers):

| Book | Good for |
|---|---|
| Goodfellow, Bengio and Courville, *Deep Learning* (MIT Press, 2016) | Classic foundations of deep learning |
| Prince, *Understanding Deep Learning* (MIT Press, 2023; free PDF updated 2026) | Modern, well-illustrated deep learning, including transformers and diffusion |
| Zhang, Lipton, Li and Smola, *Dive into Deep Learning* (Cambridge University Press, 2023) | Code-first deep learning with runnable notebooks |
| Murphy, *Probabilistic Machine Learning: An Introduction* (2022) and *Advanced Topics* (2023) | Probabilistic view of ML, reference depth |
| Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning* (2nd ed., 2009); James et al., *An Introduction to Statistical Learning* (Python edition 2023) | Statistical learning, classical ML |
| Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018) | Reinforcement learning (L3-05) |
| Barocas, Hardt and Narayanan, *Fairness and Machine Learning* (MIT Press, 2023) | Fairness (L4-05) |

**Communities.** Open research communities let you contribute without an institution: EleutherAI (open-source language-model research), Cohere Labs (Cohere's research lab and open science community, called Cohere For AI until April 2025), ML Collective (an open research collective), and the Hugging Face hub and forums for models, datasets and courses. Local reading groups and paper clubs are easy to start and are one of the best ways to keep reading.

## Hands-on

The scripts are in `code/L4-06-research-skills-and-careers/`. Run them from any directory; scripts 01 and 02 write and read run folders under `runs/ablation` relative to where you run them (pass `--root` to change it). I ran them from a scratch directory with the shared course environment (Python 3.12, PyTorch 2.15.0+cpu, NumPy 2.5.4, scikit-learn 1.9.1). All data is scikit-learn's bundled digits set, so nothing is downloaded.

### tracker.py: a minimal experiment tracker

```python
"""A minimal, dependency-free experiment tracker (template).

Each run gets its own directory under a root folder:
  <root>/<run_id>/config.json   the full configuration, including the seed
  <root>/<run_id>/env.json      Python and package versions, platform, git commit,
                                and a hash of the code file that produced the run
  <root>/<run_id>/metrics.jsonl one JSON object per logged step
  <root>/<run_id>/summary.json  final metrics and wall time, written on success

A run that crashes has no summary.json, so it is easy to find and never silently
mixed into results. Tools like MLflow or Weights & Biases do the same job with a
UI, a server and many more features; the ideas are identical.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _git_commit(path):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True,
                             text=True, timeout=5)
        if out.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=path,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def environment(code_file):
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "argv": sys.argv}
    for mod in ["numpy", "torch", "sklearn"]:
        if mod in sys.modules:
            env[mod] = getattr(sys.modules[mod], "__version__", "?")
    code = Path(code_file).resolve()
    env["code_file"] = code.name
    env["code_sha256"] = hashlib.sha256(code.read_bytes()).hexdigest()[:16]
    env["git_commit"] = _git_commit(code.parent)
    return env


def config_id(config):
    """Deterministic short id from the config, so reruns of the same config are
    recognizable and duplicates can be detected."""
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


class Run:
    def __init__(self, root, config, code_file):
        self.config = dict(config)
        self.dir = Path(root) / f"{config_id(self.config)}"
        self.code_file = code_file

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, sort_keys=True))
        (self.dir / "env.json").write_text(json.dumps(environment(self.code_file), indent=2))
        self._metrics = open(self.dir / "metrics.jsonl", "w")
        self._t0 = time.perf_counter()
        return self

    def log(self, step, **metrics):
        self._metrics.write(json.dumps({"step": step, **metrics}) + "\n")

    def finish(self, **final):
        self._final = final

    def __exit__(self, exc_type, exc, tb):
        self._metrics.close()
        if exc_type is None:  # only successful runs get a summary
            summary = {**getattr(self, "_final", {}),
                       "wall_seconds": round(time.perf_counter() - self._t0, 3)}
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return False  # never swallow exceptions


def load_runs(root):
    """Return a list of dicts: config merged with summary, for finished runs only,
    plus the number of unfinished (crashed or running) runs."""
    rows, unfinished = [], 0
    for d in sorted(Path(root).glob("*/")):
        if not (d / "summary.json").exists():
            unfinished += 1
            continue
        rows.append({**json.loads((d / "config.json").read_text()),
                     **json.loads((d / "summary.json").read_text()), "run_id": d.name})
    return rows, unfinished
```

### Script 01: an ablation harness, `01_ablation_harness.py`

The configuration is a frozen dataclass, so every run is fully described by one object, and ablations are written as `replace(full, ...)`. The data split is fixed by `data_seed`; the run `seed` controls initialization, dropout, data order and augmentation.

```python
"""Ablation harness template: one config per run, several seeds per config, every
run logged with tracker.py. Analysis lives in 02_analyze_runs.py.

Task: classify scikit-learn's bundled 8x8 digits with a small MLP trained on only
400 images, so that seed-to-seed variation is visible. The "full" method has five
components; each ablation removes one. Usage:
    python 01_ablation_harness.py [--root runs/ablation] [--seeds 5]
"""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

from tracker import Run

torch.set_num_threads(1)


@dataclass(frozen=True)
class Config:
    name: str = "full"
    seed: int = 0
    data_seed: int = 0         # fixed: every config sees the same split
    n_train: int = 400
    epochs: int = 40
    lr: float = 3e-3
    layernorm: bool = True
    dropout: float = 0.2
    weight_decay: float = 1e-3
    augment: bool = True       # random 1-pixel shifts
    cosine_lr: bool = True


def get_data(cfg):
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    idx = np.random.default_rng(cfg.data_seed).permutation(len(X))
    tr, va, te = idx[:cfg.n_train], idx[cfg.n_train:cfg.n_train + 400], idx[cfg.n_train + 400:]
    t = lambda a: torch.tensor(a)
    return (t(X[tr]), t(y[tr])), (t(X[va]), t(y[va])), (t(X[te]), t(y[te]))


def shift(xb, g):
    """Random shift by up to one pixel in each direction (zero padding)."""
    imgs = F.pad(xb.view(-1, 1, 8, 8), (1, 1, 1, 1))
    dx, dy = torch.randint(0, 3, (2,), generator=g).tolist()
    return imgs[:, :, dy:dy + 8, dx:dx + 8].reshape(-1, 64)


def build(cfg):
    layers, d = [], 64
    for _ in range(2):
        layers += [torch.nn.Linear(d, 256)]
        layers += [torch.nn.LayerNorm(256)] if cfg.layernorm else []
        layers += [torch.nn.ReLU(), torch.nn.Dropout(cfg.dropout)]
        d = 256
    return torch.nn.Sequential(*layers, torch.nn.Linear(d, 10))


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    a = (model(X).argmax(1) == y).float().mean().item()
    model.train()
    return a


def run_one(cfg, root):
    torch.manual_seed(cfg.seed)              # init and dropout
    g = torch.Generator().manual_seed(cfg.seed)  # data order and augmentation
    (Xtr, ytr), (Xva, yva), (Xte, yte) = get_data(cfg)
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(cfg.n_train / 32))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cfg.cosine_lr
             else torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0))
    with Run(root, asdict(cfg), __file__) as run:
        for ep in range(cfg.epochs):
            perm = torch.randperm(cfg.n_train, generator=g)
            for i in range(0, cfg.n_train, 32):
                b = perm[i:i + 32]
                xb = shift(Xtr[b], g) if cfg.augment else Xtr[b]
                loss = F.cross_entropy(model(xb), ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if ep % 10 == 9:
                run.log(ep + 1, loss=loss.item(), val_acc=accuracy(model, Xva, yva))
        # The test set is evaluated once, at the end, and never used for choices.
        run.finish(val_acc=accuracy(model, Xva, yva), test_acc=accuracy(model, Xte, yte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/ablation")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    full = Config()
    configs = [full,
               replace(full, name="- layernorm", layernorm=False),
               replace(full, name="- dropout", dropout=0.0),
               replace(full, name="- weight decay", weight_decay=0.0),
               replace(full, name="- augmentation", augment=False),
               replace(full, name="- cosine lr", cosine_lr=False),
               replace(full, name="none (plain MLP)", layernorm=False, dropout=0.0,
                       weight_decay=0.0, augment=False, cosine_lr=False)]
    t0 = time.perf_counter()
    for cfg in configs:
        for s in range(args.seeds):
            run_one(replace(cfg, seed=s), args.root)
    print(f"{len(configs) * args.seeds} runs written to {args.root} "
          f"in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
```

Output:

```
35 runs written to runs/ablation in 15.2 s
```

### Script 02: analysis with intervals and paired differences, `02_analyze_runs.py`

```python
"""Analyze the runs written by 01_ablation_harness.py.

1. Mean and 95% confidence interval per config (t interval over seeds).
2. Paired differences from "full": the same seed index shares data order and
   augmentation draws, so pairing removes some shared noise.
3. What one seed would have told you, and how much "best seed" reporting inflates.
Usage:  python 02_analyze_runs.py [--root runs/ablation] [--metric test_acc]
"""
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats

from tracker import load_runs


def t_ci(x, level=0.95):
    x = np.asarray(x)
    half = stats.t.ppf((1 + level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - half, x.mean() + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/ablation")
    ap.add_argument("--metric", default="test_acc")
    args = ap.parse_args()

    rows, unfinished = load_runs(args.root)
    by = defaultdict(dict)  # config name -> {seed: metric}
    for r in rows:
        by[r["name"]][r["seed"]] = r[args.metric]
    print(f"{len(rows)} finished runs, {unfinished} unfinished, metric = {args.metric}\n")

    full = by["full"]
    print(f"{'config':18} {'n':>2} {'mean':>6} {'95% CI':>15} | {'diff vs full':>12} "
          f"{'paired 95% CI':>17} {'p':>6} | {'seed 0 diff':>11}")
    for name, res in sorted(by.items(), key=lambda kv: -np.mean(list(kv[1].values()))):
        seeds = sorted(res)
        m, lo, hi = t_ci([res[s] for s in seeds])
        line = f"{name:18} {len(seeds):2d} {m:6.3f} [{lo:6.3f}, {hi:6.3f}] |"
        if name != "full":
            common = sorted(set(seeds) & set(full))
            d = np.array([res[s] - full[s] for s in common])
            dm, dlo, dhi = t_ci(d)
            p = stats.ttest_rel([res[s] for s in common], [full[s] for s in common]).pvalue
            line += f" {dm:+12.3f} [{dlo:+7.3f}, {dhi:+7.3f}] {p:6.3f} | {d[0]:+11.3f}"
        print(line)

    # Reporting the best of k seeds instead of the mean.
    print("\nBest-seed reporting (max over seeds minus mean over seeds):")
    for name in ["full", "none (plain MLP)"]:
        v = np.array(list(by[name].values()))
        print(f"  {name:18} mean {v.mean():.3f}, best {v.max():.3f}, inflation {v.max() - v.mean():+.3f}")

    # A results sentence a reader can check, with the uncertainty in it.
    m, lo, hi = t_ci(list(full.values()))
    print(f"\nReport line: the full method reaches {m:.3f} test accuracy "
          f"(95% CI {lo:.3f} to {hi:.3f}, {len(full)} seeds, 400 training images).")


if __name__ == "__main__":
    main()
```

Output:

```
35 finished runs, 0 unfinished, metric = test_acc

config              n   mean          95% CI | diff vs full     paired 95% CI      p | seed 0 diff
- augmentation      5  0.973 [ 0.968,  0.979] |       +0.017 [ +0.015,  +0.020]  0.000 |      +0.015
none (plain MLP)    5  0.960 [ 0.957,  0.963] |       +0.005 [ -0.002,  +0.011]  0.118 |      +0.003
- dropout           5  0.960 [ 0.951,  0.970] |       +0.005 [ +0.001,  +0.009]  0.035 |      +0.004
full                5  0.956 [ 0.950,  0.962] |
- weight decay      5  0.955 [ 0.948,  0.963] |       -0.000 [ -0.005,  +0.004]  0.822 |      -0.002
- layernorm         5  0.936 [ 0.932,  0.941] |       -0.019 [ -0.022,  -0.017]  0.000 |      -0.017
- cosine lr         5  0.930 [ 0.903,  0.957] |       -0.026 [ -0.049,  -0.003]  0.037 |      +0.000

Best-seed reporting (max over seeds minus mean over seeds):
  full               mean 0.956, best 0.961, inflation +0.005
  none (plain MLP)   mean 0.960, best 0.964, inflation +0.003

Report line: the full method reaches 0.956 test accuracy (95% CI 0.950 to 0.962, 5 seeds, 400 training images).
```

What the table shows:

- **Augmentation hurts here.** Removing the random one-pixel shift raises test accuracy by 0.017 (paired 95% interval 0.015 to 0.020). On an 8x8 image a one-pixel shift moves the digit by an eighth of the image and pushes pixels off the edge, so it is not the label-preserving transformation that small shifts are on large images. Without the ablation, the "full" method would have carried a component that costs accuracy.
- **The plain MLP is as good as the full method.** Its difference, +0.005 with an interval from -0.002 to +0.011, is not distinguishable from zero with five seeds. "Our method with five tricks" is not better than no tricks on this task.
- **Layer normalization helps** (-0.019 without it) and **weight decay does nothing measurable** (interval -0.005 to +0.004).
- **One seed would have misled you about the cosine schedule.** With seed 0 the difference is +0.000 (0.957 for both). The other four seeds are worse without the schedule, by 0.022 to 0.052 (per-seed accuracies 0.938, 0.901, 0.937, 0.918 against 0.960, 0.953, 0.961, 0.949), so the five-seed mean difference is -0.026 with a wide interval. The spread itself is a result: the constant learning rate is less stable.
- **Six comparisons were made.** Two have $p$ near 0.035 to 0.037; with six tests, treat those as suggestive, not established.
- **Best-seed reporting** inflates the full method by 0.005, about the size of several of the effects in the table.

These conclusions are about this small task. On larger images, augmentation usually helps; the point is that you find out by ablating, not by assuming.

### Script 03: what breaks bit-for-bit reproducibility, `03_determinism_check.py`

```python
"""Reproducibility check: is the training run bit-for-bit repeatable, and what breaks it?

Trains the same small model under several conditions and compares a hash of the
final weights and the test accuracy with a reference run.
"""
import hashlib
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
Xtr, ytr, Xte, yte = X[:1200], y[:1200], X[1200:], y[1200:]


def train(seed=0, threads=1, batch=32, dtype=torch.float32, shuffle_seed=None):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(64, 512), torch.nn.ReLU(),
                                torch.nn.Linear(512, 10)).to(dtype)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    g = torch.Generator().manual_seed(seed if shuffle_seed is None else shuffle_seed)
    for _ in range(10):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(model(Xtr[b].to(dtype)), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = (model(Xte.to(dtype)).argmax(1) == yte).float().mean().item()
        flat = torch.cat([p.detach().double().flatten() for p in model.parameters()])
    digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:12]
    return digest, acc, flat


ref_hash, ref_acc, ref_w = train()
print(f"torch {torch.__version__}, reference run: weights {ref_hash}, test acc {ref_acc:.4f}\n")
print(f"{'condition':40} {'weights hash':>13} {'same?':>6} {'max |dw|':>9} {'test acc':>8}")
conditions = [
    ("same seed, same settings (rerun)", {}),
    ("same seed, 8 threads", {"threads": 8}),
    ("same init, different data order", {"shuffle_seed": 1}),
    ("different seed (init and order)", {"seed": 1}),
    ("batch size 64 instead of 32", {"batch": 64}),
    ("float64 instead of float32", {"dtype": torch.float64}),
]
for name, kw in conditions:
    h, a, w = train(**kw)
    print(f"{name:40} {h:>13} {str(h == ref_hash):>6} {(w - ref_w).abs().max().item():9.2e} {a:8.4f}")

# Across 10 seeds, how much does test accuracy move? That spread is the noise
# floor any claimed improvement must clear.
accs = [train(seed=s)[1] for s in range(10)]
print(f"\ntest accuracy over 10 seeds: mean {np.mean(accs):.4f}, sd {np.std(accs, ddof=1):.4f}, "
      f"range [{min(accs):.4f}, {max(accs):.4f}]")
```

Output:

```
torch 2.15.0+cpu, reference run: weights 9c5a6c971315, test acc 0.9062

condition                                 weights hash  same?  max |dw| test acc
same seed, same settings (rerun)          9c5a6c971315   True  0.00e+00   0.9062
same seed, 8 threads                      c0ae888e7f76  False  1.49e-07   0.9062
same init, different data order           96ade2700078  False  1.74e-02   0.9079
different seed (init and order)           4ff4080dc248  False  5.27e-01   0.9079
batch size 64 instead of 32               8f2767a6bcb2  False  7.06e-02   0.8911
float64 instead of float32                ae604924ff94  False  3.05e-07   0.9062

test accuracy over 10 seeds: mean 0.9047, sd 0.0043, range [0.8945, 0.9095]
```

A rerun with the same seed and settings is bit-identical. Eight threads instead of one changes the weights by about $1.5 \times 10^{-7}$ (different summation order) without changing accuracy here; over a long run such differences can grow. Changing the data order alone moves weights by $1.7 \times 10^{-2}$, and a new seed by 0.5. The seed-to-seed standard deviation of test accuracy, 0.0043, is the noise floor: a claimed improvement of 0.002 from one run of each configuration means nothing. Thread-count effects depend on your CPU and PyTorch build, so your hashes will differ from mine; the pattern should not.

## Common mistakes

- **Reading every paper fully.** Use passes; most papers deserve five minutes.
- **Trusting a comparison without checking tuning effort.** Ask how much tuning the baseline received.
- **One seed per configuration.** Gains smaller than the seed spread are noise. Run several seeds and report intervals.
- **Peeking at the test set.** Choices made by looking at test results inflate them. Decide on validation, test once.
- **Starting the reproduction with the new method.** Reproduce the baseline first.
- **Untracked runs.** A number you cannot trace to a config and commit is a rumor. Log from the first run.
- **Reporting only what worked.** Report all ablations and comparisons, including null results.
- **Explaining before measuring.** Write the measured finding first; label explanations as hypotheses.
- **Choosing problems only by importance or only by ease.** You need both, plus an advantage.
- **A giant first pull request.** Start small and talk to maintainers first.

## Exercises

1. [warm-up] Take a recent paper in an area you know. Do a first pass in ten minutes and write its claim in the one-sentence form from section 1.
2. [warm-up] Open one run directory written by script 01. Which fields would you need to rerun that exact run? Is anything missing?
3. [warm-up] With five seeds and sample standard deviation 0.004, what is the half-width of the 95% $t$ interval for the mean? How many seeds halve it?
4. [core] Rerun script 01 with `--seeds 10` into a new root and rerun script 02. Which conclusions change? Do the cosine-schedule and dropout intervals narrow as $1/\sqrt{n}$?
5. [core] Add an add-one-in ablation to script 01 (plain MLP plus each component alone). Compare with the leave-one-out results. Which components interact?
6. [core] Add a "tuned baseline": a small grid over learning rate and weight decay for the plain MLP, selected on validation accuracy. Does the full method still look the same relative to it?
7. [core] Extend `tracker.py` so that `Run` refuses to start if the git working tree is dirty, unless a flag allows it. Why might you want this, and why might it be annoying?
8. [core] Write a two-page report on the script 01 and 02 results using the structure in section 5, including a limitations section.
9. [stretch] Pick a small result from a paper (one that runs on CPU in under an hour) and reproduce it with the checklist. Write the report. This is capstone 4.
10. [stretch] Add test-set sampling uncertainty to script 02 with a bootstrap over test examples (resample the same test indices for all configs). Compare intervals for absolute accuracy and for paired differences.

### Hints and answers

1. If you cannot fill in "under conditions C", look for the experimental setup section; missing conditions are a red flag.
2. `config.json` plus the code hash and package versions are enough on the same machine; data is bundled with scikit-learn, but for a real dataset you would also need a data version or hash. The git commit is `null` here because I ran outside a repository.
3. $2.78 \times 0.004 / \sqrt 5 \approx 0.005$. The half-width scales as $t_{n-1}/\sqrt n$; about $n = 15$ to 16 seeds roughly halves it (the $t$ value also shrinks).
4. Expect the large effects (augmentation, layer norm) to stay; borderline ones (dropout) may move. Widths shrink roughly as $1/\sqrt n$, faster at first because $t$ shrinks too.
5. Regularizers often interact: on 400 images several may help alone and be redundant together. Report both tables.
6. A tuned baseline is often the strongest result in a table; this is Melis et al.'s lesson at small scale.
7. It forces every run to correspond to a commit, so results are traceable; it slows rapid exploration. A common compromise is to store the diff with the run.
8. Lead with the augmentation finding, give intervals, state that conclusions hold for this task and scale.
9. Good candidates are small results with released code. Reproduce the baseline first, and decide your success criterion before running.
10. Sample test indices with replacement once per bootstrap round and apply them to every config. Paired-difference intervals change little (shared test noise cancels); absolute-accuracy intervals widen.

## Check your understanding

**1. What should you be able to say after a first pass over a paper?** What kind of paper it is, what it builds on, whether its assumptions look plausible, what it claims to contribute, and whether it is worth more time. For ML papers, add the one-sentence claim and where the main evidence is.

**2. Why reproduce the baseline before the method?** Because the claim is a difference between the two. If your baseline is weaker or stronger than the paper's, your method comparison is shifted by the same amount, and you will not know whether the method or your setup explains a gap.

**3. Why use a paired comparison across seeds in script 02?** Each seed index fixes the data order and augmentation draws, which are shared across configurations, so per-seed differences cancel that shared noise. The paired interval is then narrower and more honest about the effect than comparing two independent spreads.

**4. Why can the same seed give different results?** Floating-point addition is not associative, so anything that changes the order of operations (threads, batch size, hardware, library versions, nondeterministic GPU kernels) changes the bits, and training can amplify small differences. That is why claims should hold across seeds rather than depend on one.

**5. What makes a problem worth working on?** It matters if solved, you can see a way to attack it, and you have some advantage. A cheap first experiment that could falsify the idea tells you quickly whether it is tractable.

## Further reading

- **S. Keshav, "How to Read a Paper"** (2007, ACM SIGCOMM Computer Communication Review 37(3)). Free copy: https://www.cs.princeton.edu/courses/archive/fall10/cos597B/papers/howtoread-keshav.pdf
  *Read for:* the three-pass method and the five Cs in two pages. *Level:* intro.
- **Richard W. Hamming, "You and Your Research"** (talk at Bellcore, March 7, 1986; transcript). https://www.cs.virginia.edu/~robins/YouAndYourResearch.html
  *Read for:* choosing important problems and working habits, from someone who did important work. *Level:* intro.
- **Zachary C. Lipton, Jacob Steinhardt, "Troubling Trends in Machine Learning Scholarship"** (2018, arXiv:1807.03341). https://arxiv.org/abs/1807.03341
  *Read for:* the four patterns to avoid in your own writing and to notice in others'. *Level:* intro.
- **Joelle Pineau, Philippe Vincent-Lamarre, Koustuv Sinha, Vincent Larivière, Alina Beygelzimer, Florence d'Alché-Buc, Emily Fox, Hugo Larochelle, "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)"** (2021, JMLR 22(164)). https://jmlr.org/papers/v22/20-303.html
  *Read for:* what code policies, checklists and reproducibility challenges change in practice. *Level:* intro.
- **Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger, "Deep Reinforcement Learning that Matters"** (2018, AAAI; arXiv:1709.06560). https://arxiv.org/abs/1709.06560
  *Read for:* how seeds, hyperparameters and codebases change conclusions; the 10-seed experiment is the memorable one. *Level:* intermediate.
- **Gábor Melis, Chris Dyer, Phil Blunsom, "On the State of the Art of Evaluation in Neural Language Models"** (2017, arXiv:1707.05589; presented at ICLR 2018). https://arxiv.org/abs/1707.05589
  *Read for:* a tuned LSTM baseline beating newer architectures; the case for equal tuning effort. *Level:* intermediate.
- **Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, et al. (16 authors), "Accounting for Variance in Machine Learning Benchmarks"** (2021, MLSys; arXiv:2103.03098). https://arxiv.org/abs/2103.03098
  *Read for:* which sources of variance matter and how to compare methods given them. *Level:* intermediate.
- **Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice"** (2021, NeurIPS; arXiv:2108.13264). https://arxiv.org/abs/2108.13264
  *Read for:* interval estimates, the interquartile mean and performance profiles for reporting few-run results. *Level:* intermediate.
- **NeurIPS, "NeurIPS Paper Checklist Guidelines"** (official page, 2026 version, accessed October 2026). https://neurips.cc/public/guides/PaperChecklist ; and the **ML Reproducibility Challenge** (official site). https://reproml.org/
  *Read for:* what reviewers now expect a paper to report, and a structured way to publish a reproduction. *Level:* intro.
- **Andrej Karpathy, "A Recipe for Training Neural Networks"** (blog, April 25, 2019). https://karpathy.github.io/2019/04/25/recipe/
  *Read for:* a practical sequence (data first, dumb baselines, overfit, regularize, tune) that prevents most wasted experiments. *Level:* intro.
- **PyTorch documentation, "Reproducibility"** (version 2.15). https://docs.pytorch.org/docs/2.15/notes/randomness.html ; **MLflow Tracking** documentation. https://mlflow.org/docs/latest/ml/tracking/
  *Read for:* what determinism PyTorch does and does not promise, and a production version of `tracker.py`. *Level:* intro.
- **Open Source Guides, "How to Contribute to Open Source"** (GitHub's opensource.guide). https://opensource.guide/how-to-contribute/ ; **80,000 Hours, "AI safety technical research" career review** (Benjamin Hilton, updated October 2026). https://80000hours.org/career-reviews/ai-safety-researcher/
  *Read for:* the mechanics and etiquette of a first contribution; one organization's view of the path into safety research. *Level:* intro.

The textbooks in section 9 have free official versions: deeplearningbook.org, udlbook.github.io/udlbook, d2l.ai, probml.github.io/pml-book, hastie.su.domains/ElemStatLearn, statlearning.com, incompleteideas.net/book/the-book-2nd.html and fairmlbook.org.
