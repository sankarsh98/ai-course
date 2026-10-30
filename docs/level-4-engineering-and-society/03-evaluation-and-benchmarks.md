---
title: "Evaluation and Benchmarks"
---

> **Level 4 · Module 3** · Prerequisites: L0-04, L1-01, L3-02, L3-06, L3-07 · Time: 7 to 9 hours · Code: `code/L4-03-evaluation-and-benchmarks/`

## What you will be able to do

- Design an evaluation that starts from a decision and a target population, and choose a metric and a grader that measure what that decision needs.
- Report an eval score with an honest interval: clustered standard errors, resampled answers, paired comparisons, and a power calculation done before the eval is built.
- Explain how benchmarks saturate and become contaminated, detect simple contamination, and say what a "decontaminated" score does and does not guarantee.
- Measure agreement between human annotators with Cohen's kappa and Krippendorff's alpha, and read those numbers correctly when classes are imbalanced.
- Use an LLM judge responsibly: test it for position and length bias, debias it, and validate it against human labels.
- Read a published evaluation claim and list the questions that decide whether to believe it.

## 1. What an evaluation is for

An evaluation is a measurement taken to support a decision: ship model B or keep A, choose this prompt, report progress on a capability, decide whether a system is safe enough to deploy. The decision fixes what the evaluation must measure. "Accuracy on benchmark X" is never the goal. It is a proxy for something like "users with these questions get correct answers more often".

Three gaps separate a number from the goal, and each has its own failure mode:

1. **The items are not the population.** A benchmark is a sample of tasks someone found convenient to collect. If your users ask about tax law and the benchmark is high-school science, the score answers a different question. This is a question of **construct validity**: does the test measure the thing it claims to measure? Raji et al. argue that a few influential benchmarks are treated as measures of general capability that they were never designed to capture, and that this framing has construct validity problems [Raji et al., 2021].
2. **The metric is not the outcome.** Exact match, BLEU or a judge's preference is a cheap stand-in for "the answer helped". The stand-in breaks when it is optimized: Goodhart's law and reward hacking are covered in L3-02, and Thomas and Uminsky collect cases where optimizing a metric led to harm and recommend combining metrics with qualitative checks and the views of affected people ([Thomas and Uminsky, 2020](https://arxiv.org/abs/2002.08512), later published in *Patterns*).
3. **The sample is finite.** Even a perfect benchmark gives a noisy estimate. Section 3 is about how noisy.

**Designing an eval, in order.** Write down the decision and the smallest difference that would change it (for example, "we switch if the new prompt is at least 3 points better"). Describe the population of inputs and sample from it, including the hard and ugly cases (paraphrases, typos, unanswerable questions). Decide the unit of analysis (a question, a conversation, a document with several questions) because that unit is what you resample. Write the grader and test it on known-good and known-bad outputs before trusting it. Compute the sample size you need (section 3). Keep a held-out split that you look at rarely. Agent-specific advice (outcome grading, pass@k versus pass^k, capability versus regression suites) is in L3-07 and is not repeated here.

**One number is rarely enough.** HELM evaluates each model on many scenarios and seven metrics (accuracy, calibration, robustness, fairness, bias, toxicity and efficiency), on the grounds that a single accuracy hides trade-offs ([Liang et al., 2023](https://arxiv.org/abs/2211.09110), "Holistic Evaluation of Language Models", TMLR). In your own work, break scores down by slice (topic, length, language, difficulty) and look at the slices before the average.

## 2. Where the number comes from

A reported score is a function of the model *and* of everything around it: the prompt template, the few-shot examples, the decoding settings, how the answer is extracted and how it is compared with the reference. These choices move scores by more than the differences people argue about.

**Format sensitivity.** Biderman et al., describing lessons from maintaining EleutherAI's lm-evaluation-harness, show ARC-Challenge scored two ways: as a "cloze" task, where the model's likelihood of each answer text is compared, and in MMLU style, where the model picks a letter. GPT-NeoX-20B scores 38.0 versus 26.6, while Mixtral-8x7B scores 56.7 versus 81.3: the ranking of the two models flips with the format [Biderman et al., 2024]. Their advice is to share the exact evaluation code and prompts, and to report enough detail that someone else can rerun the number.

**The grader is part of the metric.** Script 1 shows a small version. Model B is more accurate than model A (0.721 against 0.667 with a forgiving grader), but B phrases answers as "The answer is 160." A strict string-match grader scores B at 0.233 and A at 0.479 and reverses the ranking. Neither number is "wrong"; they measure different things. The fix is to decide what counts as correct, write a grader for that, and check it by reading a sample of graded outputs, especially disagreements between graders.

**Sampling settings.** With temperature above zero the same model gives different answers on reruns. Either fix the seed and temperature and say so, or treat the randomness as part of what you measure (section 3).

## 3. Statistics for small eval sets

L0-04 and L1-01 introduced confidence intervals, the bootstrap, paired comparisons and McNemar's test. This section applies them to the way LLM evals are actually built, following the framework in Miller's "Adding Error Bars to Evals" [Miller, 2024].

**The sampling model.** Treat the $n$ items as a random sample from a large population of possible questions, and the model's score $s_i$ on item $i$ as a random variable. The eval score is $\bar s = \frac1n \sum_i s_i$, and what we want is the population mean $\mu$. For independent items the central limit theorem gives

$$\mathrm{SE}(\bar s) = \frac{\sigma}{\sqrt n}, \qquad \text{for 0/1 scores } \sigma^2 = \bar s(1 - \bar s).$$

At $n = 240$ and $\bar s = 0.667$ that is $\sqrt{0.667 \cdot 0.333 / 240} = 0.0304$, so the 95% interval is about $\pm 6$ points (script 1 prints 0.0305, using the $n - 1$ denominator). A 240-item eval cannot distinguish 0.67 from 0.70.

**Clustered items.** Many benchmarks contain groups of related questions: several questions about one reading passage, several test cases for one function, several turns of one conversation. Items in a group succeed or fail together, so they carry less information than independent items. The cluster-robust standard error sums the residuals within each cluster $c$ before squaring:

$$\mathrm{SE}_{\text{cl}}(\bar s) = \frac{1}{n}\sqrt{\frac{G}{G - 1}\sum_{c=1}^{G} \Big(\sum_{i \in c} (s_i - \bar s)\Big)^2},$$

with $G$ clusters (the factor $G/(G-1)$ is a common small-sample correction). If items within a cluster were independent, the cross terms would average to zero and this would reduce to the usual formula. In script 1, with 40 passages of 6 questions, the clustered SE is 0.0368 against 0.0305 naive, and over 1,000 simulated eval sets the naive "95%" interval covered the true accuracy only 89.7% of the time, against 94.3% for the clustered one. The bootstrap equivalent is to resample whole clusters, which `evalkit.bootstrap_ci` does when given cluster labels.

**Resampling answers.** Each item's score has two sources of variance: which item was drawn (item difficulty) and the model's sampling randomness on that item. Averaging $K$ samples per item removes most of the second but none of the first. In script 1, ten samples per item cut the clustered SE from 0.0345 to 0.0304, a modest gain, because most of the variance in this simulation comes from item difficulty. Whether resampling pays depends on how much of your variance is sampling noise, which you can measure from a pilot run.

**Paired comparisons.** When two models answer the same items, analyse the per-item differences $d_i = s^B_i - s^A_i$:

$$\mathrm{Var}(\bar d) = \frac{1}{n}\left(\sigma_A^2 + \sigma_B^2 - 2\,\mathrm{Cov}(s^A, s^B)\right).$$

Models tend to find the same items hard, so the covariance is positive and the paired SE is smaller than treating the two scores as independent samples. In script 1 the paired SE is 0.0335 against 0.0421 unpaired. Even so, B's observed advantage of +0.054 has a paired permutation $p$ of 0.134. The true gap in this simulation is +0.052, so B really is better, and 240 items simply cannot show it. The **paired permutation test** used here flips the sign of each $d_i$ at random: under "no difference", A's and B's results on an item are exchangeable. Dror et al. survey which significance tests suit which NLP metrics and argue for permutation or bootstrap tests when the distribution of the metric is unknown [Dror et al., 2018].

**Power, before you build.** A two-sided paired $z$-test at level $\alpha$ detects a true difference $\delta$ with power $1 - \beta$ when

$$n \geq \left(\frac{(z_{1-\alpha/2} + z_{1-\beta})\,\sigma_d}{\delta}\right)^2,$$

where $\sigma_d$ is the standard deviation of the per-item differences (estimate it from a pilot or a previous comparison). For script 1's models this gives about 768 items to detect the true 5-point gap with 80% power; a 2-point gap would need roughly six times as many. Card et al. examined published NLP comparisons and found underpowered experiments to be common, including comparisons on small test sets and typical human-rating designs [Card et al., 2020]. An underpowered eval is not merely uninformative: the "significant" results it does produce overstate the effect, because only lucky draws cross the threshold.

**Many comparisons and reused test sets.** L0-04 showed that testing 20 useless variants yields a "significant" one about half the time. Script 2 repeats this with correlated variants of equal true accuracy: in 67.2% of repeats at least one variant passed $p < 0.05$ uncorrected, against 4.6% with the **Holm-Bonferroni** correction (multiply the $k$-th smallest of $m$ p-values by $m - k + 1$ and enforce monotonicity; it controls the chance of any false positive). **Benjamini-Hochberg** (scale the $k$-th smallest by $m/k$) controls instead the expected fraction of false discoveries, which suits screening many candidates. Correction does not remove the **winner's curse**: the best of 20 equal variants beat the baseline by +3.0 points on average, and on a fresh test set the chosen variant was no better (0.667 against 0.683).

Adaptive reuse is worse than multiple testing, because each choice depends on earlier results. Blum and Hardt showed that a competitor who only observes public leaderboard scores can climb the leaderboard without learning anything, and proposed a leaderboard mechanism, the Ladder, that limits this ([Blum and Hardt, 2015](https://arxiv.org/abs/1502.04585), ICML). Script 2 runs their idea: 400 random submissions, keep the 209 that scored above 0.5 on the public half, take a majority vote. The vote scores 0.597 publicly and 0.473 on the private half. Every time a team tunes against a fixed public benchmark, it runs a slow version of this attack.

## 4. Saturation and the landscape as of October 2026

A benchmark is useful while models spread out across its range. When the best models score near the ceiling, or near the level of label noise, differences stop meaning anything; that is **saturation**.

**Ceilings are lower than 100%.** Northcutt et al. estimated an average of at least 3.3% label errors across the test sets of 10 common datasets, and at least 6% in the ImageNet validation set. They also found that ResNet-18 would outperform ResNet-50 if the share of originally mislabeled test examples increased by just 6%: small differences between strong models can be decided by the errors [Northcutt et al., 2021]. For MMLU, the 57-subject multiple-choice benchmark ([Hendrycks et al., 2021](https://arxiv.org/abs/2009.03300), ICLR), Gema et al. found errors in 57% of the analysed Virology questions, estimated that 6.49% of all MMLU questions contain errors, and released MMLU-Redux, 5,700 re-annotated questions ([Gema et al., 2025](https://arxiv.org/abs/2406.04127), "Are We Done with MMLU?", NAACL). The Stanford AI Index 2026 cites a review that found invalid-question rates from 2% (MMLU math) to 42% (GSM8K) ([AI Index 2026, Technical Performance chapter](https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance)).

**The cycle is getting shorter.** The AI Index 2025 described MMLU, GSM8K and HumanEval as saturated and reported that scores on GPQA rose by 48.9 percentage points during 2024 ([AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report/technical-performance)). GPQA had been built to be hard: 448 graduate-level questions on which experts reached 65% (74% discounting clear mistakes), skilled non-experts with unrestricted web access 34%, and a GPT-4 baseline 39% ([Rein et al., 2023](https://arxiv.org/abs/2311.12022), "GPQA"). The AI Index 2026 says evaluations meant to be hard for years now saturate in months, and that frontier models gained 30 percentage points on Humanity's Last Exam (HLE) in a single year. HLE has 2,500 expert-written questions ([Phan et al., 2025](https://arxiv.org/abs/2501.14249); later published in *Nature* under a different title). As of October 2026, the top entry on Scale's HLE leaderboard scored 54.80% ± 1.94 (page last updated 17 September 2026, the date CAIS released a continuously updated "HLE-Rolling" set; [Scale leaderboard](https://labs.scale.com/leaderboard/humanitys_last_exam)). Hard benchmarks also have errors: FutureHouse reported that for 29% ± 3.7% of the 321 text-only chemistry and biology questions they reviewed, the official answer directly conflicts with peer-reviewed literature ([FutureHouse, July 2025](https://www.futurehouse.org/research/hle-exam)).

Other signs of the October 2026 landscape, each verified on the maintainer's page:

- **ARC-AGI-3**, launched 25 March 2026, is interactive: turn-based game environments with no stated rules. At launch the ARC Prize team reported humans at 100% and frontier AI at 0.51% ([ARC Prize](https://arcprize.org/blog/arc-agi-3-launch)).
- **FrontierMath** v2 (12 June 2026) corrected errors in 42% of its problems. Epoch AI notes that the benchmark was funded by OpenAI, which has exclusive access to a subset ([Epoch AI](https://epoch.ai/benchmarks/frontiermath)). Funding and access arrangements are part of what a score means.
- **Agent benchmarks**: in April 2026 Berkeley researchers reported an automated agent that reached near-perfect scores on eight agent benchmarks, including 100% on SWE-bench Verified and Terminal-Bench, without solving any task, by exploiting the graders (for example, a pytest hook that forces tests to pass, and prompt injection against LLM judges) ([Wang et al., Berkeley RDI, 2026](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)). L3-07 covers SWE-bench's contamination and weak tests.

**Responses.** *Harder static sets* (GPQA, HLE, FrontierMath) buy time. *Dynamic* benchmarks keep collecting new items: Dynabench has humans write examples that fool current models ([Kiela et al., 2021](https://aclanthology.org/2021.naacl-main.324/), NAACL), and "live" benchmarks draw on material published after models' training cutoffs. *Private* held-out sets limit contamination but ask you to trust the holder. None of these replaces an eval built for your own decision.

## 5. Contamination

**Contamination** means test items, or their answers, were in the training data. The model can then score by recall rather than by the skill the benchmark claims to measure. For web-scale pretraining it is the default, not an accident: benchmarks are published on the web, discussed, copied into tutorials and translated.

**Detection by n-gram overlap.** GPT-3's authors searched for 13-gram overlaps between test sets and the training data, reported scores on "clean" subsets with the overlapping items removed, and noted that a bug in their filtering caused some overlaps to be missed ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)). Script 4 shows the trade-off in miniature. A 13-gram check catches all 30 verbatim leaks but only 5 of 30 paraphrased ones; shorter n-grams catch paraphrases but flag clean items that merely share template text (at $n = 8$, 214 of 240 clean items are flagged). The "13-gram clean" subset then still contains 25 leaked items, and the model scores 0.641 on it against a true skill of 0.600. A clean subset bounds contamination only as well as the detector does.

**Detection without the training data.** Most model developers do not publish their training data. Oren et al. observe that if a benchmark's items are exchangeable, a model that never saw it should assign the same likelihood to every ordering of the items, while a model trained on the published file will prefer the canonical order; this gives a test with a p-value that needs only the model's log-probabilities ([Oren et al., 2024](https://arxiv.org/abs/2310.17623), "Proving Test Set Contamination in Black-Box Language Models", ICLR). It detects verbatim, ordered contamination, not paraphrases.

**Detection by fresh items.** The most convincing check is a new test set built the same way. Zhang et al. wrote GSM1k, a new set of grade-school maths problems designed to match GSM8k's style and difficulty, and found accuracy drops of up to 8%, with some model families showing signs of having memorized GSM8k, and a positive relationship (Spearman's $r^2 = 0.36$) between how likely a model is to generate GSM8k examples and its GSM8k-GSM1k gap. Many frontier models showed little sign of overfitting [Zhang et al., 2024]. Script 4's last line is this test: with new numbers and names, the memorizer falls to its true 0.600.

Sainz et al. argue that contamination should be measured and reported for each benchmark and model, rather than assumed away ([Sainz et al., 2023](https://aclanthology.org/2023.findings-emnlp.722/), Findings of EMNLP). In your own evals, write new items, keep them off the public web, and record when each item was written.

## 6. Human evaluation and agreement

For open-ended outputs, humans are often the reference standard, and they disagree. Before using human labels as ground truth, measure how much.

**Raw agreement misleads.** If 90% of responses are acceptable, two annotators who both say "acceptable" to everything agree 90% or more of the time and convey nothing. **Cohen's kappa** corrects for the agreement expected from each annotator's label frequencies:

$$\kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_e = \sum_k p^{(1)}_k\, p^{(2)}_k,$$

with $p_o$ the observed agreement and $p^{(j)}_k$ the fraction of items annotator $j$ gives label $k$. **Krippendorff's alpha** generalizes this to any number of annotators, missing ratings and other label types: $\alpha = 1 - D_o / D_e$, the ratio of observed to expected disagreement, where the expected disagreement pools all annotators' labels. Artstein and Poesio's survey explains the coefficients, their assumptions and how to interpret them in computational linguistics [Artstein and Poesio, 2008]. (`evalkit.py` implements both; I checked them against scikit-learn's `cohen_kappa_score` and the `krippendorff` package on random data, and they agreed to 12 or more decimal places.)

Script 3 shows why the chance correction matters. Two annotators who are each 90% accurate, on a set where 90% of responses are acceptable, agree 79.7% of the time but have $\kappa = 0.339$. On a balanced set the same accuracy gives $\kappa = 0.665$. An annotator who ignores the item and says "ok" 97% of the time agrees 79.0% of the time with a careful colleague, which looks almost as good, while $\kappa = -0.054$ exposes it. This prevalence effect is a known property of kappa, not a bug: report the raw agreement, the class balance and the chance-corrected coefficient together.

**What low agreement means.** It can mean careless annotators, an ambiguous guideline, or a task where reasonable people disagree. The fixes differ: training and attention checks; rewriting the guideline with worked examples of edge cases; or treating the disagreement as data (report the distribution of labels, not a forced majority). In every case, the agreement between humans is a ceiling on how well any automatic judge can be shown to agree with "humans".

**Pairwise preferences at scale.** Chatbot Arena (later LMArena, and since January 2026 branded Arena, according to its Wikipedia entry) collects anonymous side-by-side votes from users and fits a Bradley-Terry model, the same model used for reward models in L3-02 ([Chiang et al., 2024](https://arxiv.org/abs/2403.04132), ICML). Crowd votes measure what those users prefer on the prompts they choose to type, which is a real but narrow construct. Singh et al. documented practices that distort the leaderboard: providers testing many private variants and publishing only the best (they report 27 private variants tested by Meta before the Llama 4 release), and unequal access to the data, with Google and OpenAI receiving an estimated 19.2% and 20.4% of all Arena data against 29.7% for 83 open-weight models combined [Singh et al., 2025]. Testing many variants and reporting the best is section 3's winner's curse. As of March 2026, the AI Index reports the top four developers within 25 Elo points of each other on Arena, so ranking differences at the top are small.

## 7. LLM-as-judge and its biases

Human labels are slow and expensive, so models are increasingly used to grade other models: scoring a single answer on a rubric, or choosing the better of two. Zheng et al. found that GPT-4 as a judge agreed with human preferences over 80% of the time on MT-Bench and Chatbot Arena data, the same level as agreement between humans, and named three biases: **position**, **verbosity** and **self-enhancement** [Zheng et al., 2023]. Each has since been studied directly:

- **Position bias.** Wang et al. showed that with ChatGPT as the judge, Vicuna-13B could beat ChatGPT on 66 of 80 queries just by changing the order in which the two answers were shown. They proposed evaluating both orders (balanced position calibration), having the judge produce evidence before its verdict, and sending low-confidence cases to humans [Wang et al., 2024].
- **Length bias.** Saito et al. found GPT-4 preferring longer answers more than humans do ([Saito et al., 2023](https://arxiv.org/abs/2310.10076)). Length-controlled AlpacaEval fits a generalized linear model of the judge's preference that includes the length difference, then reports the win rate predicted with that difference set to zero; this raised the Spearman correlation of the leaderboard with Chatbot Arena from 0.94 to 0.98 [Dubois et al., 2024].
- **Self-preference.** Panickssery et al. found that LLM evaluators can recognize their own outputs and that the strength of self-preference correlates linearly with that self-recognition ability ([Panickssery et al., NeurIPS 2024](https://arxiv.org/abs/2404.13076)). Do not let a model family judge itself against competitors without a check.
- **Manipulability.** A judge reads the output it grades, so the output can contain instructions to the judge. The Berkeley study in section 4 used exactly this against LLM-judged agent benchmarks.

Script 3 simulates a judge with known biases: it sees quality, adds a bonus of 0.6 logits for the answer shown first, and likes length. Model X is slightly worse than the reference (humans prefer it 49.3% of the time) but writes 40% longer answers. The judge gives X 62.8% when X is shown first and 50.7% when shown second; the verdict flips on 22.2% of prompts when the order is swapped. Averaging both orders removes the position effect but leaves the length effect (56.8%, interval [0.532, 0.603], which excludes the human value). Length control brings it to 50.6%. The fitted length coefficient (0.66) is smaller than the 1.5 in the simulated judge because the regression omits the quality difference, which shrinks logistic coefficients; the prediction at zero length difference still works because quality and length are independent in this part.

Part 4 of script 3 shows the limit of length control. When longer answers are genuinely better (humans prefer X 60.3% of the time), length control removes real signal along with the bias and reports 51.3%. Any "control" for a variable assumes the variable should not matter; decide that on purpose.

**How to use a judge.** Write a specific rubric, and ask for reasoning before the verdict. Evaluate pairwise comparisons in both orders. Use a judge from a different model family from the models being compared, or several judges. Above all, **validate it**: label a few hundred items with humans, report judge-human agreement next to human-human agreement, and recheck when you change the judge, the rubric or the kind of outputs. In script 3 the judge agrees with one human 71.8% of the time ($\kappa = 0.439$) against 74.2% ($\kappa = 0.483$) for two humans; "as good as a human annotator" is what that means, and it is a modest bar when humans agree this little.

## 8. Red flags in published claims

When you read "model X achieves state of the art on Y", ask:

1. **What exactly was run?** Prompt, few-shot examples, decoding, number of samples, tools, answer extraction. Without these the number cannot be reproduced (section 2).
2. **Is there an interval, and what does it cover?** Item sampling only, or also run-to-run randomness? Clustered where items are grouped? A one-point gain on a 500-item set without an interval is not evidence.
3. **Paired with the baseline, on the same items and harness?** Comparing your number with a number copied from another paper mixes harness differences into the gain.
4. **How many things were tried?** Prompts, checkpoints, seeds, benchmarks. Reporting the best of many is selection, and a table of only favourable benchmarks is selection too.
5. **Is the benchmark saturated or noisy?** Compare the gap with the estimated label-error rate.
6. **Could it be contaminated?** Was the benchmark public before the training cutoff? Is there a fresh-item or perturbation check?
7. **Who graded?** If an LLM judge, which one, validated how, and is it related to the model being evaluated?
8. **Does the metric match the claim?** "Reasoning", "understanding" and "general" are construct claims; a multiple-choice score supports a narrower one.
9. **Who built and funds the benchmark, and who has access to the test set?**
10. **Is it a pass@k or best-of-n result presented as typical behavior?** (L3-07.)

None of these makes a result false. Each is a reason to widen your own error bars.

## Hands-on

The scripts are in `code/L4-03-evaluation-and-benchmarks/`. Run them from the repository root, for example `.venv/bin/python code/L4-03-evaluation-and-benchmarks/01_harness.py`. They need NumPy, SciPy and scikit-learn, download nothing, call no model API, and each ran in under 3 seconds here. All data and "models" are simulated so that the true values are known, which is what lets the scripts check whether an interval covers the truth. The outputs below are from NumPy 2.5.4, SciPy 1.18.1 and scikit-learn 1.9.1 on Linux; random streams can differ across NumPy versions.

### The toolkit, `evalkit.py` (imported by the scripts; prints nothing)

```python
"""Small statistics toolkit for evaluations (imported by the scripts; prints nothing).

Conventions: `scores` is a 1-D array with one number per eval item (1/0 for
correct/incorrect, or a mean over several samples of the same item).
`clusters` is an optional array of the same length giving each item's group
(for example, several questions written about the same passage).
"""
import numpy as np
from scipy.stats import norm


def mean_se(scores):
    """Standard error of the mean from the central limit theorem."""
    s = np.asarray(scores, float)
    return s.std(ddof=1) / np.sqrt(len(s))


def clustered_se(scores, clusters):
    """Cluster-robust standard error of the mean: residuals are summed within
    each cluster before squaring, so correlated items count once, not n times."""
    s = np.asarray(scores, float)
    resid = s - s.mean()
    sums = np.array([resid[clusters == c].sum() for c in np.unique(clusters)])
    G = len(sums)
    return np.sqrt(G / (G - 1) * (sums ** 2).sum()) / len(s)


def bootstrap_ci(scores, clusters=None, n_boot=4000, level=0.95, seed=0):
    """Percentile bootstrap interval for the mean. With clusters, resample
    whole clusters (the unit that was sampled), not individual items."""
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, float)
    if clusters is None:
        idx = rng.integers(len(s), size=(n_boot, len(s)))
        stats = s[idx].mean(1)
    else:
        groups = [s[clusters == c] for c in np.unique(clusters)]
        sums = np.array([g.sum() for g in groups])
        sizes = np.array([len(g) for g in groups])
        pick = rng.integers(len(groups), size=(n_boot, len(groups)))
        stats = sums[pick].sum(1) / sizes[pick].sum(1)
    a = (1 - level) / 2
    return np.quantile(stats, [a, 1 - a])


def paired_permutation_p(a, b, clusters=None, n_perm=10000, seed=0):
    """Two-sided p-value for 'A and B have the same mean score' on the same
    items. Under that null, swapping A's and B's result on an item (or on a
    whole cluster) is equally likely, so we flip the sign of differences."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    if clusters is not None:
        d = np.array([d[clusters == c].sum() for c in np.unique(clusters)])
    obs = abs(d.sum())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return (np.abs((signs * d).sum(1)) >= obs - 1e-12).mean()


def power_paired(sd_diff, delta, n, alpha=0.05):
    """Approximate power of a two-sided paired z-test to detect a true mean
    difference `delta` with `n` items, given the per-item sd of differences."""
    z = norm.ppf(1 - alpha / 2)
    return norm.cdf(abs(delta) * np.sqrt(n) / sd_diff - z)


def n_for_power(sd_diff, delta, power=0.8, alpha=0.05):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil((z * sd_diff / delta) ** 2))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order]).clip(max=1)
    out = np.empty(m)
    out[order] = adj
    return out


def benjamini_hochberg(pvals):
    """BH adjusted p-values (controls the false discovery rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = (p[order] * m / np.arange(1, m + 1))[::-1]
    adj = np.minimum.accumulate(adj)[::-1].clip(max=1)
    out = np.empty(m)
    out[order] = adj
    return out


def cohen_kappa(r1, r2):
    """Agreement between two raters beyond what their label frequencies
    would produce by chance: (p_o - p_e) / (1 - p_e)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    cats = np.union1d(r1, r2)
    p_o = (r1 == r2).mean()
    p_e = sum((r1 == c).mean() * (r2 == c).mean() for c in cats)
    return (p_o - p_e) / (1 - p_e)


def krippendorff_alpha_nominal(ratings):
    """Krippendorff's alpha for nominal labels. `ratings` is (raters, items)
    with np.nan for missing ratings. alpha = 1 - observed / expected disagreement."""
    R = np.asarray(ratings, float)
    cats = np.unique(R[~np.isnan(R)])
    # Coincidence matrix: within each item, every ordered pair of ratings from
    # different raters, weighted by 1 / (m_u - 1).
    o = np.zeros((len(cats), len(cats)))
    for u in range(R.shape[1]):
        vals = R[~np.isnan(R[:, u]), u]
        m = len(vals)
        if m < 2:
            continue
        counts = np.array([(vals == c).sum() for c in cats])
        o += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    d_o = (o.sum() - np.trace(o)) / n
    d_e = (n ** 2 - (n_c ** 2).sum()) / (n * (n - 1))
    return 1 - d_o / d_e
```

### Script 1: an eval harness with honest error bars, `01_harness.py`

```python
"""A minimal offline eval harness: items, models, a grader, and honest statistics.

The "models" are simulated so that we know the truth: each has a skill, each
item a difficulty, and each passage (a cluster of 6 questions written about
the same scenario) a shared difficulty. Model B is slightly more accurate
than A but phrases its answers more verbosely.
"""
import re

import numpy as np

from evalkit import (bootstrap_ci, clustered_se, mean_se, n_for_power,
                     paired_permutation_p)

rng = np.random.default_rng(0)
N_PASSAGES, Q_PER = 40, 6
PASSAGE_SD, ITEM_SD = 1.0, 2.0          # spread of difficulty, in logits


def make_items(n_passages, rng):
    """Arithmetic word problems; questions about one passage share its difficulty."""
    items = []
    for p in range(n_passages):
        p_diff = rng.normal(0, PASSAGE_SD)
        for q in range(Q_PER):
            a, b = rng.integers(2, 30, size=2)
            c = rng.integers(1, a * b)
            items.append(dict(
                passage=p,
                question=f"A crate holds {a} boxes of {b} pens; {c} pens are lost. How many remain?",
                answer=int(a * b - c),
                difficulty=p_diff + rng.normal(0, ITEM_SD)))
    return items


class SimulatedModel:
    def __init__(self, name, skill, formats):
        self.name, self.skill, self.formats = name, skill, formats

    def p_correct(self, item):
        return 1 / (1 + np.exp(-(self.skill - item["difficulty"])))

    def answer(self, item, rng):
        """One sampled response: right with probability p_correct, else off by a bit."""
        value = item["answer"]
        if rng.random() > self.p_correct(item):
            value += int(rng.choice([-10, -2, -1, 1, 2, 10]))
        return rng.choice(self.formats).format(value)


def grade_strict(response, item):
    return float(response.strip() == str(item["answer"]))


def grade_normalized(response, item):
    """Take the last number in the response; compare numerically."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return float(bool(nums) and float(nums[-1]) == item["answer"])


A = SimulatedModel("A", skill=1.0, formats=["{}", "{}", "{}", "{}."])
B = SimulatedModel("B", skill=1.4, formats=["{}", "The answer is {}.", "{} pens"])

items = make_items(N_PASSAGES, rng)
clusters = np.array([it["passage"] for it in items])
print(f"{len(items)} items in {N_PASSAGES} passages; example: {items[0]['question']} -> {items[0]['answer']}")

# ---- 1. The grader is part of the metric ------------------------------------
resp = {m.name: [m.answer(it, rng) for it in items] for m in (A, B)}
print("\n1. accuracy, one sample per item")
for m in (A, B):
    strict = np.mean([grade_strict(r, it) for r, it in zip(resp[m.name], items)])
    norm_ = np.mean([grade_normalized(r, it) for r, it in zip(resp[m.name], items)])
    print(f"   model {m.name}: strict match {strict:.3f}   normalized {norm_:.3f}   "
          f"responses look like {sorted(set(resp[m.name][:8]), key=len)[-1]!r}")

score = {m.name: np.array([grade_normalized(r, it) for r, it in zip(resp[m.name], items)])
         for m in (A, B)}

# ---- 2. Standard errors: items are not independent ---------------------------
s = score["A"]
print("\n2. model A, normalized accuracy {:.3f}".format(s.mean()))
print(f"   naive SE {mean_se(s):.4f}   clustered SE {clustered_se(s, clusters):.4f}")
print("   95% bootstrap CI, resampling items:    [{:.3f}, {:.3f}]".format(*bootstrap_ci(s)))
print("   95% bootstrap CI, resampling passages: [{:.3f}, {:.3f}]".format(*bootstrap_ci(s, clusters)))

# ---- 3. Several samples per item shrink only the sampling part ----------------
K = 10
multi = np.array([[grade_normalized(A.answer(it, rng), it) for _ in range(K)] for it in items])
print(f"\n3. model A, mean of {K} samples per item: accuracy {multi.mean():.3f}")
print(f"   clustered SE with 1 sample {clustered_se(multi[:, 0], clusters):.4f}, "
      f"with {K} samples {clustered_se(multi.mean(1), clusters):.4f}")

# ---- 4. Comparing A and B on the same items ------------------------------------
d = score["B"] - score["A"]
unpaired = np.sqrt(mean_se(score["A"]) ** 2 + mean_se(score["B"]) ** 2)
print(f"\n4. B - A = {d.mean():+.3f}")
print(f"   SE if treated as independent samples {unpaired:.4f}; paired SE {mean_se(d):.4f}; "
      f"paired clustered SE {clustered_se(d, clusters):.4f}")
print(f"   paired permutation p: items {paired_permutation_p(score['B'], score['A']):.3f}, "
      f"passages {paired_permutation_p(score['B'], score['A'], clusters):.3f}")
true_gap = np.mean([B.p_correct(it) - A.p_correct(it) for it in make_items(5000, rng)])
print(f"   true expected gap on this item distribution (from 30,000 fresh items): {true_gap:+.3f}")
sd_d = d.std(ddof=1) * clustered_se(d, clusters) / mean_se(d)   # inflate for clustering
print(f"   items needed for 80% power at that gap: about {n_for_power(sd_d, true_gap):,}")

# ---- 5. Do the intervals cover the truth? -----------------------------------------
# Repeat the whole experiment on 1,000 fresh eval sets (model A, one sample per
# item). The truth is A's expected accuracy over the item distribution.
truth = np.mean([A.p_correct(it) for it in make_items(5000, rng)])
cover_naive = cover_clust = 0
for rep in range(1000):
    its = make_items(N_PASSAGES, rng)
    cl = np.array([it["passage"] for it in its])
    sc = np.array([float(rng.random() < A.p_correct(it)) for it in its])
    cover_naive += abs(sc.mean() - truth) < 1.96 * mean_se(sc)
    cover_clust += abs(sc.mean() - truth) < 1.96 * clustered_se(sc, cl)
print(f"\n5. coverage of nominal 95% intervals over 1,000 fresh eval sets "
      f"(true accuracy {truth:.3f}):")
print(f"   naive {cover_naive / 1000:.3f}   clustered {cover_clust / 1000:.3f}")
```

Output:

```text
240 items in 40 passages; example: A crate holds 16 boxes of 9 pens; 45 pens are lost. How many remain? -> 99

1. accuracy, one sample per item
   model A: strict match 0.479   normalized 0.667   responses look like '126.'
   model B: strict match 0.233   normalized 0.721   responses look like 'The answer is 160.'

2. model A, normalized accuracy 0.667
   naive SE 0.0305   clustered SE 0.0368
   95% bootstrap CI, resampling items:    [0.608, 0.725]
   95% bootstrap CI, resampling passages: [0.596, 0.738]

3. model A, mean of 10 samples per item: accuracy 0.654
   clustered SE with 1 sample 0.0345, with 10 samples 0.0304

4. B - A = +0.054
   SE if treated as independent samples 0.0421; paired SE 0.0335; paired clustered SE 0.0329
   paired permutation p: items 0.134, passages 0.137
   true expected gap on this item distribution (from 30,000 fresh items): +0.052
   items needed for 80% power at that gap: about 768

5. coverage of nominal 95% intervals over 1,000 fresh eval sets (true accuracy 0.637):
   naive 0.897   clustered 0.943
```

Part 1: the grader reverses the ranking. Part 2: clustering widens the interval by about a fifth here; with more questions per passage or stronger passage effects it widens more. Part 3: resampling helps only with the sampling part of the variance. Part 4: B is truly better by 5.2 points, the observed gap is about right, and the test still cannot detect it. Part 5 is the reason to use the clustered SE: the naive interval undercovers. Note also that the accuracy in part 2 (0.667) differs from the population value (0.637) by almost one clustered SE, as expected for one draw.

### Script 2: reusing the test set, `02_reusing_the_test_set.py`

```python
"""Two ways that looking at the test set many times manufactures progress.

1. Twenty prompt variants with exactly the same true accuracy: pick the best
   on the test set, then see what corrections and a fresh test set say.
2. A leaderboard attack in the style of Blum and Hardt (2015): random guesses,
   kept or dropped according to their public score, then combined.
"""
import numpy as np
from scipy.stats import norm

from evalkit import benjamini_hochberg, holm, paired_permutation_p

rng = np.random.default_rng(1)

# ---- 1. selecting the best of 20 equal variants ------------------------------
N_ITEMS, N_VARIANTS = 300, 20
p_item = 1 / (1 + np.exp(-rng.normal(1.0, 2.0, size=N_ITEMS)))   # true per-item accuracy


def run_variants(p, rng):
    """Variant 0 is the baseline prompt. Every variant agrees with the baseline
    on a random 70% of items and is redrawn on the rest: correlated results,
    identical expected accuracy."""
    base = rng.random(len(p)) < p
    out = [base]
    for _ in range(N_VARIANTS):
        redraw = rng.random(len(p)) < 0.3
        out.append(np.where(redraw, rng.random(len(p)) < p, base))
    return np.array(out, float)


res = run_variants(p_item, rng)
acc = res.mean(1)
pvals = np.array([paired_permutation_p(res[v], res[0], n_perm=4000, seed=v)
                  for v in range(1, N_VARIANTS + 1)])
best = 1 + int(np.argmax(acc[1:]))
print(f"1. true accuracy of every variant: {p_item.mean():.3f}")
print(f"   baseline on test set {acc[0]:.3f}; best of {N_VARIANTS} variants (#{best}) {acc[best]:.3f}")
print(f"   best variant p-value {pvals[best - 1]:.4f}; smallest p {pvals.min():.4f}; "
      f"variants with p < 0.05: {(pvals < 0.05).sum()}")
print(f"   after Holm: {(holm(pvals) < 0.05).sum()} significant; "
      f"after Benjamini-Hochberg: {(benjamini_hochberg(pvals) < 0.05).sum()}")

# Fresh items from the same distribution, and fresh runs of every variant.
fresh = run_variants(1 / (1 + np.exp(-rng.normal(1.0, 2.0, size=N_ITEMS))), rng)
print(f"   on a fresh test set: baseline {fresh[0].mean():.3f}, chosen variant {fresh[best].mean():.3f}")

# How big is the winner's curse on average? Repeat the whole selection 500 times.
# (A paired z-test, sum(d) / sqrt(sum(d^2)), stands in for the slower permutation test.)
gains, any_raw, any_holm = [], 0, 0
for _ in range(500):
    r = run_variants(p_item, rng)
    a = r.mean(1)
    gains.append(a[1:].max() - a[0])
    d = r[1:] - r[0]
    z = d.sum(1) / np.sqrt((d ** 2).sum(1)).clip(min=1)
    p = 2 * norm.sf(np.abs(z))
    any_raw += (p < 0.05).any()
    any_holm += (holm(p) < 0.05).any()
print(f"   over 500 repeats, best-minus-baseline averaged {np.mean(gains):+.3f} "
      f"(true gain 0); 90th percentile {np.quantile(gains, 0.9):+.3f}")
print(f"   repeats with at least one 'significant' variant: uncorrected {any_raw / 500:.3f}, "
      f"Holm {any_holm / 500:.3f}")

# ---- 2. climbing a public leaderboard with no skill ---------------------------
N_TEST, N_SUB = 4000, 400
labels = rng.integers(0, 2, N_TEST)
public = np.arange(N_TEST) < N_TEST // 2            # half the test set is scored publicly
subs = rng.integers(0, 2, (N_SUB, N_TEST))          # random guesses: true accuracy 0.5
pub_score = (subs[:, public] == labels[public]).mean(1)
keep = subs[pub_score > 0.5]                        # keep guesses that "did well"
combined = (keep.mean(0) > 0.5).astype(int)         # majority vote of the keepers
print(f"\n2. {N_SUB} random submissions, kept {len(keep)} that scored above 0.5 publicly")
print(f"   best single submission: public {pub_score.max():.3f}")
print(f"   majority vote of kept:  public {(combined[public] == labels[public]).mean():.3f}, "
      f"private {(combined[~public] == labels[~public]).mean():.3f}")
```

Output:

```text
1. true accuracy of every variant: 0.627
   baseline on test set 0.613; best of 20 variants (#11) 0.637
   best variant p-value 0.2910; smallest p 0.0970; variants with p < 0.05: 0
   after Holm: 0 significant; after Benjamini-Hochberg: 0
   on a fresh test set: baseline 0.683, chosen variant 0.667
   over 500 repeats, best-minus-baseline averaged +0.030 (true gain 0); 90th percentile +0.043
   repeats with at least one 'significant' variant: uncorrected 0.672, Holm 0.046

2. 400 random submissions, kept 209 that scored above 0.5 publicly
   best single submission: public 0.534
   majority vote of kept:  public 0.597, private 0.473
```

In this particular draw no variant was nominally significant, but across repeats one usually is (67.2%), and the best one always looks better than the baseline. On the fresh test set the baseline happened to score higher than the "winner". Part 2 shows a public score climbing from 0.5 to almost 0.6 with no skill at all.

### Script 3: annotators and judges, `03_humans_and_judges.py`

```python
"""Annotator agreement, and the biases of a (simulated) LLM judge.

Nothing here calls a real model. The judge is a formula with known position
and length biases, so we can check whether standard diagnostics find them.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from evalkit import bootstrap_ci, cohen_kappa, krippendorff_alpha_nominal

rng = np.random.default_rng(2)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# ---- 1. Agreement between human annotators -----------------------------------
# 400 responses; 90% are truly acceptable. Each annotator labels correctly with
# probability 0.9, except annotator 3, who barely reads and says "ok" 97% of the time.
N = 400
truth = (rng.random(N) < 0.9).astype(int)
noisy = lambda acc: np.where(rng.random(N) < acc, truth, 1 - truth)
ann = np.array([noisy(0.9), noisy(0.9), (rng.random(N) < 0.97).astype(int)])  # 3 ignores the item
print("1. three annotators, binary acceptable / not, 90% truly acceptable")
for i, j in [(0, 1), (0, 2), (1, 2)]:
    print(f"   annotators {i + 1} & {j + 1}: raw agreement {(ann[i] == ann[j]).mean():.3f}, "
          f"Cohen's kappa {cohen_kappa(ann[i], ann[j]):.3f}")
print(f"   Krippendorff's alpha, all three: {krippendorff_alpha_nominal(ann):.3f}")
# Missing data is normal: each item labeled by only two of the three.
sparse = ann.astype(float)
sparse[rng.integers(0, 3, N), np.arange(N)] = np.nan
print(f"   alpha with one random label missing per item: {krippendorff_alpha_nominal(sparse):.3f}")
# The same raters on a balanced set: same accuracy, very different kappa.
truth_b = (rng.random(N) < 0.5).astype(int)
b1, b2 = [np.where(rng.random(N) < 0.9, truth_b, 1 - truth_b) for _ in range(2)]
print(f"   balanced set, two 90%-accurate raters: raw {(b1 == b2).mean():.3f}, "
      f"kappa {cohen_kappa(b1, b2):.3f}")

# ---- 2. A pairwise judge with position and length bias -------------------------
# 600 prompts. Model X writes answers 40% longer than the reference model R and
# is slightly worse on true quality. Humans judge quality only.
P = 600
q_x, q_r = rng.normal(-0.15, 1, P), rng.normal(0, 1, P)
len_x, len_r = rng.normal(1.4, 0.3, P), rng.normal(1.0, 0.3, P)   # in units of R's mean length


def human_prefers_first(qa, qb):
    return rng.random(len(qa)) < sigmoid(2.0 * (qa - qb))


def judge_prefers_first(qa, qb, la, lb, pos_bias=0.6, len_bias=1.5):
    """Sees quality (a bit noisily), likes longer answers, and likes whatever is shown first."""
    z = 2.0 * (qa - qb) + len_bias * (la - lb) + pos_bias
    return rng.random(len(qa)) < sigmoid(z)


human = human_prefers_first(q_x, q_r).astype(float)
j_xfirst = judge_prefers_first(q_x, q_r, len_x, len_r).astype(float)        # X shown first
j_rfirst = 1 - judge_prefers_first(q_r, q_x, len_r, len_x).astype(float)    # R shown first
print("\n2. win rate of model X against reference R over 600 prompts")
print(f"   humans:                                   {human.mean():.3f}")
print(f"   judge, X always shown first:              {j_xfirst.mean():.3f}")
print(f"   judge, X always shown second:             {j_rfirst.mean():.3f}")
both = (j_xfirst + j_rfirst) / 2
print(f"   judge, both orders averaged:              {both.mean():.3f}  "
      "95% CI [{:.3f}, {:.3f}]".format(*bootstrap_ci(both)))
print(f"   verdict flips when order is swapped:      {(j_xfirst != j_rfirst).mean():.3f} of prompts")

# Length control in the spirit of length-controlled AlpacaEval: model the
# judge's verdict as logistic in the length difference, then predict the win
# rate the judge would give if the lengths were equal.
dlen = np.concatenate([len_x - len_r, len_x - len_r])
y = np.concatenate([j_xfirst, j_rfirst])
pos = np.concatenate([np.ones(P), -np.ones(P)])                # +1: X shown first
lr = LogisticRegression(C=1e6).fit(np.column_stack([dlen, pos]), y)
lc = sigmoid(lr.intercept_[0])                                 # dlen = 0, positions balanced
print(f"   length-controlled judge win rate:         {lc:.3f}  "
      f"(fitted length coefficient {lr.coef_[0][0]:.2f})")

# ---- 3. Agreement of the judge with humans --------------------------------------
print("\n3. judge vs human labels (per prompt, both-order majority; ties broken by X-first)")
j_major = np.where(both == 0.5, j_xfirst, both > 0.5).astype(int)
h2 = human_prefers_first(q_x, q_r).astype(int)                 # a second human
print(f"   judge-human agreement {(j_major == human).mean():.3f}, kappa {cohen_kappa(j_major, human):.3f}")
print(f"   human-human agreement {(h2 == human).mean():.3f}, kappa {cohen_kappa(h2, human):.3f}")

# ---- 4. When longer really is better, length control removes real signal --------
q_x2 = q_r + 0.5 * (len_x - len_r) + rng.normal(0, 0.5, P)    # quality now grows with length
human2 = human_prefers_first(q_x2, q_r).mean()
jx = judge_prefers_first(q_x2, q_r, len_x, len_r).astype(float)
jr = 1 - judge_prefers_first(q_r, q_x2, len_r, len_x).astype(float)
lr2 = LogisticRegression(C=1e6).fit(np.column_stack([dlen, pos]), np.concatenate([jx, jr]))
print("\n4. same judge, but now longer answers are genuinely better")
print(f"   humans {human2:.3f}   judge both orders {(jx + jr).mean() / 2:.3f}   "
      f"length-controlled {sigmoid(lr2.intercept_[0]):.3f}")
```

Output:

```text
1. three annotators, binary acceptable / not, 90% truly acceptable
   annotators 1 & 2: raw agreement 0.797, Cohen's kappa 0.339
   annotators 1 & 3: raw agreement 0.790, Cohen's kappa -0.054
   annotators 2 & 3: raw agreement 0.782, Cohen's kappa -0.009
   Krippendorff's alpha, all three: 0.106
   alpha with one random label missing per item: 0.059
   balanced set, two 90%-accurate raters: raw 0.833, kappa 0.665

2. win rate of model X against reference R over 600 prompts
   humans:                                   0.493
   judge, X always shown first:              0.628
   judge, X always shown second:             0.507
   judge, both orders averaged:              0.568  95% CI [0.532, 0.603]
   verdict flips when order is swapped:      0.222 of prompts
   length-controlled judge win rate:         0.506  (fitted length coefficient 0.66)

3. judge vs human labels (per prompt, both-order majority; ties broken by X-first)
   judge-human agreement 0.718, kappa 0.439
   human-human agreement 0.742, kappa 0.483

4. same judge, but now longer answers are genuinely better
   humans 0.603   judge both orders 0.664   length-controlled 0.513
```

Alpha falls when one label per item is removed (0.106 to 0.059) partly because the remaining pairs more often include the inattentive annotator; it is computed from fewer pairs and is noisier. The judge results are discussed in section 7.

### Script 4: contamination, `04_contamination.py`

```python
"""Contamination: what n-gram overlap checks catch, what they miss, and how a
leak inflates scores even on a "decontaminated" subset.

A 300-item benchmark of templated word problems; a synthetic pretraining
corpus into which some items leak verbatim and some leak paraphrased.
"""
import re

import numpy as np

rng = np.random.default_rng(4)
NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun"]
THINGS = ["apples", "stamps", "marbles", "coins", "shells", "cards", "beads", "pencils"]
TEMPLATES = [
    "{n} has {a} {t} and buys {b} more each week for {c} weeks. How many {t} does {n} have now?",
    "{n} shares {a} {t} equally among {b} friends and keeps the {c} left over. How many does each friend get?",
    "A shelf holds {a} boxes with {b} {t} in each box, and {n} removes {c} {t}. How many {t} remain on the shelf?",
]
SYNONYMS = {"has": "owns", "buys": "gets", "more": "extra", "each": "every", "week": "weekly batch",
            "How": "Tell me how", "many": "much", "now": "at the end", "holds": "carries",
            "remain": "are left", "shares": "splits", "friends": "pals", "keeps": "holds on to"}


def make_item(rng):
    k = rng.integers(len(TEMPLATES))
    a, b, c = (int(v) for v in rng.integers(2, 40, size=3))
    n, t = rng.choice(NAMES), rng.choice(THINGS)
    if k == 1:
        a, c = a * b + c % b, c % b          # make the division come out even
    ans = [a + b * c, (a - c) // b, a * b - c][k]
    return dict(text=TEMPLATES[k].format(n=n, a=a, b=b, c=c, t=t), answer=ans)


def paraphrase(text):
    """Word substitutions plus moving the question to the front. Same meaning,
    different surface form."""
    words = [SYNONYMS.get(w, w) for w in text.split()]
    s = " ".join(words)
    body, q = s.split(". ", 1) if ". " in s else (s, "")
    return f"{q} Context: {body}."


bench = [make_item(rng) for _ in range(300)]
leak_verbatim = set(range(0, 30))      # items 0-29 appear word for word in the corpus
leak_para = set(range(30, 60))         # items 30-59 appear paraphrased

FILLER = ("the of and to in is was for on that with as by at from it this be are or "
          "an which one all were we when there can more if will their has would time "
          "people who about other into some could them see only new these two may").split()


def filler(n_words):
    return " ".join(rng.choice(FILLER, n_words))


corpus = [filler(rng.integers(50, 200)) for _ in range(2000)]
for i in leak_verbatim:
    corpus.append(f"{filler(30)} Q: {bench[i]['text']} A: {bench[i]['answer']} {filler(30)}")
for i in leak_para:
    corpus.append(f"{filler(30)} {paraphrase(bench[i]['text'])} Answer {bench[i]['answer']} {filler(30)}")
# Clean items still share template text with many web pages about word problems.
for _ in range(40):
    corpus.append(f"{filler(20)} {make_item(rng)['text']} {filler(20)}")


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def ngrams(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def flagged(n):
    """Items sharing at least one n-gram with any corpus document."""
    corpus_grams = set().union(*(ngrams(tokens(d), n) for d in corpus))
    return {i for i, it in enumerate(bench) if ngrams(tokens(it["text"]), n) & corpus_grams}


print("1. n-gram overlap detection (300 items: 30 leaked verbatim, 30 paraphrased, 240 clean)")
print("    n   verbatim caught   paraphrased caught   clean items flagged")
for n in [5, 8, 13]:
    f = flagged(n)
    clean = set(range(300)) - leak_verbatim - leak_para
    print(f"   {n:2d}   {len(f & leak_verbatim):9d}/30   {len(f & leak_para):12d}/30   "
          f"{len(f & clean):14d}/240")

# ---- 2. What the leak does to scores -------------------------------------------
# A model with genuine accuracy 0.60 that recalls the stored answer 90% of the
# time when it has seen the item (in either form) in training. We use expected
# scores rather than sampled ones, so the numbers below carry no sampling noise.
seen = leak_verbatim | leak_para


def expected_score(item_ids, perturbed=False):
    p_seen = 0.6 if perturbed else 0.9 + 0.1 * 0.6
    return np.mean([p_seen if i in seen else 0.6 for i in item_ids])


clean13 = sorted(set(range(300)) - flagged(13))
print("\n2. expected scores of a model with true skill 0.60 that memorized the leaked items")
print(f"   full benchmark:                           {expected_score(range(300)):.3f}")
print(f"   '13-gram clean' subset ({len(clean13)} items):        {expected_score(clean13):.3f}"
      f"   ({len(set(clean13) & seen)} leaked items survive the filter)")
print(f"   truly clean items (240):                  {expected_score(range(60, 300)):.3f}")
print(f"   same items with new numbers and names:    {expected_score(range(300), perturbed=True):.3f}")
```

Output:

```text
1. n-gram overlap detection (300 items: 30 leaked verbatim, 30 paraphrased, 240 clean)
    n   verbatim caught   paraphrased caught   clean items flagged
    5          30/30             30/30              240/240
    8          30/30             28/30              214/240
   13          30/30              5/30               45/240

2. expected scores of a model with true skill 0.60 that memorized the leaked items
   full benchmark:                           0.672
   '13-gram clean' subset (220 items):        0.641   (25 leaked items survive the filter)
   truly clean items (240):                  0.600
   same items with new numbers and names:    0.600
```

The 13-gram check also flags 45 clean items, because the templates share long word sequences with unrelated "web pages" that use the same templates. Removing them throws away clean data and leaves most paraphrased leaks in place.

## Common mistakes

- **Reporting a score without an interval**, or with a naive interval on clustered items. Compute the clustered SE whenever items share a source, and say what the interval covers.
- **Comparing unpaired.** Two separate intervals that overlap do not show "no difference"; analyse per-item differences.
- **Building the eval before computing its power.** If the decision needs a 2-point difference and you have 200 items, you will be guessing. Do the power calculation with a pilot estimate of $\sigma_d$.
- **Tuning on the test set.** Every look at the test set while choosing prompts or checkpoints spends some of its validity. Keep a development split for iteration and a test split you touch rarely, and refresh it.
- **Trusting a grader you have not read.** Strict string matching, loose regexes and LLM judges all fail in characteristic ways. Read graded samples, especially where two graders disagree.
- **Treating a judge's verdict as ground truth** without measuring its agreement with humans and its position and length biases on your data.
- **Quoting raw annotator agreement** on an imbalanced task. Report kappa or alpha with the class balance.
- **Assuming a "decontaminated" benchmark is clean.** n-gram filters miss paraphrases and translations; fresh items are the stronger test.
- **Comparing your number with someone else's** without running both models in the same harness.

## Exercises

1. [warm-up] An eval has 400 independent items and a model scores 0.82. Compute the 95% CLT interval. How many items would halve its width?
2. [warm-up] Two annotators label 200 items, 180 positive by both, 10 negative by both, and 10 disagreements split 5 and 5. Compute raw agreement and Cohen's kappa by hand, then check with `evalkit.cohen_kappa`.
3. [core] In script 1, change `Q_PER` to 12 and `N_PASSAGES` to 20 (same 240 items). What happens to the clustered SE and to the coverage of the naive interval? Explain.
4. [core] In script 1, set `ITEM_SD` to 0.5. The paired SE gets closer to the unpaired one. Why? What does this tell you about when pairing helps most?
5. [core] Use `power_paired` to plot (or tabulate) power against $n$ for $\delta = 0.02$ and $\sigma_d = 0.4$. How many items do you need for 80% power? For 90%?
6. [core] In script 2, replace Holm with Benjamini-Hochberg in the repeat loop. How often does at least one false discovery occur now, and why is that not a contradiction of BH's guarantee?
7. [core] In script 3, set `pos_bias=0` and `len_bias=0`. Does the judge's win rate now match the human rate? How large is judge-human kappa compared with human-human kappa, and why?
8. [core] In script 4, add a third kind of leak: the item translated word by word with a substitution table into a "language" of your choice. Which $n$ catches it?
9. [stretch] Implement the paired bootstrap with cluster resampling and compare its p-value or interval with the permutation test in script 1.

### Hints and answers

1. SE $= \sqrt{0.82 \cdot 0.18 / 400} = 0.0192$, interval $0.82 \pm 0.038$. Width scales as $1/\sqrt n$, so 1,600 items halve it.
2. $p_o = 0.95$. Each annotator says positive 185/200 = 0.925 of the time, so $p_e = 0.925^2 + 0.075^2 = 0.861$, and $\kappa = (0.95 - 0.861)/(1 - 0.861) \approx 0.64$.
3. Fewer, larger clusters: in my run the clustered SE rose from 0.0368 to 0.0468 while the naive SE barely moved (0.0309), and naive coverage fell to 0.844. The effective number of independent units is now much smaller than 240.
4. In my run the paired SE was 0.0383 against 0.0398 unpaired. With a small item-difficulty spread, the models' per-item results are nearly independent coin flips, so the covariance term is small. Pairing helps most when items vary a lot in difficulty, which is typical of real benchmarks. (Naive coverage also got worse, 0.832, because the passage effect is now a larger share of the variance.)
5. From the formula, $n = ((1.96 + 0.84) \cdot 0.4 / 0.02)^2 \approx 3{,}140$ for 80% and about 4,200 for 90% (with $z_{0.9} = 1.28$).
6. Under the global null every discovery is false, and BH then controls the probability of any discovery at $\alpha$ too, so you should see a rate near 0.05 (it may differ somewhat because the tests are correlated). BH is less strict than Holm only when some nulls are false.
7. Yes, up to sampling noise (I got 0.495 against 0.493). Judge-human kappa then comes out slightly *above* human-human kappa (0.523 against 0.483 in my run), because the judge's label is a majority over two orderings, which averages away some noise that a single human label keeps. Beating a single human is not the same as being right.
8. A word-by-word substitution changes every token, so no n-gram on the original vocabulary matches; only a detector that maps back (or a fresh-item test) finds it. Translation is a documented route for contamination for exactly this reason.
9. Resample passages with replacement, recompute the mean difference, and take the share of bootstrap means on the other side of zero, doubled, as an approximate p-value. It should be close to the cluster permutation p.

## Check your understanding

1. **Why can clustered items make the naive standard error too small?** The naive formula assumes each item adds independent information. Items about the same passage or function tend to be right or wrong together, so 240 items in 40 clusters behave more like somewhere between 40 and 240 independent ones. Summing residuals within clusters before squaring counts each cluster's shared deviation once, which is what script 1's coverage check confirms.
2. **What does pairing buy you, and when does it buy little?** Pairing removes the variance due to which items happened to be drawn, which both models share. It helps most when models agree on which items are hard (high covariance) and little when their per-item results are nearly independent.
3. **A paper filters its test set by 13-gram overlap and reports the clean-subset score as uncontaminated. What is missing?** n-gram filters miss paraphrases, translations and answer-only leaks, and they can also remove clean items that share boilerplate. The clean subset is only as clean as the detector; a fresh, matched test set or a perturbation test is stronger evidence.
4. **Why can two annotators with 80% raw agreement have a kappa near zero?** If one label is very common, two annotators who mostly use it will agree often by chance. Kappa subtracts the agreement predicted from each annotator's label frequencies; if the observed agreement is no higher than that, kappa is near zero even though raw agreement is high.
5. **Your LLM judge prefers your new model 58% of the time. What do you check before believing it?** Evaluate both orders and report the flip rate; check whether the new model's answers are longer and compute a length-controlled rate (and decide whether length should count); make sure the judge is not from the same family as the model; and compare the judge's agreement with human labels on a sample to human-human agreement. Then put an interval on the 58%.

## Further reading

- **Evan Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations"** (2024, arXiv:2411.00640). https://arxiv.org/abs/2411.00640
  *Read for:* the sampling model behind eval scores, clustered standard errors, paired differences, power analysis and resampling answers, with formulas you can use directly. *Level:* intro.
- **Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, Dan Jurafsky, "With Little Power Comes Great Responsibility"** (2020, EMNLP; arXiv:2010.06595). https://arxiv.org/abs/2010.06595
  *Read for:* how to do power analysis for NLP comparisons and human evaluations, and evidence that many published comparisons are underpowered. *Level:* intermediate.
- **Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart, "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing"** (2018, ACL). https://aclanthology.org/P18-1128/
  *Read for:* which significance tests fit which metrics, and how often published NLP work skipped testing. *Level:* intermediate.
- **Inioluwa Deborah Raji, Emily M. Bender, Amandalynne Paullada, Emily Denton, Alex Hanna, "AI and the Everything in the Whole Wide World Benchmark"** (2021, NeurIPS Datasets and Benchmarks; arXiv:2111.15366). https://arxiv.org/abs/2111.15366
  *Read for:* the construct-validity critique of treating a few benchmarks as measures of general capability. *Level:* intro.
- **Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, et al. (30 authors), "Lessons from the Trenches on Reproducible Evaluation of Language Models"** (2024, arXiv:2405.14782). https://arxiv.org/abs/2405.14782
  *Read for:* how prompt format and implementation details change scores, from the maintainers of lm-evaluation-harness. *Level:* intermediate.
- **Curtis G. Northcutt, Anish Athalye, Jonas Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"** (2021, NeurIPS Datasets and Benchmarks; arXiv:2103.14749). https://arxiv.org/abs/2103.14749
  *Read for:* how common test-label errors are and how they can reorder strong models. *Level:* intro.
- **Hugh Zhang et al. (15 authors), "A Careful Examination of Large Language Model Performance on Grade School Arithmetic"** (2024, NeurIPS Datasets and Benchmarks; arXiv:2405.00332). https://arxiv.org/abs/2405.00332
  *Read for:* building a fresh matched test set (GSM1k) to measure overfitting and contamination. Note that the abstract's headline number changed between versions (13% in v1, 8% in the camera-ready). *Level:* intermediate.
- **Ron Artstein, Massimo Poesio, "Inter-Coder Agreement for Computational Linguistics"** (2008, Computational Linguistics 34(4)). https://aclanthology.org/J08-4004/
  *Read for:* kappa, alpha and their relatives, their assumptions, and how to interpret them. Open access. *Level:* intermediate.
- **Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. (13 authors), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"** (2023, NeurIPS Datasets and Benchmarks; arXiv:2306.05685). https://arxiv.org/abs/2306.05685
  *Read for:* the case for LLM judges, their agreement with humans, and the first catalogue of their biases. *Level:* intro.
- **Peiyi Wang, Lei Li, Liang Chen, et al., "Large Language Models are not Fair Evaluators"** (2024, ACL; arXiv:2305.17926). https://aclanthology.org/2024.acl-long.511/
  *Read for:* position bias measured directly, and calibration methods that reduce it. The ACL version lists 11 authors, the arXiv version 10. *Level:* intermediate.
- **Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto, "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators"** (2024, COLM; arXiv:2404.04475). https://arxiv.org/abs/2404.04475
  *Read for:* the regression approach to removing length bias from judge win rates. *Level:* intermediate.
- **Shivalika Singh, Yiyang Nan, Alex Wang, et al. (13 authors), "The Leaderboard Illusion"** (2025, arXiv:2504.20879). https://arxiv.org/abs/2504.20879
  *Read for:* how private testing and data access asymmetries distort a crowd-sourced leaderboard; read with Arena's public response. *Level:* intermediate.
