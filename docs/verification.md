---
title: "Verification notes"
---

This page says what was checked in this course, how, and what is weaker. Read it before relying on any specific fact, number or date.

## How the course was made

The modules were drafted by AI writing agents (Claude Opus), each working from a shared style guide with strict rules: look up every source before citing it, run every script and paste the real output, and leave out anything that can't be verified. A coordinating session then reviewed each report, re-ran code, and checked a sample of claims independently. **No human domain expert has reviewed the content.** Treat this as a carefully checked draft, not a peer-reviewed text.

## What was checked, and how

| Check | Result |
|---|---|
| Every script run from a clean process on a quiet machine | **162 of 162 pass.** The slowest takes 79 s (`L2-05/03_layernorm_placement.py`), over the 60 s target |
| Output on the page vs a fresh run (random sample of 20 fast scripts) | 18 match line for line. The other two differ only in timing columns (a profiler listing, and per-query milliseconds in an index benchmark; the recall figures match) |
| arXiv-linked reading entries compared with arXiv's own title for the same ID | 139 entries, 136 distinct papers. All match, apart from two cases where the entry uses the published title and arXiv uses a different preprint title (the log-sum-exp paper; the GAN paper) |
| Selected non-arXiv sources checked against Crossref (title, authors, year, venue, pages) | Moreno-Torres et al. (2012), Liu et al. (Isolation Forest, 2008), Efron (1979), and Blanchard et al. (log-sum-exp) all matched. For Eckart and Young (1936) I only confirmed that the DOI link loads |
| High-risk time-sensitive claims checked against primary or authoritative pages | See the next table |
| Sources the agents could not verify | Dropped, not kept. The agents reported dropping dozens, which is why some well-known references are absent |

### Claims I checked myself against primary or authoritative pages

| Claim (module) | Checked against | Result |
|---|---|---|
| EU Digital Omnibus on AI: Regulation (EU) 2026/1744, in force 27 July 2026, high-risk dates 2 Dec 2027 and 2 Aug 2028 (L3-10) | Several independent law-firm reports | Matches |
| US Executive Order 14434, 29 Sept 2026, "Inaugurating the Era of Super Intelligence" (L4-05) | Federal Register, FR Doc 2026-20321 | Matches |
| Colorado SB 26-189 signed 14 May 2026, effective 1 Jan 2027, attorney-general enforcement (L4-05) | Multiple law-firm reports | Matches. I did not read the bill text |
| Rentosertib phase 3 trial NCT07687459 (L3-10) | ClinicalTrials.gov record | Phase 3, Insilico Medicine, recruiting, started Sept 2026 |
| OpenAI's post on retiring SWE-bench Verified, 23 Feb 2026 (L3-07, L4-03) | Search results and the post's listing | Exists and says what the modules say |
| ARC-AGI-3 launch scores, 25 Mar 2026 (L4-03) | ARC Prize launch post | Matches (0.51%) |
| TabPFN-3.5 existence and release (L1-03) | Search results and the model card | Released Sept 2026; results are company-reported, and the module says so |
| Z-Image and Wan reports (L3-03) | arXiv listings | Both exist; the 6B-parameter figure matches. I did not verify the GPU-hours and cost figures |
| Sutton and Barto chapter references (L3-05) | Authors' draft contents | Chapters 3 and 6 confirmed; unconfirmed references to chapters 7 and 11 were removed |

## What was not checked

- **Whether a source supports the exact sentence it is cited for.** Title, authors, year and venue were checked. Whether each cited paper really says what the module says it says was checked by the writing agents when they read it, not independently by me.
- **Most non-arXiv sources** were confirmed by the writing agents (publisher pages, search records), not re-checked by me beyond the ones listed above.
- **Exercise answers.** Most were tested. A few hints are expectations that were not run. They are marked in the text where the agent flagged them (for example L1-04 hints 1, 4, 6, 7, 9 and 10; L1-05 hints 7 and 9; L2-02 hints 6 and 9; L4-04 hints 4 and 10).
- **Study-time estimates** in module headers are the writers' estimates, not measured.

## Sources that are weaker than the rest

The writing agents flagged these. Each is also marked in the module text where it matters.

**Verified only through search results or secondary pages** (the primary page was blocked or unavailable)
- Hoerl and Kennard (1970), via an archive copy and search metadata (L1-02); the two Breiman papers, via search records.
- ESL section 7.10.2 content, via secondary summaries; Zou and Hastie's "grouping effect" wording, via a seminar abstract.
- Moreno-Torres et al. and Liu et al., whose publisher pages blocked fetching (I confirmed both against Crossref).
- OpenAI's SWE-bench post and its sycophancy postmortem, via search snippets (L3-07, L3-02). The "Arena" rename in L4-03, via Wikipedia and snippets.
- The EU Digital Services Act Article 38 wording, via a copy of the Official Journal text (L3-08); the AI Act Annex I coverage of medical devices, via a secondary site (L3-10); fev-bench figures, via summarised reads of the arXiv HTML (L3-09).
- Qwen3.5 "early fusion", from its model card, presented as a claim (L3-04). Flamingo details, via a blog rather than the paper.
- The deadly triad description (L3-05), via secondary sources. ColBERT and DPR details that go beyond the abstracts the agent read (L3-06).

**Statements that are the writer's own reasoning, not from a source**
- The repeated-eigenvalue explanation for why momentum needs more steps (L0-05).
- Why class reweighting raised ROC AUC in one experiment (L1-05).
- Rules of thumb such as "3 to 10 times smaller learning rate when fine-tuning" (L2-03), which is labeled as a rule of thumb.
- "Decoder-only" as a description of GPT-style models (L2-05): the GPT-2 paper does not use that wording.

**Governance and policy status that was verified only through secondary sources** (L4-05): Korea's fine grace period, Japan's lack of penalties, the status of the Brazilian and UK bills, and whether the Council of Europe convention is in force. These change often. The FDA guidance on predetermined change-control plans is dated August 2025 on the FDA site, while law firms reported it as final in December 2024 (L3-10); the discrepancy is noted in the text.

**Numbers with known version or source differences:** the HLE top score on Scale's leaderboard does not say its tools setting (L4-03); the Mistral 7B parameter count (7.24B) rests on assumptions not stated in the paper's table (L4-02); how widely FP8 training is used is the authors' claim only (L4-02); GSM1k's "up to 8%" is from the final paper version (an earlier version said 13%).

## Things that will go out of date

Everything marked "as of October 2026" should be re-checked before you rely on it. The most time-sensitive modules, in order: **L4-05** (laws, standards, company policies), **L3-10** (regulatory status, trial status), **L3-07** (agent benchmarks), **L3-01 and L3-02** (current models and methods), **L4-03** (leaderboards and benchmark status), **L3-09** (forecasting benchmarks) and **L4-01** (tools).

## Known limitations of the code

- Scripts are small by design. Their results show the shape of an effect, not production behavior. Where a result depended on a seed, the page says so.
- Timings depend on the machine; the writers measured on a busy laptop-class CPU under WSL2.
- Some scripts were written to run from inside their module folder; some write plots to `docs/assets/` relative to the current directory.
- Several scripts download small public datasets on first use (for example California housing via scikit-learn).

## Reproducing these checks

The `tools/` folder has the scripts used: `run_all_code.sh` (run every script), `check_arxiv.py` (title cross-check), `check_outputs.py` (sample output check) and `build_indexes.py` (regenerates level pages and the bibliography).
