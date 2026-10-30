# The AI Course

A multi-level, hands-on course in AI, from the math you need to the questions the field is still arguing about. It has **32 modules in five levels**, each with runnable code, exercises with worked answers, and annotated sources that were looked up and checked as the modules were written.

## How it is built

- **You run things.** Every module has scripts (162 in total) that run on a laptop CPU in under about a minute. The output you see on the page is the output the script printed.
- **Intuition, then math, then code, then practice.** Each module explains an idea, derives what's worth deriving, implements the core, and ends on what goes wrong in practice.
- **Sources you can check.** Each module ends with 9 to 12 annotated sources (373 in all, plus many more cited inline) that were each looked up in the original venue or an authoritative page. Where a claim rests on a weaker source, the text says so.
- **Honest about limits.** Toy experiments are labeled as toys. Disagreements among experts are shown as disagreements. Time-sensitive claims say "as of October 2026".

## The five levels

| Level | Theme | Modules |
|---|---|---|
| [0. Foundations](level-0-foundations/index.md) | Math and tools everything else uses | 5 |
| [1. Classical ML](level-1-classical-ml/index.md) | Learning from data without deep networks | 5 |
| [2. Deep learning](level-2-deep-learning/index.md) | From backprop to a working GPT | 6 |
| [3. Modern AI across domains](level-3-modern-ai/index.md) | LLMs, diffusion, vision, RL, RAG, agents, recommenders, time series, science and healthcare | 10 |
| [4. Engineering and society](level-4-engineering-and-society/index.md) | MLOps, efficiency, evaluation, interpretability, safety and governance, research skills | 6 |

Also: [learning paths](learning-paths.md) for different goals, five [capstone projects](capstones.md), a combined [bibliography](bibliography.md), and the [verification notes](verification.md), which say exactly what was checked and what is weaker.

## Setup

```bash
git clone https://github.com/sankarsh98/ai-course.git
cd ai-course
uv venv
uv pip install numpy scipy scikit-learn matplotlib pandas
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run any script from inside its module folder, for example:

```bash
cd code/L2-05-transformers
python train_gpt.py
```

Some scripts write plots to `docs/assets/` relative to where they run. If you want those plots, run them from the repository root instead and point at the script path.

## Prerequisites

You should be able to read Python and be comfortable with high-school algebra. Level 0 builds the rest (matrices, derivatives, probability) as needed, with code.

## A note on length

Module pages run about 6,000 to 13,700 words because they contain the full code of every script and its real output. Skim the prose and run the code; the exercises are where understanding is built.

## A note on time

The module headers carry estimated study times (from 5 to 12 hours each, 214 to 275 hours for everything). These are the authors' estimates, not measurements of real learners, so treat them as rough.

## More from this series

- [Thoughts: the notes garden](https://sankarsh98.github.io/garden/)
- [How Computers Play Chess](https://sankarsh98.github.io/chess-brains/): build a tiny chess engine
- [AI Systems](https://sankarsh98.github.io/ai-systems/): agents, MCP and inference efficiency (L3-07 and L4-02 build on it)
- [All sites](https://sankarsh98.github.io/)
