# The AI Course

A multi-level, hands-on AI course: 32 modules in five levels, from the math you need to safety and governance. About 246,000 words, 162 runnable scripts, five capstone projects, and 373 annotated sources.

Site: https://sankarsh98.github.io/ai-course/

```bash
uv venv
uv pip install numpy scipy scikit-learn matplotlib pandas mkdocs-material
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/mkdocs serve                      # preview the site
tools/run_all_code.sh                       # run every script (about 25 minutes)
```

Read `docs/verification.md` for what was checked, what wasn't, and which sources are weaker. The content was drafted by AI agents under strict source-checking rules and has not been reviewed by a human domain expert.
