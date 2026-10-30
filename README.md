# The AI Course

A multi-level, hands-on AI course: 32 modules in five levels, from the math you need to safety and governance. About 246,000 words, 162 runnable scripts, five capstone projects, and 373 annotated sources.

Site: https://sankarsh98.github.io/ai-course/ (built with Quartz; the content is plain Markdown in `docs/`)

```bash
# preview the site
npm ci && npx quartz plugin install
npx quartz build -d docs --serve

# run the course code
uv venv
uv pip install numpy scipy scikit-learn matplotlib pandas
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
tools/run_all_code.sh                       # run every script (about 25 minutes)
```

Read `docs/verification.md` for what was checked, what wasn't, and which sources are weaker. The content was drafted by AI agents under strict source-checking rules and has not been reviewed by a human domain expert.
