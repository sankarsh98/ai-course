"""Generate level index pages and the combined bibliography from the module files."""
import re, glob, os

LEVELS = {
    "level-0-foundations": ("Level 0: Foundations",
        "The math and tools that everything else uses: arrays and floating point, linear algebra, automatic differentiation, probability and statistics, and optimization. Each module is built around code you run, including a small autograd engine you write yourself."),
    "level-1-classical-ml": ("Level 1: Classical machine learning",
        "Learning from data without deep networks: evaluation done honestly, linear and logistic models, trees and boosting, unsupervised learning, and the data and feature work that decides most real projects. These habits carry into every later level."),
    "level-2-deep-learning": ("Level 2: Deep learning",
        "From backpropagation by hand to a working GPT: training practice, convolutional networks, sequence models, transformers, and embeddings. Everything trains on a CPU in under a minute."),
    "level-3-modern-ai": ("Level 3: Modern AI across domains",
        "Language-model pretraining and post-training, generative models and diffusion, vision and multimodal models, reinforcement learning, retrieval, agents, recommenders, time series and graphs, and AI in science and healthcare. Time-sensitive claims are dated and verified as of October 2026."),
    "level-4-engineering-and-society": ("Level 4: Engineering and society",
        "Running, measuring, understanding and governing AI systems: MLOps, efficient training and inference, evaluation and benchmarks, interpretability, safety and governance, and research skills and careers."),
}

def parse(path):
    text = open(path, encoding="utf-8").read()
    fm = re.match(r"^---\ntitle: \"(.*?)\"\n---", text, re.S)
    title = fm.group(1) if fm else text.splitlines()[0].lstrip("# ").strip()
    meta = next((l for l in text.splitlines() if l.startswith("> **Level")), "")
    prereq = re.search(r"Prerequisites: (.*?) ·", meta)
    time_ = re.search(r"Time: (.*?) ·", meta)
    code = re.search(r"Code: `(.*?)`", meta)
    outcomes = []
    m = re.search(r"## What you will be able to do\n(.*?)\n## ", text, re.S)
    if m:
        outcomes = [l[2:].strip() for l in m.group(1).splitlines() if l.startswith("- ")]
    fr = text.split("## Further reading")[-1]
    entries = [l for l in fr.splitlines() if l.startswith("- **")]
    # keep each entry's continuation line ("Read for")
    lines = fr.splitlines(); full = []
    for i, l in enumerate(lines):
        if l.startswith("- **"):
            nxt = lines[i+1].strip() if i+1 < len(lines) else ""
            full.append(l + ("\n  " + nxt if nxt.startswith("*Read for") else ""))
    return dict(path=path, file=os.path.basename(path), title=title,
                prereq=prereq.group(1) if prereq else "", time=time_.group(1) if time_ else "",
                code=code.group(1) if code else "", words=len(text.split()),
                outcomes=outcomes, entries=full)

def hours(t):
    m = re.search(r"(\d+) to (\d+)", t)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)", t)
    return (int(m.group(1)),) * 2 if m else (0, 0)

mods, totals = [], dict(lo=0, hi=0, words=0, scripts=0, entries=0)
for d, (name, blurb) in LEVELS.items():
    rows, ms = [], []
    for f in sorted(glob.glob(f"docs/{d}/[0-9]*.md")):
        m = parse(f); ms.append(m)
        mid = m["code"].rstrip("/").split("/")[-1]
        short = mid.split("-")[0] + "-" + mid.split("-")[1]
        n_scripts = len(glob.glob(m["code"] + "*.py"))
        lo, hi = hours(m["time"]); totals["lo"] += lo; totals["hi"] += hi
        totals["words"] += m["words"]; totals["scripts"] += n_scripts; totals["entries"] += len(m["entries"])
        m.update(short=short, mid=mid, scripts=n_scripts)
        rows.append(f"| {short} | {m['title']} | {m['prereq']} | {m['time']} | {n_scripts} |")
    mods.append((d, name, blurb, ms))
    out = ["---", f'title: "{name}"', "---", "", blurb, "", "| ID | Module | Prerequisites | Time | Scripts |", "|---|---|---|---|---|", *rows, ""]
    out += ["## What each module teaches", ""]
    for m in ms:
        out += [f"### {m['short']}: {m['title']}", ""] + [f"- {o}" for o in m["outcomes"][:4]] + [""]
    open(f"docs/{d}/index.md", "w", encoding="utf-8").write("\n".join(out))

# combined bibliography
bib = ["---", 'title: "Bibliography"', "---", "",
       "Every entry below is the \"Further reading\" list of one module, collected in one place. Each source was looked up and checked while the module was written; see the verification notes for how, and for what is weaker. Inline citations in the module text that are not in these lists are not repeated here.", ""]
seen = {}
for d, name, blurb, ms in mods:
    bib += [f"## {name}", ""]
    for m in ms:
        bib += [f"### {m['short']}: {m['title']}", ""]
        for e in m["entries"]:
            bib.append(e)
        bib.append("")
open("docs/bibliography.md", "w", encoding="utf-8").write("\n".join(bib))
print("modules:", sum(len(x[3]) for x in mods), "| words (incl. code):", totals["words"], "| scripts:", totals["scripts"],
      "| reading entries:", totals["entries"], "| hours: %d to %d" % (totals["lo"], totals["hi"]))
