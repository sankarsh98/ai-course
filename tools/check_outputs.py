"""Sample check: does a script's fresh stdout appear on its module page?"""
import re, subprocess, sys, glob, os, random
log = open("/tmp/run_all_final.log").read().splitlines()
fast = [l.split()[2] for l in log if l.startswith("PASS") and int(l.split()[1].rstrip("s")) <= 4]
random.seed(0); sample = random.sample(fast, 20)
total_ok = total = 0
for script in sorted(sample):
    mod = script.split("/")[1]                       # e.g. L2-05-transformers
    lvl = "level-" + mod[1] + "-*"
    page = next(p for p in glob.glob(f"docs/{lvl}/*.md")
                if re.search(re.escape(mod), next(l for l in open(p, encoding="utf-8").read().splitlines() if l.startswith("> **Level"))))
    text = open(page, encoding="utf-8").read()
    d, b = os.path.split(script)
    r = subprocess.run([sys.executable, "-I", b] if False else [".venv/bin/python".replace(".venv", os.path.abspath(".venv")), b],
                       cwd=d, capture_output=True, text=True, timeout=120)
    lines = [l.rstrip() for l in r.stdout.splitlines() if l.strip()]
    found = sum(1 for l in lines if l in text)
    # lines containing wall-clock timings legitimately differ
    timing = sum(1 for l in lines if l not in text and re.search(r"\d+(\.\d+)?\s*(ms|s\b|seconds|sec)", l))
    total += len(lines); total_ok += found
    flag = "" if found + timing == len(lines) else "   <-- check"
    print(f"{found:>3}/{len(lines):<3} lines on page  (+{timing} timing lines)  {script}{flag}")
print(f"\nTOTAL {total_ok}/{total} output lines found verbatim on the module pages")
