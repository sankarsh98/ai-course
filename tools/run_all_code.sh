#!/usr/bin/env bash
# Run every script under code/ (skipping test_*.py, which are run with pytest-style main) and report pass/fail with timing.
cd "$(dirname "$0")/.."
pass=0; fail=0
for f in $(ls code/*/*.py | sort); do
  d=$(dirname "$f"); b=$(basename "$f")
  s=$(date +%s)
  if (cd "$d" && timeout 180 ../../.venv/bin/python "$b" > /tmp/run_$$.out 2>&1); then st=PASS; pass=$((pass+1)); else st=FAIL; fail=$((fail+1)); fi
  echo "$st $(( $(date +%s)-s ))s $f"
  [ $st = FAIL ] && tail -5 /tmp/run_$$.out | sed 's/^/     /'
done
rm -rf code/*/docs   # scripts write plots relative to cwd; keep only the real docs/assets
echo "TOTAL pass=$pass fail=$fail"
