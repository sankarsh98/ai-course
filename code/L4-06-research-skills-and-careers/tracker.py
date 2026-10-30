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
