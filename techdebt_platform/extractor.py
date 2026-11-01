"""Turn ANY git repository into rows that follow the training-dataset schema.

Schema (identical to dataset.csv): file name, loc, sloc, comment_ratio, blank_ratio, cyclomatic_complexity,
max_complexity, maintainability_index, halstead_effort, num_functions, avg_function_len, max_nesting_depth,
is_tested, dependency_count, commit_count, author_count, days_since_change, churn_rate, bug_fix_commits,
refactor_commits, repo, label

The definitions were reverse-engineered from the dataset and checked against real repos (validate_extractor.py):
  * static columns  -> radon (raw, cc_visit, mi_visit(multi=True), h_visit().total.effort)
  * commit_count / author_count / days_since_change / churn_rate -> per-file raw value / the repo's median
  * bug_fix_commits / refactor_commits -> commits whose subject matches a keyword regex
  * label = bug_fix_commits >= 2
"""
import ast
import os
import re
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze

COLUMNS = ["file name", "loc", "sloc", "comment_ratio", "blank_ratio", "cyclomatic_complexity", "max_complexity",
           "maintainability_index", "halstead_effort", "num_functions", "avg_function_len", "max_nesting_depth",
           "is_tested", "dependency_count", "commit_count", "author_count", "days_since_change", "churn_rate",
           "bug_fix_commits", "refactor_commits", "repo", "label"]
LABEL_MIN_BUG_FIXES = 2
SKIP_DIRS = ("node_modules/", "venv/", ".venv/", "site-packages/", "/build/", "/dist/", ".git/")
BUG_RE = re.compile(r"fix|error|bug", re.I)
REFACTOR_RE = re.compile(r"remove|refactor|deprecat|split", re.I)
NEST = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True,
                          encoding="utf-8", errors="replace").stdout


def clone_if_url(src, workdir):
    """Accepts a local path or a git URL (full history is cloned into workdir)."""
    if os.path.isdir(src):
        return os.path.abspath(src)
    os.makedirs(workdir, exist_ok=True)
    name = re.sub(r"\.git$", "", src.rstrip("/").split("/")[-1])
    dest = os.path.join(workdir, name)
    if os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(["git", "-C", dest, "pull", "--quiet"], check=False)
    else:
        r = subprocess.run(["git", "clone", "--quiet", src, dest], capture_output=True, text=True)
        if r.returncode:
            raise ValueError(f"git clone failed: {r.stderr.strip()[:300]}")
    return dest


def read_blobs(repo, rev, paths):
    """Read many files at a revision through ONE git process (much faster than `git show` per file)."""
    p = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = {}
    for path in paths:
        p.stdin.write(f"{rev}:{path}\n".encode("utf-8")); p.stdin.flush()
        header = p.stdout.readline().split()
        if len(header) < 3 or header[1] != b"blob":
            continue
        size = int(header[2])
        data = p.stdout.read(size); p.stdout.read(1)
        out[path] = data.decode("utf-8", "replace")
    p.stdin.close(); p.wait()
    return out


def max_depth(node, d=0):
    m = d
    for ch in ast.iter_child_nodes(node):
        m = max(m, max_depth(ch, d + 1 if isinstance(ch, NEST) else d))
    return m


def static_features(code):
    if not code.strip():
        return {"loc": 0, "sloc": 0, "comment_ratio": 0.0, "blank_ratio": 0.0, "cyclomatic_complexity": 0.0,
                "max_complexity": 0.0, "maintainability_index": 100.0, "halstead_effort": 0.0, "num_functions": 0,
                "avg_function_len": 0.0, "max_nesting_depth": 0, "dependency_count": 0}
    r = analyze(code)
    blocks = cc_visit(code)
    cc = [b.complexity for b in blocks]
    lens = [b.endline - b.lineno + 1 for b in blocks]
    tree = ast.parse(code)
    n_imports = sum(len(n.names) if isinstance(n, ast.Import) else 1
                    for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
    loc = max(r.loc, 1)
    return {
        "loc": r.loc, "sloc": r.sloc,
        "comment_ratio": (r.comments + r.multi) / loc, "blank_ratio": r.blank / loc,
        "cyclomatic_complexity": float(np.mean(cc)) if cc else 0.0,
        "max_complexity": float(max(cc)) if cc else 0.0,
        "maintainability_index": mi_visit(code, True),
        "halstead_effort": h_visit(code).total.effort,
        "num_functions": len(blocks),
        "avg_function_len": float(np.mean(lens)) if lens else 0.0,
        "max_nesting_depth": max_depth(tree),
        "dependency_count": n_imports,
    }


def history_raw(repo, rev):
    """One row per (commit, file) up to `rev`."""
    fmt = "%x1d%H%x1f%ae%x1f%at%x1f%s%x1e"
    out = git(repo, "log", rev, "--no-merges", "--no-renames", "--numstat", f"--format={fmt}")
    rows = []
    for chunk in out.split("\x1d")[1:]:
        head, _, rest = chunk.partition("\x1e")
        h, a, t, s = head.split("\x1f", 3)
        bug, ref = bool(BUG_RE.search(s)), bool(REFACTOR_RE.search(s))
        for line in rest.splitlines():
            p = line.split("\t")
            if len(p) == 3 and p[0] != "-":
                rows.append((h, a, int(t), bug, ref, p[2], int(p[0]) + int(p[1])))
    return pd.DataFrame(rows, columns=["commit", "author", "ts", "bug", "ref", "path", "lines"])


def median_ratio(s):
    med = s.median()
    if med == 0:
        med = s.mean() if s.mean() > 0 else 1.0
    return s / med


def history_features(hist, paths, now):
    g = hist.groupby("path")
    raw = pd.DataFrame({
        "commit_count": g.commit.nunique(), "author_count": g.author.nunique(),
        "days_since_change": (now - g.ts.max()) / 86400, "churn_rate": g.lines.sum(),
        "bug_fix_commits": hist[hist.bug].groupby("path").commit.nunique(),
        "refactor_commits": hist[hist.ref].groupby("path").commit.nunique(),
    }).reindex(paths)
    raw[["bug_fix_commits", "refactor_commits"]] = raw[["bug_fix_commits", "refactor_commits"]].fillna(0).astype(int)
    raw = raw.fillna(0)
    out = pd.DataFrame(index=paths)
    for c in ["commit_count", "author_count", "days_since_change", "churn_rate"]:
        out[c] = median_ratio(raw[c])
    out["bug_fix_commits"], out["refactor_commits"] = raw.bug_fix_commits, raw.refactor_commits
    return out


def extract_repo(repo_path, name=None, rev="HEAD", caps=None, now=None, progress=None):
    """All .py files at `rev` -> DataFrame in dataset schema (label included)."""
    say = progress or (lambda m: None)
    repo_path = os.path.abspath(repo_path)
    name = name or os.path.basename(repo_path.rstrip("/\\"))
    paths = [p for p in git(repo_path, "ls-tree", "-r", "--name-only", rev).splitlines()
             if p.endswith(".py") and not any(s in "/" + p for s in SKIP_DIRS)]
    say(f"reading {len(paths)} Python files")
    blobs = read_blobs(repo_path, rev, paths)
    feats, contents = {}, {}
    for i, (p, code) in enumerate(blobs.items(), 1):
        if len(code) > 1_000_000:
            continue
        try:
            feats[p] = static_features(code)
            contents[p] = code
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue   # unparsable (e.g. Python 2) files are skipped
        if i % 200 == 0:
            say(f"computed code metrics for {i}/{len(blobs)} files")
    if not feats:
        raise ValueError("No analysable .py files found in this repository.")
    paths = list(feats)
    df = pd.DataFrame.from_dict(feats, orient="index")
    df["is_tested"] = [int(os.path.basename(p).startswith("test_")) for p in df.index]

    say("mining git history")
    hist = history_raw(repo_path, rev)
    df = df.join(history_features(hist, paths, now if now is not None else time.time()))
    for c, cap in (caps or {}).items():      # same outlier caps as the training data
        if c in df:
            df[c] = df[c].clip(upper=cap)
    df["repo"] = name
    df["file name"] = [name + "/" + p.replace("/", "\\") for p in df.index]
    df["label"] = (df.bug_fix_commits >= LABEL_MIN_BUG_FIXES).astype(int)
    return df.reset_index(drop=True)[COLUMNS]
