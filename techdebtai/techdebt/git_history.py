"""Mine a git repository: per-commit, per-file change history + bug-fix labelling.

A commit counts as a bug fix when:
  * it references (#123) an issue that the tracker labels as a bug, OR
  * its subject looks like a fix (keywords) and it does NOT reference a known non-bug issue.
"""
import re
import subprocess
from datetime import datetime, timezone

import pandas as pd

BUG_RE = re.compile(r"\b(fix(es|ed)?|bug(s)?|defect|hotfix|patch|crash|regression)\b", re.I)
REF_RE = re.compile(r"(?<![\w/])#(\d+)\b")  # "#123" in subject or body


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, check=True, encoding="utf-8", errors="replace",
    ).stdout


def load_history(repo: str, issues: dict = None) -> pd.DataFrame:
    """One row per (commit, file): commit, author, ts, is_bug, path, added, deleted.
    `issues` = {"bug": [numbers], "other": [numbers]} from issues.py (optional)."""
    bug = set((issues or {}).get("bug", []))
    other = set((issues or {}).get("other", []))
    fmt = "%x1d%H%x1f%an%x1f%at%x1f%s%x1f%b%x1e"
    out = git(repo, "log", "--no-merges", "--no-renames", "--numstat", f"--format={fmt}")
    rows = []
    for chunk in out.split("\x1d")[1:]:
        head, _, rest = chunk.partition("\x1e")
        h, a, t, subj, body = head.split("\x1f", 4)
        refs = {int(n) for n in REF_RE.findall(subj + "\n" + body)}
        is_bug = bool(refs & bug) or (bool(BUG_RE.search(subj)) and not (refs & other))
        for line in rest.splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[0] != "-":  # skip binary files
                rows.append((h, a, int(t), is_bug, parts[2], int(parts[0]), int(parts[1])))
    cols = ["commit", "author", "ts", "is_bug", "path", "added", "deleted"]
    return pd.DataFrame(rows, columns=cols)


def rev_at(repo: str, ts: int) -> str:
    """Last commit on HEAD's history at or before unix time `ts`."""
    when = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")
    return git(repo, "rev-list", "-1", f"--before={when}", "HEAD").strip()
