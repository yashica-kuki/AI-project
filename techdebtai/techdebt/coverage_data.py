"""Read test coverage (Cobertura XML from coverage.py/Jest/etc., or lcov.info) -> % lines covered per file."""
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd


def _read_cobertura(path):
    cov = defaultdict(dict)
    for cls in ET.parse(path).getroot().iter("class"):
        fn = cls.get("filename", "").replace("\\", "/")
        for ln in cls.iter("line"):
            n, hits = int(ln.get("number")), int(ln.get("hits", 0))
            cov[fn][n] = max(hits, cov[fn].get(n, 0))
    return cov


def _read_lcov(path):
    cov, cur = defaultdict(dict), None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                cur = line[3:].replace("\\", "/")
            elif line.startswith("DA:") and cur:
                n, hits = line[3:].split(",")[:2]
                cov[cur][int(n)] = max(int(hits), cov[cur].get(int(n), 0))
    return cov


def load_coverage(path: str, repo_paths) -> pd.Series:
    """Return % covered per repo path. Coverage paths are matched by trailing path components."""
    cov = _read_lcov(path) if path.endswith((".info", ".lcov")) else _read_cobertura(path)
    by_base = defaultdict(list)
    for f in cov:
        by_base[f.rsplit("/", 1)[-1]].append(f)
    out = {}
    for p in repo_paths:
        for f in by_base.get(p.rsplit("/", 1)[-1], []):
            if p == f or p.endswith("/" + f) or f.endswith("/" + p):
                lines = cov[f]
                out[p] = 100.0 * sum(h > 0 for h in lines.values()) / max(len(lines), 1)
                break
    return pd.Series(out, dtype=float, name="coverage")
