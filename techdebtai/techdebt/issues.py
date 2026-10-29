"""Fetch bug / non-bug issue numbers from GitHub (cached to JSON so you only hit the API once).

Offline / other trackers (Jira, GitLab): just write the same JSON by hand or by script:
    {"bug": [12, 40, 77], "other": [5, 6, 9]}
and pass it with --issues-file.
"""
import json
import os
import re
import urllib.error
import urllib.request

LABEL_RE = re.compile(r"bug|defect|regression|crash|error", re.I)


def fetch_github_issues(slug: str, token: str = None, max_pages: int = 30) -> dict:
    bug, other = [], []
    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{slug}/issues?state=all&per_page=100&page={page}"
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "techdebt-ai"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"GitHub API error {e.code} for {slug}. Rate-limited? "
                             f"Set the GITHUB_TOKEN environment variable and retry.")
        if not data:
            break
        for it in data:
            is_bug = any(LABEL_RE.search(l["name"]) for l in it.get("labels", []))
            if is_bug:
                bug.append(it["number"])
            elif "pull_request" not in it:  # PRs without a bug label are ignored (neither list)
                other.append(it["number"])
        if len(data) < 100:
            break
    return {"bug": bug, "other": other}


def resolve_issues(github: str = None, issues_file: str = None, token: str = None):
    """Load from cache file if present, else fetch from GitHub (and cache). Returns None if neither given."""
    if issues_file and os.path.exists(issues_file):
        with open(issues_file, encoding="utf-8") as f:
            data = json.load(f)
    elif github:
        data = fetch_github_issues(github, token)
        if issues_file:
            with open(issues_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
    else:
        return None
    print(f"Issue tracker: {len(data['bug'])} bug issues, {len(data['other'])} other issues")
    return data
