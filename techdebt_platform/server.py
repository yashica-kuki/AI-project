"""Local web app.   python server.py [--base dataset.csv] [--port 8000]
Open http://localhost:8000 - pick a repo, add new repos (git URL or local folder), see the model, compare repos."""
import argparse
import json
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import engine

HTML = Path(__file__).resolve().parent / "index.html"
JOBS, LOCK = {}, threading.Lock()


def run_job(kind, fn):
    if not LOCK.acquire(blocking=False):
        return None
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"status": "running", "kind": kind, "log": [], "result": None}

    def work():
        try:
            JOBS[jid]["result"] = fn(lambda m: (JOBS[jid]["log"].append(m), print("  >", m)))
            JOBS[jid]["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            JOBS[jid].update(status="error", error=str(e)[:600])
        finally:
            LOCK.release()
    threading.Thread(target=work, daemon=True).start()
    return jid


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200, headers=None):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def err(self, msg, code=400):
        self.send(json.dumps({"error": msg}), code=code)

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.send(HTML.read_text(encoding="utf-8"), "text/html")
            if u.path == "/api/repos":
                return self.send(json.dumps({"repos": engine.repo_table(), "busy": LOCK.locked()}))
            if u.path == "/api/model":
                return self.send(json.dumps(engine._load(engine.META, {})))
            if u.path == "/api/report":
                return self.send(json.dumps({"repo": q["repo"], "rows": engine.repo_report(q["repo"])}))
            if u.path == "/api/job":
                return self.send(json.dumps(JOBS.get(q.get("id"), {"status": "unknown"})))
            if u.path == "/api/download":
                return self.send(engine.repo_csv(q["repo"]), "text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{q["repo"]}_risk.csv"'})
            self.err("not found", 404)
        except KeyError:
            self.err("unknown repository", 404)
        except Exception as e:
            traceback.print_exc(); self.err(str(e), 500)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/analyze":
                src = (body.get("source") or "").strip()
                if not src:
                    return self.err("Give a git URL or a local folder path.")
                jid = run_job("analyze", lambda p: engine.analyze_repo(src, (body.get("name") or "").strip() or None, p))
            elif self.path == "/api/mode":
                if body.get("mode") not in ("static", "history"):
                    return self.err("bad mode")
                jid = run_job("retrain", lambda p: engine.train_all(body["mode"], p) and body["mode"])
            elif self.path == "/api/delete":
                jid = run_job("delete", lambda p: engine.delete_repo(body.get("repo", ""), p))
            else:
                return self.err("not found", 404)
            if jid is None:
                return self.err("Another job is still running - wait for it to finish.", 409)
            self.send(json.dumps({"job_id": jid}))
        except Exception as e:
            traceback.print_exc(); self.err(str(e), 500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="original training CSV (only needed on the very first run)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    engine.ensure_ready(a.base)
    url = f"http://localhost:{a.port}"
    print(f"\nDashboard running at {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
