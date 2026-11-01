# TechDebt-AI platform: one model trained on many repos

Requirements: Python 3.9+, git on PATH.

## Run (Windows)
    pip install -r requirements.txt
    python server.py --base "C:\path\to\dataset.csv"      # first run only (trains the base model, ~1 min)
    python server.py                                       # every later run
Opens http://localhost:8000.

## What you do in the dashboard
1. "Add repository" -> paste a git URL (full history is cloned) or a local folder.
2. The tool extracts the same 20 features as dataset.csv for every .py file, appends the rows to
   data/master_dataset.csv, retrains on ALL rows, and opens the analysis.
3. Risk scores in the analysis come from a model that did NOT see that repo (leave-repos-out), so they show
   how the model behaves on a genuinely new project. The rows are then kept for all future training.

Headless:  python engine.py add https://github.com/pallets/flask

## Files
extractor.py            git repo -> dataset-schema rows (definitions validated against dataset.csv)
validate_extractor.py   re-check the extractor against a repo from your dataset
engine.py               master dataset, training, leave-repos-out scoring, SHAP, suggestions
server.py + index.html  local web app (no extra packages beyond requirements.txt)
data/                   created on first run: master_dataset.csv, model.joblib, scored.csv, meta.json (back this up)

## Notes
- label = "file has >= 2 bug-fix commits". bug_fix_commits is therefore never used as a model input.
- "Code metrics only" (default) is reproducible for any repo. "+ commit history" scores higher but is partly circular.
- Python files only (Python 3 syntax); unparsable files are skipped.
- AUC for a repo with fewer than ~30 files is noisy.
