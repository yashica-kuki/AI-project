"""Compare the extractor with the original dataset for a repo you cloned and checked out at the dataset's snapshot.
   python validate_extractor.py dataset.csv REPO_NAME PATH_TO_CLONE REV
(days_since_change depends on the date the dataset was built, so that reference date is fitted automatically.)"""
import sys
import numpy as np
import pandas as pd
import extractor as E

ds, name, path, rev = sys.argv[1:5]
ref = pd.read_csv(ds); ref = ref[ref.repo == name].copy()
hist = E.history_raw(path, rev); last = hist.ts.max()
key = lambda s: s.str.replace("\\", "/", regex=False)
ref["k"] = key(ref["file name"])
probe = E.extract_repo(path, name, rev, now=last)
paths = [k.split("/", 1)[1] for k in key(probe["file name"])]
best = (-1, 0)
for off in np.arange(0, 1500, 0.5):          # fit the dataset's reference date
    d = E.history_features(hist, paths, last + off * 86400)["days_since_change"].values
    m = np.isclose(d, ref.set_index("k").reindex(key(probe["file name"]))["days_since_change"].values, rtol=.03, atol=.03)
    if np.nansum(m) > best[0]: best = (np.nansum(m), off)
mine = E.extract_repo(path, name, rev, now=last + best[1] * 86400)
mine["k"] = key(mine["file name"])
c = ref.merge(mine, on="k", suffixes=("_d", "_m"))
print(f"dataset rows {len(ref)} | extracted {len(mine)} | matched by path {len(c)} | fitted reference date = last commit + {best[1]:.1f} days")
print(f"\n{'column':24s} match rate (within 2%)")
for col in [x for x in E.COLUMNS if x not in ("file name", "repo")]:
    d, m = c[col + "_d"].astype(float), c[col + "_m"].astype(float)
    print(f"{col:24s} {np.isclose(d, m, rtol=0.02, atol=0.02).mean():6.1%}")
