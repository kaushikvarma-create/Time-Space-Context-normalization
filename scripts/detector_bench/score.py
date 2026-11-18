"""Score every candidate's saved boxes. Chairs: truth is 7 in every cabin photo. Other types: the hand labels.
The counting threshold is chosen on half the photos (even positions) and reported on the other half."""
import json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "data" / "exports" / "detector_bench"
MAP = {"chair": "chair", "person": "person", "tv": "monitor", "monitor": "monitor", "laptop": "laptop", "backpack": "bag", "handbag": "bag",
       "car": "car", "motorcycle": "motorcycle", "couch": "sofa", "office chair": "chair", "computer monitor": "monitor",
       "laptop computer": "laptop", "scooter": "motorcycle"}
c = sqlite3.connect(f"file:{ROOT/'data/sbi.db'}?mode=ro", uri=True)
LAB = {}
for fid, typ, n in c.execute("select frame_id, object_type, true_count from labels"): LAB.setdefault(str(fid), {})[typ] = n
THR = [round(x, 2) for x in np.arange(0.10, 0.81, 0.05)]

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)

def counts(boxes, label, thr, dedupe=0.6):
    kept = []
    for name, conf, box in sorted(boxes, key=lambda x: -x[1]):
        if MAP.get(name.strip().lower()) != label or conf < thr: continue
        if not any(iou(k, box) > dedupe for k in kept): kept.append(box)
    return len(kept)

rows = []
for f in sorted(OUT.glob("*.json")):
    res = json.loads(f.read_text())
    cab = [(fid, r) for fid, r in res.items() if r["space"] == "admin_cabin"]
    tune, test = cab[0::2], cab[1::2]
    best = min(THR, key=lambda t: np.mean([abs(counts(r["boxes"], "chair", t) - 7) for _, r in tune]))
    got = np.array([counts(r["boxes"], "chair", best) for _, r in test]); night = np.array([(r["brightness"] or 200) < 95 for _, r in test])
    at25 = np.array([counts(r["boxes"], "chair", 0.25) for _, r in test])
    row = {"detector": f.stem, "chair thr": best, "chairs =7": (got == 7).mean(), "within 1": (abs(got - 7) <= 1).mean(), "avg miss": abs(got - 7).mean(),
           "=7 day": (got[~night] == 7).mean(), "=7 night": (got[night] == 7).mean(), "min": got.min(), "max": got.max(), "=7 @0.25": (at25 == 7).mean()}
    for typ in ("person", "monitor", "laptop", "bag", "car", "motorcycle"):
        lab = [(fid, LAB[fid][typ]) for fid in res if fid in LAB and typ in LAB[fid]]
        thr = 0.25 if typ != "monitor" else min(THR, key=lambda t: np.mean([abs(counts(res[fid]["boxes"], typ, t) - n) for fid, n in lab]))
        err = [abs(counts(res[fid]["boxes"], typ, thr) - n) for fid, n in lab]
        row[f"{typ} miss"] = np.mean(err) if err else np.nan
    rows.append(row)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).sort_values("avg miss").round(2).to_string(index=False))
print("photos scored for chairs:", len(test), " night:", int(night.sum()), " labelled photos:", len(LAB))
