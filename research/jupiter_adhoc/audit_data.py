import numpy as np
import random
from pathlib import Path
from datetime import datetime

events_dir = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
micro_dir = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_microbatch")

events_files = sorted(events_dir.glob("*.npz"))
micro_files = sorted(micro_dir.glob("*.npz"))

print(f"Events files: {len(events_files)}")
print(f"Microbatch files: {len(micro_files)}")

nan_events = []
nan_micro = []
small_files = []
total_windows = 0
nan_labels = 0

random.seed(42)
sample_e = random.sample(events_files, min(20, len(events_files)))
for f in sample_e:
    try:
        d = np.load(f)
        feat = d["features"] if "features" in d else d[d.files[0]]
        if np.isnan(feat).any():
            nan_events.append(f.name)
        for k in ["labels", "label", "targets"]:
            if k in d:
                lbl = d[k]
                total_windows += len(lbl)
                nan_labels += int(np.isnan(lbl).sum())
                break
        if f.stat().st_size < 1000:
            small_files.append((f.name, f.stat().st_size))
    except Exception as e:
        print(f"ERROR {f.name}: {e}")

sample_m = random.sample(micro_files, min(20, len(micro_files)))
for f in sample_m:
    try:
        d = np.load(f)
        for arr in d.files:
            if np.isnan(d[arr]).any():
                nan_micro.append(f.name)
                break
    except Exception as e:
        print(f"MICRO_ERR {f.name}: {e}")

dates = []
for f in events_files:
    parts = f.stem.split("_")
    if parts[0].isdigit() and len(parts[0]) == 8:
        try:
            dates.append(datetime.strptime(parts[0], "%Y%m%d"))
        except Exception:
            pass

dates.sort()
gaps = []
if dates:
    prev = dates[0]
    for d in dates[1:]:
        delta = (d - prev).days
        if delta > 5:
            gaps.append((prev.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"), delta))
        prev = d
    print(f"\nDate range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(dates)}")
    print(f"Gaps >5 cal days: {len(gaps)}")
    for g in gaps[:15]:
        print(f"  {g[0]} -> {g[1]} ({g[2]} days)")

print(f"\nNaN events (20-file sample): {nan_events or 'NONE'}")
print(f"NaN microbatch (20-file sample): {nan_micro or 'NONE'}")
print(f"Tiny/corrupt files: {small_files or 'NONE'}")
print(f"Label audit: {total_windows} windows, {nan_labels} NaN labels")
print("AUDIT_DONE")
