"""
Build data/INVENTORY.md per HC #563 R1.

Walks /home/jupiter/Lvl3Quant for parquet / csv / npz / json (data only) / sqlite,
extracts schema (column list), frequency hint, coverage (date min/max if date col),
row count, last modified.

Best-effort SSH probe to Neptune (nick@neptune) and Razer
(claude@razer) to enumerate top-level data dirs.

Inventory is grouped by candidate feature family per HC #563 R2.
"""
from __future__ import annotations
import os, sys, json, traceback, subprocess, datetime as dt
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT  = ROOT / "data" / "INVENTORY.md"

# Directories that contain DATA (not configs/logs/code/output predictions)
DATA_ROOTS = [
    ROOT / "data",
    ROOT / "wheel_strategy_v1" / "data" / "cache",
    ROOT / "live_trading_linux" / "models",
    ROOT / "research" / "findings",
]

EXTS = {".parquet", ".csv", ".npz", ".json", ".sqlite", ".db"}

# Skip these subtrees — too noisy / not data we want in the picker
SKIP_DIRS = {
    "__pycache__", ".dolt", "directive_backups", "fill_sim_cache",
    "precomputed_obs", "razer_pull", "mlruns", "mlruns.trash",
    "v3_2_prior_session_state",
}

# Heuristic family mapper from filename or path. Keys = substring patterns
# (lowercased), values = family label.
FAMILY_MAP = [
    ("fundamentals_pit",      "fundamentals_numeric_pit"),
    ("fundamentals",          "fundamentals_numeric"),
    ("iv_features",           "options_iv_signals"),
    ("options_modeled",       "options_iv_signals"),
    ("options_real",          "options_iv_signals"),
    ("vol_history",           "options_iv_signals"),
    ("sector_etf",            "sector_flows"),
    ("sector_flow",           "sector_flows"),
    ("flow_features",         "sector_flows"),
    ("factor_features",       "factor_exposures"),
    ("regime",                "macro_regime"),
    ("theme",                 "themes"),
    ("intraday_features",     "intraday"),
    ("naaim",                 "macro_state"),
    ("fred_",                 "macro_state"),
    ("macro",                 "macro_state"),
    ("economic_calendar",     "calendar_events"),
    ("price_features",        "price_returns"),
    ("ga_name_table",         "ranker_artifact"),
    ("prices",                "price_returns"),
    ("universe",              "universe_def"),
    ("mbo",                   "intraday_mbo_internal"),  # microstructure, not stock-picking
    ("mfe_mae",               "intraday_mbo_internal"),
    ("oot_predictions",       "model_predictions"),
    ("predictions",           "model_predictions"),
    ("orderflow_features",    "intraday_mbo_internal"),
    ("session_features",      "intraday_mbo_internal"),
    ("salience_tags",         "intraday_mbo_internal"),
    ("mid_price_cache",       "intraday_mbo_internal"),
]

def classify(path: Path) -> str:
    p = str(path).lower()
    for sub, fam in FAMILY_MAP:
        if sub in p:
            return fam
    return "unclassified"

def frequency_hint(name: str) -> str:
    n = name.lower()
    if "daily" in n or "pit_daily" in n: return "daily"
    if "weekly" in n: return "weekly"
    if "quarterly" in n: return "quarterly"
    if "mbo" in n: return "tick / event"
    if "intraday" in n: return "intraday"
    if "predictions" in n: return "per-fold"
    return "?"

def schema_parquet(path: Path):
    try:
        sch = pq.read_schema(path)
        cols = [f.name for f in sch]
        meta = pq.ParquetFile(path).metadata
        n = meta.num_rows
        return cols, n
    except Exception as e:
        return [f"<err:{e.__class__.__name__}:{str(e)[:60]}>"], -1

def schema_csv(path: Path):
    try:
        df = pd.read_csv(path, nrows=1)
        # rough row count
        try:
            with open(path, "rb") as f:
                n = sum(1 for _ in f) - 1
        except Exception:
            n = -1
        return list(df.columns), n
    except Exception as e:
        return [f"<err:{e.__class__.__name__}:{str(e)[:60]}>"], -1

def schema_npz(path: Path):
    try:
        import numpy as np
        z = np.load(path, allow_pickle=True)
        cols = list(z.files)
        # row count = first array length
        n = -1
        for k in cols:
            try:
                arr = z[k]
                if hasattr(arr, "shape") and len(arr.shape):
                    n = int(arr.shape[0]); break
            except Exception:
                continue
        return cols, n
    except Exception as e:
        return [f"<err:{e.__class__.__name__}:{str(e)[:60]}>"], -1

def schema_json(path: Path):
    try:
        # only inspect top-level keys; don't load big files entirely
        size = path.stat().st_size
        if size > 50_000_000:  # >50 MB — skip
            return ["<large-json-skipped>"], -1
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return list(obj.keys())[:30], 1
        if isinstance(obj, list):
            cols = list(obj[0].keys())[:30] if obj and isinstance(obj[0], dict) else []
            return cols, len(obj)
        return ["<scalar>"], 1
    except Exception as e:
        return [f"<err:{e.__class__.__name__}:{str(e)[:60]}>"], -1

def coverage_parquet(path: Path, cols):
    # try common date columns
    candidates = [c for c in cols if c.lower() in ("date","dt","ts_event","timestamp","asof","ts","day")]
    if not candidates:
        return None
    c = candidates[0]
    try:
        df = pq.read_table(path, columns=[c]).to_pandas()
        s = pd.to_datetime(df[c], errors="coerce")
        return f"{s.min()} → {s.max()}"
    except Exception:
        return None

def walk():
    rows = []
    for root in DATA_ROOTS:
        if not root.exists(): continue
        for dirpath, dirnames, filenames in os.walk(root):
            # prune
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext not in EXTS: continue
                p = Path(dirpath) / fn
                try:
                    stat = p.stat()
                except FileNotFoundError:
                    continue
                size = stat.st_size
                # Skip near-empty files
                if size < 100 and ext != ".json": continue
                # Skip huge MBO event files — too many to list individually
                # but include them as a single rollup later.
                mtime = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")

                if ext == ".parquet":
                    cols, n = schema_parquet(p)
                    cov = coverage_parquet(p, cols)
                elif ext == ".csv":
                    cols, n = schema_csv(p); cov = None
                elif ext == ".npz":
                    cols, n = schema_npz(p); cov = None
                elif ext == ".json":
                    cols, n = schema_json(p); cov = None
                else:  # sqlite/db
                    cols, n, cov = ["<sqlite>"], -1, None

                fam = classify(p)
                rel = p.relative_to(ROOT)
                rows.append({
                    "path": str(rel),
                    "ext": ext.lstrip("."),
                    "size_mb": round(size/1e6, 2),
                    "rows": n,
                    "cols": cols[:25],
                    "n_cols": len(cols),
                    "coverage": cov,
                    "freq": frequency_hint(fn),
                    "mtime": mtime,
                    "family": fam,
                })
    return rows

def rollup_mbo():
    """Roll up tick-level MBO files into a single line — too many to list."""
    rolls = []
    for sub in ["data/raw/mbo", "data/mfe_mae_labels_v2", "mbo_oot"]:
        d = ROOT / sub
        if not d.exists(): continue
        files = list(d.rglob("*"))
        files = [f for f in files if f.is_file()]
        if not files: continue
        total_size_mb = sum(f.stat().st_size for f in files) / 1e6
        dates = []
        for f in files[:2000]:
            # try to extract a yyyymmdd from filename
            stem = f.stem
            for i in range(len(stem)-7):
                ch = stem[i:i+8]
                if ch.isdigit() and ch.startswith(("2024","2025","2026")):
                    dates.append(ch); break
        cov = ""
        if dates:
            cov = f"{min(dates)} → {max(dates)}"
        rolls.append({
            "subdir": sub, "n_files": len(files),
            "size_mb": round(total_size_mb,1), "coverage": cov,
        })
    return rolls

def ssh_probe(user_host: str, paths: list[str], timeout: int=10):
    out = {}
    for p in paths:
        cmd = ["ssh","-o","StrictHostKeyChecking=no","-o",f"ConnectTimeout={timeout}",
               user_host, f"ls -la {p} 2>&1 | head -40"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+2)
            out[p] = r.stdout if r.returncode==0 else f"<err:{r.returncode}> {r.stderr[:200]}"
        except Exception as e:
            out[p] = f"<exc:{e.__class__.__name__}:{e}>"
    return out

def main():
    rows = walk()
    mbo = rollup_mbo()

    # group by family
    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r)

    # ssh probes — best effort
    try:
        neptune = ssh_probe("nick@neptune",
                            ["/home/nick/Lvl3Quant/data",
                             "/home/nick/Lvl3Quant/wheel_strategy_v1/data/cache"])
    except Exception as e:
        neptune = {"ERR": str(e)}
    try:
        razer = ssh_probe("claude@razer",
                          ["C:/Users/claude/Lvl3Quant/data",
                           "C:/Users/claude/Lvl3Quant/wheel_strategy_v1/data/cache"], timeout=8)
    except Exception as e:
        razer = {"ERR": str(e)}

    # families mandated by HC #563 R2
    MANDATED = {
        "price_returns": "Price / return / volume (daily + intraday)",
        "realized_vol": "Realized vol multi-horizon",
        "factor_exposures": "Factor exposures (FF5+momentum+quality+lowvol)",
        "fundamentals_numeric": "Fundamentals — numeric",
        "fundamentals_numeric_pit": "Fundamentals — numeric PIT-safe",
        "fundamentals_text_10k": "Fundamentals — text (10-K/10-Q MD&A, risk factors)",
        "earnings_transcripts": "Earnings transcripts sentiment",
        "news_sentiment": "News sentiment (GDELT, RSS, NewsAPI)",
        "social_hype": "Social hype (Reddit / StockTwits / Twitter / Trends)",
        "sector_flows": "Sector ETF flow proxies",
        "ticker_flows_13f": "13F holdings deltas",
        "insider_form4": "Insider Form-4 transactions",
        "options_iv_signals": "Options-derived signals (IV rank, skew, P/C, gamma)",
        "macro_state": "Macro state (VIX/MOVE/DXY/yield-curve/HY OAS/FCI/claims/CPI)",
        "macro_regime": "Macro regime tags",
        "cross_asset": "Cross-asset (gold/oil/copper/BTC/USTs/EURUSD)",
        "sector_rotation": "Sector rotation matrices, lead-lag",
        "themes": "Theme baskets (AI infra, physical AI, etc.)",
        "intraday": "Intraday features (gap/OR/VWAP-dev/intra-RV)",
        "calendar_events": "Calendar (earnings, FOMC, CPI, OPEX)",
        "analyst_revisions": "Analyst consensus + revisions",
        "alt_web_traffic": "Alt data (SimilarWeb / app downloads / Glassdoor / Indeed)",
    }
    present = set(by_fam.keys())
    missing = [f for f in MANDATED if f not in present]
    # treat single-file presence with no real coverage / minimal cols as PARTIAL
    partial = []
    for f, rs in by_fam.items():
        if f not in MANDATED: continue
        biggest = max(rs, key=lambda r: r["size_mb"])
        if biggest["size_mb"] < 0.05:
            partial.append(f)

    lines = []
    A = lines.append
    A("# Data Inventory (HC #563 R1)")
    A("")
    A(f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_")
    A("")
    A(f"Total data files cataloged: **{len(rows)}** (parquet/csv/npz/json/sqlite)")
    A("")
    A("## Family Coverage Summary")
    A("")
    A("| Family | Status | # files | Notes |")
    A("|---|---|---|---|")
    for fam, label in MANDATED.items():
        files = by_fam.get(fam, [])
        if not files:
            status = "MISSING"
        elif fam in partial:
            status = "PARTIAL"
        else:
            status = "PRESENT"
        sample = files[0]["path"] if files else "—"
        A(f"| `{fam}` ({label}) | {status} | {len(files)} | {sample} |")
    other_fams = sorted(set(by_fam.keys()) - set(MANDATED.keys()))
    for fam in other_fams:
        files = by_fam[fam]
        A(f"| `{fam}` (not in R2 mandate) | EXTRA | {len(files)} | {files[0]['path']} |")
    A("")

    A("## PRESENT (in store)")
    A("")
    for fam in sorted(present):
        if fam in missing: continue
        A(f"### {fam}")
        files = sorted(by_fam[fam], key=lambda r: -r["size_mb"])[:40]
        A("")
        A("| Path | Rows | Size MB | Coverage | Last Mod | Cols (head) |")
        A("|---|---:|---:|---|---|---|")
        for r in files:
            cols_str = ", ".join(r["cols"][:8])
            cov = r["coverage"] or "—"
            A(f"| `{r['path']}` | {r['rows']} | {r['size_mb']} | {cov} | {r['mtime']} | {cols_str} |")
        if len(by_fam[fam]) > 40:
            A(f"_…and {len(by_fam[fam])-40} more files in this family._")
        A("")

    A("## PARTIAL (have something, but gaps)")
    A("")
    if partial:
        for fam in partial:
            A(f"- `{fam}` — only smoke / tiny file present, real ingest needed.")
    else:
        A("_(none flagged)_")
    A("")

    A("## MISSING (mandated by HC #563 R2 but not on disk)")
    A("")
    for fam in missing:
        A(f"- `{fam}` — {MANDATED[fam]}")
    A("")

    A("## Tick-Level / MBO Rollups (not stock-picking input)")
    A("")
    for r in mbo:
        A(f"- `{r['subdir']}`: {r['n_files']} files, {r['size_mb']} MB total, dates {r['coverage'] or '—'}")
    A("")

    A("## Cross-Node Snapshots")
    A("")
    A("### Neptune (nick@neptune)")
    A("")
    A("```")
    for p, out in neptune.items():
        A(f"$ {p}")
        A(out.rstrip()[:1500])
        A("")
    A("```")
    A("")
    A("### Razer (claude@razer)")
    A("")
    A("```")
    for p, out in razer.items():
        A(f"$ {p}")
        A(out.rstrip()[:1500])
        A("")
    A("```")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}  ({len(rows)} files, {len(missing)} missing families)")

if __name__ == "__main__":
    main()
