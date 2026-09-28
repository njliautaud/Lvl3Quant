#!/usr/bin/env python3
"""
comprehensive_wf_sweep_jupiter.py
Comprehensive strategy backtesting sweep on 68 OOS WF days.

SWEEP DIMENSIONS (total: ~34 configs x 68 days = 2312 fill_sim runs):

1. TP/SL Grid (12 configs, best of TP15-h2h and TP13-prime-chase lineage):
   - TP: 8, 10, 13, 15, 18, 20 ticks
   - SL: 15, 20 ticks (2 SL values per TP → 12 combos)

2. Entry threshold sweep (6 configs):
   - z-score thresholds: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5

3. Time-of-day gating (4 configs):
   - Full day (no window)
   - Morning only: 09:30–11:00
   - Midday: 11:00–14:00
   - Afternoon: 14:00–16:00

4. Dynamic exit variants (4 configs):
   - Signal reversal exit (--signal-flip-exit)
   - Trailing stop 2 ticks
   - Trailing stop 4 ticks
   - Trailing stop 6 ticks

5. Hold time sweep (5 configs):
   - 30s, 60s, 120s, 300s, 600s max hold (prime+chase base)

6. Chase variants (3 configs):
   - Passive only (no chase)
   - Chase max 2t/5r (default)
   - Chase aggressive: 4t/10r + force-cross

Anchors: TP15-h2h (Sortino 5.70) and TP13-prime-chase (Sortino 5.21).

Run:  nohup python3 comprehensive_wf_sweep_jupiter.py > /tmp/comp_sweep.log 2>&1 &
"""

import os
import json
import math
import time
import logging
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

FILL_SIM    = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR     = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PER_DAY_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/per_day_oos")
OUTPUT_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/oos_wf_fill_sim/comprehensive_sweep")
SUMMARY_FILE = Path("/home/jupiter/Lvl3Quant/data/processed/oos_wf_fill_sim/comprehensive_sweep_summary.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Jupiter CPU: 16 cores, load already ~37 from existing jobs.
# New jobs are fast (Rust binary) so 24 workers is safe saturation on top.
WORKERS = 24

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("/tmp/comp_sweep.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("comp_sweep")

# ── Data helpers ──────────────────────────────────────────────────────────────

def _mbo_path(date_str: str) -> Path:
    d = date_str.replace("-", "")
    return MBO_DIR / f"glbx-mdp3-{d}.mbo.dbn.zst"

def _pred_path(date_str: str) -> Path:
    return PER_DAY_DIR / f"{date_str}_preds.npz"

def _available_dates():
    dates = []
    for p in sorted(PER_DAY_DIR.glob("*_preds.npz")):
        d = p.name.replace("_preds.npz", "")
        if _mbo_path(d).exists():
            dates.append(d)
    return dates

# ── Config definitions ────────────────────────────────────────────────────────

def build_configs():
    configs = []

    # ── 1. TP/SL GRID ─────────────────────────────────────────────────
    # Both anchors: h2h (no prime, no chase) and prime+chase
    for tp in [8, 10, 13, 15, 18, 20]:
        for sl in [15, 20]:
            for style, extra in [
                ("h2h",         []),
                ("prime_chase", ["--prime-hours", "--chase-entry"]),
            ]:
                configs.append({
                    "name":     f"tpsl_tp{tp}_sl{sl}_{style}",
                    "tp":       tp,
                    "sl":       sl,
                    "sig":      0.1,
                    "latency":  10,
                    "hold_ms":  7_200_000,
                    "extra":    extra,
                    "group":    "tpsl_grid",
                })

    # ── 2. ENTRY THRESHOLD SWEEP ──────────────────────────────────────
    # Both styles, TP15 as baseline (our best TP from initial scan)
    for thresh_str, sig in [("z10", 0.1), ("z15", 0.15), ("z20", 0.2),
                             ("z25", 0.25), ("z30", 0.3), ("z35", 0.35)]:
        for style, extra in [
            ("h2h",         []),
            ("prime_chase", ["--prime-hours", "--chase-entry"]),
        ]:
            configs.append({
                "name":     f"threshold_{thresh_str}_{style}",
                "tp":       15,
                "sl":       20,
                "sig":      sig,
                "latency":  10,
                "hold_ms":  7_200_000,
                "extra":    extra,
                "group":    "threshold_sweep",
            })

    # ── 3. TIME-OF-DAY GATING ─────────────────────────────────────────
    tod_windows = [
        ("full_day",   "",      ""),
        ("morning",    "09:30", "11:00"),
        ("midday",     "11:00", "14:00"),
        ("afternoon",  "14:00", "16:00"),
    ]
    for window_name, tw_start, tw_end in tod_windows:
        extra = []
        if tw_start:
            extra += [f"--time-window-start {tw_start}", f"--time-window-end {tw_end}"]
        # Both passive and prime+chase
        for style, style_extra in [
            ("passive",     []),
            ("prime_chase", ["--prime-hours", "--chase-entry"]),
        ]:
            all_extra = extra + style_extra
            configs.append({
                "name":     f"tod_{window_name}_{style}",
                "tp":       15,
                "sl":       20,
                "sig":      0.1,
                "latency":  10,
                "hold_ms":  7_200_000,
                "extra":    all_extra,
                "group":    "tod_gating",
            })

    # ── 4. DYNAMIC EXIT VARIANTS ──────────────────────────────────────
    # Signal reversal exit
    configs.append({
        "name":     "dyn_sigflip_tp15",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--signal-flip-exit"],
        "group":    "dynamic_exit",
    })
    # Trailing stops (2t, 4t, 6t, 8t)
    for trail in [2, 4, 6, 8]:
        configs.append({
            "name":     f"dyn_trail{trail}t_tp15",
            "tp":       15,
            "sl":       0,   # trailing stop replaces fixed SL
            "sig":      0.1,
            "latency":  10,
            "hold_ms":  7_200_000,
            "extra":    [f"--trailing-ticks {trail}"],
            "group":    "dynamic_exit",
        })
    # Conviction exit (delayed signal flip: 30 bars = 3s)
    configs.append({
        "name":     "dyn_conviction30b_tp15",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--conviction-exit-bars 30", "--conviction-exit-mag 0.3"],
        "group":    "dynamic_exit",
    })

    # ── 5. HOLD TIME SWEEP ────────────────────────────────────────────
    for hold_sec in [30, 60, 120, 300, 600]:
        configs.append({
            "name":     f"hold_{hold_sec}s_prime_chase",
            "tp":       15,
            "sl":       20,
            "sig":      0.1,
            "latency":  10,
            "hold_ms":  hold_sec * 1000,
            "extra":    ["--prime-hours", "--chase-entry"],
            "group":    "hold_time",
        })
    # Also h2h variants to compare
    for hold_sec in [120, 300, 600]:
        configs.append({
            "name":     f"hold_{hold_sec}s_h2h",
            "tp":       15,
            "sl":       20,
            "sig":      0.1,
            "latency":  10,
            "hold_ms":  hold_sec * 1000,
            "extra":    [],
            "group":    "hold_time",
        })

    # ── 6. CHASE VARIANTS ─────────────────────────────────────────────
    # Passive only (no chase)
    configs.append({
        "name":     "chase_passive_prime",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--prime-hours"],
        "group":    "chase_variants",
    })
    # Chase default (2t/5r)
    configs.append({
        "name":     "chase_default_prime",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--prime-hours", "--chase-entry",
                     "--chase-max-ticks 2", "--chase-max-reprices 5"],
        "group":    "chase_variants",
    })
    # Chase aggressive (4t/10r + force-cross)
    configs.append({
        "name":     "chase_aggressive_prime",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--prime-hours", "--chase-entry",
                     "--chase-max-ticks 4", "--chase-max-reprices 10",
                     "--chase-force-cross"],
        "group":    "chase_variants",
    })
    # Market entry baseline (100% fill rate, pays spread)
    configs.append({
        "name":     "market_entry_prime",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--prime-hours", "--market-entry"],
        "group":    "chase_variants",
    })

    # ── 7. POSITION SIZING ────────────────────────────────────────────
    # Fixed 2 contracts (scale PnL comparison)
    configs.append({
        "name":     "size2_prime_chase",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--prime-hours", "--chase-entry", "--size 2"],
        "group":    "position_sizing",
    })
    # Fixed 3 contracts
    configs.append({
        "name":     "size3_prime_chase",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--prime-hours", "--chase-entry", "--size 3"],
        "group":    "position_sizing",
    })

    # ── 8. LATENCY SENSITIVITY ────────────────────────────────────────
    for lat_ms in [0, 5, 20, 50, 100]:
        configs.append({
            "name":     f"latency_{lat_ms}ms_prime_chase",
            "tp":       15,
            "sl":       20,
            "sig":      0.1,
            "latency":  lat_ms,
            "hold_ms":  7_200_000,
            "extra":    ["--prime-hours", "--chase-entry"],
            "group":    "latency_sensitivity",
        })

    # ── 9. RATCHET STOP (lock in profits) ────────────────────────────
    configs.append({
        "name":     "ratchet_prime_chase",
        "tp":       15,
        "sl":       20,
        "sig":      0.1,
        "latency":  10,
        "hold_ms":  7_200_000,
        "extra":    ["--prime-hours", "--chase-entry", "--ratchet-stop"],
        "group":    "ratchet_stop",
    })

    return configs


# ── Per-job runner ─────────────────────────────────────────────────────────────

def run_one(date_str: str, cfg: dict) -> dict:
    mbo   = _mbo_path(date_str)
    pred  = _pred_path(date_str)
    name  = cfg["name"]
    out   = OUTPUT_DIR / f"{date_str}_{name}.json"

    # Skip if already done
    if out.exists():
        try:
            with open(out) as f:
                r = json.load(f)
            r["config"] = name
            r["date"]   = date_str
            r["group"]  = cfg.get("group", "")
            r["_cached"] = True
            return r
        except Exception:
            out.unlink(missing_ok=True)

    if not mbo.exists() or not pred.exists():
        return {"error": "missing_files", "date": date_str, "config": name,
                "mbo": str(mbo), "pred": str(pred)}

    # Build command
    cmd = [
        FILL_SIM,
        "--mbo-file",     str(mbo),
        "--predictions",  str(pred),
        "--output",       str(out),
        "--signal-threshold", str(cfg["sig"]),
        "--latency-ms",   str(cfg["latency"]),
        "--hold-ms",      str(cfg["hold_ms"]),
        "--quiet",
    ]

    tp = cfg.get("tp", 0)
    sl = cfg.get("sl", 0)
    if tp > 0:
        cmd += ["--take-profit-ticks", str(tp)]
    if sl > 0:
        cmd += ["--stop-loss-ticks", str(sl)]

    # Append extra flags (handle flags that may contain spaces)
    for flag in cfg.get("extra", []):
        cmd.extend(flag.split())

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            return {
                "error":   "fill_sim_failed",
                "exitcode": res.returncode,
                "stderr":  res.stderr[-500:],
                "date":    date_str,
                "config":  name,
            }
        if out.exists():
            with open(out) as f:
                r = json.load(f)
            r["config"] = name
            r["date"]   = date_str
            r["group"]  = cfg.get("group", "")
            return r
        else:
            return {"error": "no_output_file", "date": date_str, "config": name}
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "date": date_str, "config": name}
    except Exception as e:
        return {"error": str(e), "date": date_str, "config": name}


# ── Summarisation helpers ──────────────────────────────────────────────────────

def _pnl(r):
    for k in ("total_pnl_dollars", "total_pnl", "net_pnl_dollars", "net_pnl", "pnl"):
        v = r.get(k)
        if v is not None:
            try:
                f = float(v)
                if not math.isnan(f) and not math.isinf(f):
                    return f
            except Exception:
                pass
    return 0.0

def _trades(r):
    for k in ("total_trades", "total_filled", "n_trades"):
        v = r.get(k)
        if isinstance(v, (int, float)) and v >= 0:
            return int(v)
    return 0

def _wr(r):
    for k in ("win_rate", "wr"):
        v = r.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    total = _trades(r)
    if total > 0:
        for k in ("total_wins", "n_wins", "wins"):
            w = r.get(k)
            if w is not None:
                return int(w) / total
    return 0.0

def _fill_rate(r):
    for k in ("fill_rate", "fill_rate_pct"):
        v = r.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    total_signals = r.get("total_signals") or r.get("total_posted", 0)
    filled = r.get("total_filled", 0)
    if total_signals and filled:
        return filled / total_signals
    return 0.0

def summarize(results):
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_cfg[r["config"]].append(r)

    summary = {}
    for cfg_name in sorted(by_cfg):
        runs = by_cfg[cfg_name]
        pnls   = [_pnl(r) for r in runs]
        trades = [_trades(r) for r in runs]
        wrs    = [_wr(r) for r in runs]
        fills  = [_fill_rate(r) for r in runs]

        total_pnl    = sum(pnls)
        total_trades = sum(trades)
        days         = len(runs)
        avg_daily    = total_pnl / days if days > 0 else 0.0

        total_t = sum(trades)
        avg_wr  = (sum(w * t for w, t in zip(wrs, trades)) / total_t
                   if total_t > 0 else 0.0)
        avg_fill = sum(fills) / len(fills) if fills else 0.0

        arr = np.array(pnls)
        sharpe  = 0.0
        sortino = 0.0
        if len(arr) > 1 and arr.std() > 1e-9:
            sharpe = float(arr.mean() / arr.std() * (252 ** 0.5))
        neg = arr[arr < 0]
        if len(neg) > 1 and neg.std() > 1e-9:
            sortino = float(arr.mean() / neg.std() * (252 ** 0.5))

        summary[cfg_name] = {
            "total_pnl_dollars":  round(total_pnl, 2),
            "avg_daily_pnl":      round(avg_daily, 2),
            "sharpe_annual":      round(sharpe, 3),
            "sortino_annual":     round(sortino, 3),
            "pct_profitable_days": round(float((arr > 0).mean()), 3),
            "total_trades":       total_trades,
            "avg_win_rate":       round(avg_wr, 4),
            "avg_fill_rate_pct":  round(avg_fill * 100, 1),
            "days":               days,
            "group":              runs[0].get("group", ""),
        }

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    dates   = _available_dates()
    configs = build_configs()

    log.info(f"Dates: {len(dates)}  Configs: {len(configs)}")
    log.info(f"Total jobs: {len(dates) * len(configs)}")
    log.info(f"Dates range: {dates[0]} — {dates[-1]}")
    log.info(f"Workers: {WORKERS}")

    # Build job list (skip already-done ones)
    jobs = []
    skipped = 0
    for date in dates:
        for cfg in configs:
            out = OUTPUT_DIR / f"{date}_{cfg['name']}.json"
            if out.exists():
                skipped += 1
            else:
                jobs.append((date, cfg))

    log.info(f"Jobs queued: {len(jobs)}  (skipped already done: {skipped})")
    if not jobs:
        log.info("All jobs already complete. Re-summarizing.")

    results = []
    # Load cached results
    for date in dates:
        for cfg in configs:
            out = OUTPUT_DIR / f"{date}_{cfg['name']}.json"
            if out.exists():
                try:
                    with open(out) as f:
                        r = json.load(f)
                    r["config"] = cfg["name"]
                    r["date"]   = date
                    r["group"]  = cfg.get("group", "")
                    results.append(r)
                except Exception:
                    pass

    # Run remaining jobs
    done = 0
    errors = 0
    total = len(jobs)
    t_report = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_one, d, c): (d, c) for d, c in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if "error" in result:
                errors += 1
            done += 1
            # Progress every 60s or every 500 jobs
            if time.time() - t_report > 60 or done % 500 == 0:
                pct = done / total * 100 if total > 0 else 100
                elapsed = time.time() - start
                eta_min = (elapsed / done * (total - done)) / 60 if done > 0 and total > done else 0
                log.info(
                    f"Progress: {done}/{total} ({pct:.0f}%)  errors={errors}  "
                    f"elapsed={elapsed/60:.1f}min  ETA={eta_min:.0f}min"
                )
                t_report = time.time()

    log.info(f"All jobs complete. Results: {len(results)}  errors: {errors}")

    # Summarize
    summary = summarize(results)

    # Save
    with open(SUMMARY_FILE, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    log.info(f"Summary saved to {SUMMARY_FILE}")

    # Print rankings
    ranked = sorted(summary.items(), key=lambda x: x[1]["sortino_annual"], reverse=True)
    log.info("\n=== TOP 20 CONFIGS BY SORTINO ===")
    for name, s in ranked[:20]:
        log.info(
            f"  [{s['group']:<20}] {name:<45} "
            f"Sortino={s['sortino_annual']:>7.3f}  "
            f"PnL=${s['total_pnl_dollars']:>10,.0f}  "
            f"WR={s['avg_win_rate']:.1%}  "
            f"Fill={s['avg_fill_rate_pct']:>5.1f}%  "
            f"Trades={s['total_trades']:>5d}"
        )

    # Print by group
    log.info("\n=== BEST PER GROUP ===")
    from collections import defaultdict
    by_group = defaultdict(list)
    for name, s in summary.items():
        by_group[s["group"]].append((name, s))
    for group in sorted(by_group):
        best = sorted(by_group[group], key=lambda x: x[1]["sortino_annual"], reverse=True)[0]
        name, s = best
        log.info(
            f"  {group:<25}: {name:<45} Sortino={s['sortino_annual']:.3f}  "
            f"PnL=${s['total_pnl_dollars']:,.0f}"
        )

    total_time = (time.time() - start) / 60
    log.info(f"\nTotal time: {total_time:.1f} min")


if __name__ == "__main__":
    main()
