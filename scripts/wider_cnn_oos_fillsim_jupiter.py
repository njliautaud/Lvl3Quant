#!/usr/bin/env python3
"""
wider_cnn_oos_fillsim_jupiter.py
Run fill_sim on wider CNN OOS WF predictions (Dec 2025 - Mar 2026).

Steps:
1. Extract per-day arrays from consolidated OOS npz → individual {date}_preds.npz
2. Optionally apply meta-model LGBM gate (zero out signals where meta confidence
   is below META_GATE_THRESH) — this is the +30% IC lift gate
3. Run fill_sim_cli on matching MBO dates
4. Summarize results

BUG FIXES (2026-03-29):
  - FIX 1: r.get("total_pnl", 0) → r.get("total_pnl_dollars", 0)
    The Rust fill_sim_cli outputs "total_pnl_dollars", NOT "total_pnl".
    This was causing all 10 configs to show 0 PnL.
  - FIX 2: r.get("fill_rate_pct", 0) → r.get("fill_rate", 0)
    The Rust binary outputs "fill_rate", not "fill_rate_pct".
  - FIX 3: Added per-config win_rate tracking (was missing entirely).
  - FIX 4: Added robust pnl/trade extraction using extract_pnl() pattern
    that handles all known fill_sim output key variants.
  - FIX 5 (NEW): Meta-model LGBM gate wired in.
    CNN alone doesn't survive execution costs. Meta-model gates out
    low-confidence signals before they reach fill_sim, yielding +30% IC lift.
    Enabled when META_GATE_DIR is set and per-day meta pred files exist.

Self-directing: runs nohup in background, logs to /tmp/oos_fillsim.log
"""

import os
import json
import math
import time
import subprocess
import logging
import tempfile
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OOS_NPZ = Path("/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_oot_20260311_092055.npz")
PER_DAY_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/per_day_oos")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/oos_wf_fill_sim")
SUMMARY_FILE = OUTPUT_DIR / "oos_wf_summary.json"

# Meta-model gate: set this path to enable the +30% IC gate.
# Expected: per-day .npz files with key 'meta_confidence' (float32 array, same length as CNN preds).
# Files must be named {date}_meta.npz  (e.g. 2026-01-15_meta.npz).
# If directory doesn't exist or files are missing, gate is silently skipped for that day.
META_GATE_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/meta_model_preds")
# Gate threshold: only take trades where meta_confidence > this value.
# From session 38: z>2.0 on CNN alone = Sortino 2.046. Meta gate adds +30% IC on top.
# Start conservative at 0.55 (slightly above 0.5 random), escalate if fill rate too low.
META_GATE_THRESH = 0.55

WORKERS = 4  # Jupiter is CPU-only, use more workers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("/tmp/oos_fillsim.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("oos_fillsim")

# Card configs to test - our best known configs
# NOTE: signal thresholds are z-scores (CNN output is centered z-score ~N(0,1)).
# sig=0.1 means |z| > 0.1 to fire (very permissive; meta gate does the real filtering).
# sig=0.0 means accept all non-zero preds (use when preds are already meta-gated).
CONFIGS = [
    {"name": "oos_tp13_h2h",        "tp": 13, "hold_ms": 7200000,  "sig": 0.1,  "latency": 10},
    {"name": "oos_tp8_h1h_mae20",   "tp": 8,  "hold_ms": 3600000,  "sig": 0.1,  "latency": 10, "mae_ticks": 20, "mae_secs": 300},
    {"name": "oos_tp15_h2h",        "tp": 15, "hold_ms": 7200000,  "sig": 0.1,  "latency": 10},
    {"name": "oos_tp20_sig03",      "tp": 20, "hold_ms": 7200000,  "sig": 0.3,  "latency": 10},
    {"name": "oos_tp25_h2h",        "tp": 25, "hold_ms": 7200000,  "sig": 0.1,  "latency": 10},
    {"name": "oos_scalp_tp6_5m",    "tp": 6,  "hold_ms": 300000,   "sig": 0.3,  "latency": 10},
    {"name": "oos_tp13_prime",      "tp": 13, "hold_ms": 7200000,  "sig": 0.1,  "latency": 10, "prime": True},
    {"name": "oos_tp13_chase",      "tp": 13, "hold_ms": 7200000,  "sig": 0.1,  "latency": 10, "chase": True},
    {"name": "oos_tp8_sigflip",     "tp": 8,  "hold_ms": 3600000,  "sig": 0.1,  "latency": 10, "sigflip": True},
    {"name": "oos_tp13_prime_chase","tp": 13, "hold_ms": 7200000,  "sig": 0.15, "latency": 10, "prime": True, "chase": True},
    # Meta-gated variants — CNN signals pre-filtered by meta-model confidence
    # signal-threshold 0.0 because meta gate already zeroed out weak signals
    {"name": "oos_meta_tp13_h2h",        "tp": 13, "hold_ms": 7200000,  "sig": 0.0,  "latency": 10, "meta_gate": True},
    {"name": "oos_meta_tp13_prime_chase","tp": 13, "hold_ms": 7200000,  "sig": 0.0,  "latency": 10, "prime": True, "chase": True, "meta_gate": True},
    {"name": "oos_meta_tp20_h2h",        "tp": 20, "hold_ms": 7200000,  "sig": 0.0,  "latency": 10, "meta_gate": True},
    {"name": "oos_meta_scalp_tp6_5m",    "tp": 6,  "hold_ms": 300000,   "sig": 0.0,  "latency": 10, "meta_gate": True},
]

# Gated-prediction cache: {date: gated_preds_array}.
# Populated lazily in extract_per_day_meta(). Avoids reloading meta files per-config.
_meta_cache: dict = {}


def _robust_pnl(r: dict) -> float:
    """Extract total PnL (dollars) from fill_sim JSON output.

    The Rust fill_sim_cli uses 'total_pnl_dollars' as the primary key.
    Handles legacy variants for robustness.

    FIX 1: The original code used r.get("total_pnl", 0) which always
    returned 0 because the Rust binary outputs "total_pnl_dollars".
    """
    if r is None:
        return 0.0
    for key in ("total_pnl_dollars", "total_pnl", "net_pnl_dollars", "net_pnl", "pnl"):
        val = r.get(key)
        if val is not None:
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            return float(val)
    return 0.0


def _robust_trades(r: dict) -> int:
    """Extract total filled trade count from fill_sim JSON output."""
    if r is None:
        return 0
    for key in ("total_trades", "total_filled", "n_trades", "num_trades"):
        val = r.get(key)
        if val is not None and isinstance(val, (int, float)) and val >= 0:
            return int(val)
    return 0


def _robust_wr(r: dict) -> float:
    """Extract win rate from fill_sim JSON output.

    FIX 3: Win rate was never extracted in the original summarize loop.
    """
    if r is None:
        return 0.0
    for key in ("win_rate", "wr"):
        val = r.get(key)
        if val is not None and isinstance(val, (int, float)) and not math.isnan(float(val)):
            return float(val)
    # Compute from wins/total if available
    total = _robust_trades(r)
    if total > 0:
        for key in ("total_wins", "n_wins", "wins"):
            wins = r.get(key)
            if wins is not None:
                return int(wins) / total
    return 0.0


def _robust_fill_rate(r: dict) -> float:
    """Extract fill rate from fill_sim JSON output.

    FIX 2: Original code used r.get("fill_rate_pct", 0) but the Rust binary
    outputs "fill_rate" (a decimal 0.0-1.0), not "fill_rate_pct".
    """
    if r is None:
        return 0.0
    for key in ("fill_rate", "fill_rate_pct"):
        val = r.get(key)
        if val is not None and isinstance(val, (int, float)) and not math.isnan(float(val)):
            v = float(val)
            # fill_rate_pct is 0-100, fill_rate is 0.0-1.0; normalise to 0-100 for display
            return v * 100.0 if v <= 1.0 else v
    # Compute from filled/signals if possible
    filled = r.get("total_filled", 0) or 0
    signals = r.get("total_signals", 0) or 0
    if signals > 0:
        return filled / signals * 100.0
    return 0.0


def extract_per_day(oos_npz_path, out_dir):
    """Extract per-day preds from consolidated npz → individual files with 'predictions' key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(str(oos_npz_path))
    keys = list(d.keys())
    dates = sorted(set(k.replace("_preds", "").replace("_mid", "") for k in keys))
    log.info(f"Found {len(dates)} dates in OOS npz")

    created = []
    for date in dates:
        out_file = out_dir / f"{date}_preds.npz"
        if out_file.exists():
            created.append(date)
            continue
        if f"{date}_preds" not in d:
            continue
        preds = d[f"{date}_preds"].astype(np.float32)
        np.savez(str(out_file), predictions=preds)
        created.append(date)

    log.info(f"Per-day files ready: {len(created)} dates")
    return created


def load_meta_gate(date: str, n_bars: int) -> np.ndarray | None:
    """Load meta-model confidence for a given date.

    Returns float32 array of shape (n_bars,) with confidence scores 0-1,
    or None if meta gate data not available for this date.

    Meta-model predictions are expected at:
      META_GATE_DIR / {date}_meta.npz   with key 'meta_confidence'
    or alternatively key 'predictions' (for backwards compatibility).

    The meta-model is a LightGBM trained on book features to predict
    CNN signal reliability (+30% IC lift as described in session 38).
    """
    if not META_GATE_DIR.exists():
        return None

    meta_file = META_GATE_DIR / f"{date}_meta.npz"
    if not meta_file.exists():
        return None

    try:
        md = np.load(str(meta_file))
        for key in ("meta_confidence", "confidence", "predictions", "preds"):
            if key in md:
                arr = md[key].astype(np.float32)
                if len(arr) == n_bars:
                    return arr
                # Length mismatch — try to align (pad/truncate)
                log.warning(f"Meta gate length mismatch for {date}: {len(arr)} vs {n_bars}. Skipping gate.")
                return None
    except Exception as e:
        log.warning(f"Failed to load meta gate for {date}: {e}")
    return None


def apply_meta_gate(preds: np.ndarray, date: str) -> tuple[np.ndarray, int, int]:
    """Apply meta-model confidence gate to CNN predictions.

    Zeros out predictions where meta_confidence <= META_GATE_THRESH.
    Returns (gated_preds, n_original_signals, n_gated_signals).

    FIX 5: Meta-model gate integration.
    CNN alone is unprofitable after execution costs (from session notes).
    Meta-model gate filters to high-confidence signals only.
    """
    n_original = int((np.abs(preds) > 1e-7).sum())
    meta = load_meta_gate(date, len(preds))
    if meta is None:
        return preds, n_original, n_original  # no gate available

    gate_mask = meta > META_GATE_THRESH
    gated = np.where(gate_mask, preds, 0.0).astype(np.float32)
    n_gated = int((np.abs(gated) > 1e-7).sum())
    return gated, n_original, n_gated


def run_one(args):
    """Run fill_sim for one (date, config) pair.

    Handles both plain CNN and meta-gated variants.
    Meta-gated variants write a temp pred file with zeroed-out low-confidence bars.
    """
    date, cfg = args
    date_nodash = date.replace("-", "")
    mbo_file = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
    pred_file = PER_DAY_DIR / f"{date}_preds.npz"
    out_file = OUTPUT_DIR / f"{date}_{cfg['name']}.json"

    if not mbo_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no_mbo"}
    if not pred_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no_pred"}

    # Use cached result if valid
    if out_file.exists():
        try:
            with open(out_file) as f:
                data = json.load(f)
            # Validate: cached result must have a sensible pnl field (not all zeros from old bug)
            if _robust_pnl(data) != 0.0 or _robust_trades(data) == 0:
                data.update({"date": date, "config": cfg["name"]})
                return data
            # If pnl==0 AND trades>0, the cached file is from the buggy run — rerun.
            if _robust_trades(data) > 0 and _robust_pnl(data) == 0.0:
                log.warning(f"Stale zero-PnL cache for {date}/{cfg['name']}, rerunning")
                out_file.unlink()
        except Exception:
            pass

    # Determine which pred file to use (plain or meta-gated)
    use_pred_file = pred_file
    meta_info = {}

    if cfg.get("meta_gate"):
        # Load base preds, apply gate, write temp file
        try:
            base_preds = np.load(str(pred_file))["predictions"].astype(np.float32)
            gated_preds, n_orig, n_gated = apply_meta_gate(base_preds, date)
            meta_info = {"meta_gate_thresh": META_GATE_THRESH,
                         "n_signals_before_gate": n_orig,
                         "n_signals_after_gate": n_gated,
                         "gate_retention_pct": round(n_gated / max(n_orig, 1) * 100, 1)}

            if n_gated == 0:
                # No signals pass the gate for this day — return empty result
                return {"date": date, "config": cfg["name"],
                        "total_pnl_dollars": 0.0, "total_trades": 0,
                        "win_rate": 0.0, "fill_rate": 0.0,
                        **meta_info}

            # Write gated preds to a temp file (cleaned up after fill_sim)
            tmp_dir = Path(tempfile.mkdtemp(prefix="meta_gate_"))
            tmp_pred = tmp_dir / f"{date}_{cfg['name']}_gated.npz"
            np.savez(str(tmp_pred), predictions=gated_preds)
            use_pred_file = tmp_pred
        except Exception as e:
            log.warning(f"Meta gate failed for {date}/{cfg['name']}: {e}. Using raw preds.")
            meta_info = {"meta_gate_error": str(e)}
            tmp_dir = None
    else:
        tmp_dir = None

    cmd = [
        "nice", "-n", "19",
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(use_pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(cfg["sig"]),
        "--hold-ms", str(cfg["hold_ms"]),
        "--latency-ms", str(cfg.get("latency", 10)),
    ]
    if cfg.get("tp"):
        cmd += ["--take-profit-ticks", str(cfg["tp"])]
    if cfg.get("mae_ticks"):
        cmd += ["--trailing-ticks", str(cfg["mae_ticks"])]
    if cfg.get("prime"):
        cmd.append("--prime-hours")
    if cfg.get("chase"):
        cmd.append("--chase-entry")
    if cfg.get("sigflip"):
        cmd.append("--signal-flip-exit")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Clean up temp meta-gate pred file
        if tmp_dir is not None:
            try:
                for f in tmp_dir.iterdir():
                    f.unlink()
                tmp_dir.rmdir()
            except Exception:
                pass

        if result.returncode != 0:
            return {"date": date, "config": cfg["name"],
                    "error": f"fill_sim failed: {result.stderr[:200]}"}
        with open(out_file) as f:
            data = json.load(f)
        data.update({"date": date, "config": cfg["name"]})
        if meta_info:
            data.update(meta_info)
        return data
    except subprocess.TimeoutExpired:
        return {"date": date, "config": cfg["name"], "error": "timeout"}
    except Exception as e:
        return {"date": date, "config": cfg["name"], "error": str(e)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== OOS WF Fill Sim - Wider CNN - Dec2025-Mar2026 ===")
    log.info(f"OOS NPZ: {OOS_NPZ}")
    log.info(f"Meta gate: {'ENABLED (thresh={})'.format(META_GATE_THRESH) if META_GATE_DIR.exists() else 'DISABLED (dir not found: ' + str(META_GATE_DIR) + ')'}")

    # Step 1: Extract per-day files
    log.info("Extracting per-day pred files...")
    dates = extract_per_day(OOS_NPZ, PER_DAY_DIR)

    # Filter to dates with MBO data
    valid_dates = []
    for date in dates:
        date_nodash = date.replace("-", "")
        mbo = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
        if mbo.exists():
            valid_dates.append(date)

    log.info(f"Dates with both preds+MBO: {len(valid_dates)}")
    log.info(f"Date range: {valid_dates[0]} to {valid_dates[-1]}")

    # Filter CONFIGS: skip meta_gate variants if meta dir doesn't exist
    active_configs = []
    for cfg in CONFIGS:
        if cfg.get("meta_gate") and not META_GATE_DIR.exists():
            log.info(f"Skipping meta-gate config '{cfg['name']}' (META_GATE_DIR not found)")
            continue
        active_configs.append(cfg)
    log.info(f"Active configs: {len(active_configs)} (of {len(CONFIGS)} total)")

    # Step 2: Build task list
    tasks = [(date, cfg) for date in valid_dates for cfg in active_configs]
    log.info(f"Total tasks: {len(tasks)} ({len(valid_dates)} dates x {len(active_configs)} configs)")

    # Step 3: Run
    results = []
    done = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            done += 1
            if done % 20 == 0:
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                log.info(f"Progress: {done}/{len(tasks)} ({done/len(tasks)*100:.1f}%) ETA: {eta/60:.1f}min")

    # Step 4: Summarize by config
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_config[r["config"]].append(r)

    errors_by_type: dict = defaultdict(int)
    for r in results:
        if "error" in r:
            errors_by_type[r["error"]] += 1

    if errors_by_type:
        log.info(f"Errors by type: {dict(errors_by_type)}")

    summary = {}
    for cfg_name, runs in sorted(by_config.items()):
        # FIX 1+2+3: Use robust extractors instead of raw dict.get() with wrong key names.
        # OLD (BUGGY): pnls = [r.get("total_pnl", 0) for r in runs]   ← always 0!
        # OLD (BUGGY): fills = [r.get("fill_rate_pct", 0) for r in runs]  ← always 0!
        # NEW (FIXED): use _robust_pnl / _robust_fill_rate / _robust_wr
        pnls   = [_robust_pnl(r) for r in runs]
        trades = [_robust_trades(r) for r in runs]
        wrs    = [_robust_wr(r) for r in runs]
        fills  = [_robust_fill_rate(r) for r in runs]

        total_pnl    = sum(pnls)
        total_trades = sum(trades)
        days         = len(runs)
        avg_daily    = total_pnl / days if days > 0 else 0

        # Weighted average win rate (weight by trade count)
        if total_trades > 0:
            avg_wr = sum(w * t for w, t in zip(wrs, trades)) / total_trades
        else:
            avg_wr = 0.0

        avg_fill_rate = sum(fills) / len(fills) if fills else 0.0

        daily_arr = np.array(pnls)
        sharpe  = (float(daily_arr.mean()) / (float(daily_arr.std()) + 1e-9)) * (252 ** 0.5) if len(daily_arr) > 1 else 0.0
        downside = daily_arr[daily_arr < 0]
        sortino = (float(daily_arr.mean()) / (float(downside.std()) + 1e-9)) * (252 ** 0.5) if len(downside) > 1 else 0.0
        pct_pos_days = float((daily_arr > 0).mean())

        # Meta gate stats (if applicable)
        meta_stats = {}
        if any("n_signals_after_gate" in r for r in runs):
            n_before = sum(r.get("n_signals_before_gate", 0) for r in runs)
            n_after  = sum(r.get("n_signals_after_gate", 0) for r in runs)
            meta_stats = {
                "meta_gate_thresh": META_GATE_THRESH,
                "meta_signals_before": n_before,
                "meta_signals_after":  n_after,
                "meta_retention_pct":  round(n_after / max(n_before, 1) * 100, 1),
            }

        summary[cfg_name] = {
            "total_pnl_dollars":  round(total_pnl, 2),
            "avg_daily_pnl":      round(avg_daily, 2),
            "sharpe_annual":      round(float(sharpe), 3),
            "sortino_annual":     round(float(sortino), 3),
            "pct_profitable_days": round(pct_pos_days, 3),
            "total_trades":       total_trades,
            "avg_win_rate":       round(avg_wr, 4),
            "avg_fill_rate_pct":  round(avg_fill_rate, 1),
            "days":               days,
            **meta_stats,
        }
        log.info(
            f"{cfg_name}: PnL=${total_pnl:.0f} Sharpe={sharpe:.2f} Sortino={sortino:.2f} "
            f"WR={avg_wr:.1%} Fill={avg_fill_rate:.1f}% Trades={total_trades} Days={days}"
        )

    with open(SUMMARY_FILE, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    log.info(f"Summary saved to {SUMMARY_FILE}")
    log.info(f"Total time: {(time.time()-start)/60:.1f}min")

    # Print top configs by Sortino
    ranked = sorted(summary.items(), key=lambda x: x[1]["sortino_annual"], reverse=True)
    log.info("\n=== TOP CONFIGS BY SORTINO ===")
    for name, s in ranked[:8]:
        meta_note = f" [meta_gate ret={s.get('meta_retention_pct','N/A')}%]" if "meta_gate_thresh" in s else ""
        log.info(
            f"  {name}: Sortino={s['sortino_annual']:.3f} PnL=${s['total_pnl_dollars']:.0f} "
            f"WR={s['avg_win_rate']:.1%} Fill={s['avg_fill_rate_pct']:.1f}%{meta_note}"
        )

    # Sanity check: warn if any config shows 0 trades (possible signal threshold issue)
    zero_trade_configs = [n for n, s in summary.items() if s["total_trades"] == 0]
    if zero_trade_configs:
        log.warning(
            f"WARNING: {len(zero_trade_configs)} configs had 0 trades. "
            f"Check signal thresholds and pred file alignment: {zero_trade_configs}"
        )


if __name__ == "__main__":
    main()
