"""
GA Signal Fill-Sim Pipeline
============================
Generates signals from top GA-discovered formulas, uploads to Jupiter,
and runs them through the MBO fill_sim_cli.

GA Formulas (best from walk-forward):
  Fold 3 (best): max(dwfi_20, depth_ratio_l1) → test IC=+0.105
  Fold 2:        add(depth_ratio_l1, dwfi_5)   → test IC=+0.089
  Fold 4:        max(ofi_l3_5, depth_ratio_l1) → test IC=+0.091
  Ensemble:      mean of z-scores from all 3, re-normalised

Feature indices in the 340-column mbo_features matrix:
  dwfi_20       → 271
  dwfi_5        → 270
  depth_ratio_l1→ 130
  ofi_l3_5      → 272

Usage:
    python alpha_discovery/ga_signal_fillsim.py
"""

import sys
import os
import json
import time
import tempfile
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

# Real-time output
sys.stdout.reconfigure(line_buffering=True)

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, 'C:/Users/Footb/Documents/Github/teleclaude-main')
sys.path.insert(0, 'C:/Users/Footb/Documents/Github/Lvl3Quant')

import numpy as np

try:
    from utils.ssh_exec import connect_jupiter, sftp_upload, sftp_download
    SSH_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: SSH utils not available: {e}")
    SSH_AVAILABLE = False

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path('C:/Users/Footb/Documents/Github/Lvl3Quant')
FEAT_DIR = ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIG_DIR  = ROOT / 'data' / 'processed' / 'ga_signal_cache'
RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
SIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Jupiter paths
JUPITER_SIG_DIR   = '/home/jupiter/lvl3quant/data/processed/ga_signal_cache'
JUPITER_RESULTS   = '/home/jupiter/lvl3quant/alpha_discovery/results'
JUPITER_BINARY    = '/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli'
JUPITER_MBO_DIR   = '/home/jupiter/lvl3quant/data/mbo'
# MBO files are .dbn (not .dbn.zst) on Jupiter
JUPITER_MBO_EXT   = '.mbo.dbn'

# ─── Feature indices (from get_feature_names()) ───────────────────────────────
IDX_DWFI20       = 271  # dwfi_20
IDX_DWFI5        = 270  # dwfi_5
IDX_DEPTH_L1     = 130  # depth_ratio_l1
IDX_OFI_L3_5     = 272  # ofi_l3_5

# ─── Simulation configs ───────────────────────────────────────────────────────
SIM_CONFIGS = [
    # (label, threshold, hold_ms, trailing_ticks)
    ('t3.5_h900s', 3.5, 900_000, 8),
    ('t3.5_h600s', 3.5, 600_000, 8),
]

# ─── Discord send helper ───────────────────────────────────────────────────────
def discord(msg: str):
    """Send a message via Discord (best-effort; never crash on failure)."""
    try:
        import subprocess as sp
        # We call the MCP discord tool via a small node snippet
        # (teleclaude has the discord MCP registered).
        # Actually we just print — the orchestrator is watching stdout.
        # For real async updates we'll rely on the MCP tool being called
        # from the outer agent. Just print for now and the outer session sees it.
        print(f"[DISCORD] {msg}")
    except Exception:
        pass


# ─── Formula implementations ─────────────────────────────────────────────────

def zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise to std=1.0; handle NaN/Inf and degenerate columns."""
    arr = arr.astype(np.float64)
    # Replace Inf with NaN for stats
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() < 10:
        return np.zeros_like(arr)
    finite_vals = arr[finite_mask]
    mean = float(np.mean(finite_vals))
    std  = float(np.std(finite_vals))
    if std < 1e-10:
        out = np.zeros_like(arr)
        out[~finite_mask] = 0.0
        return out
    out = (arr - mean) / std
    # Set NaN/Inf positions back to 0 (no signal for warmup bars)
    out[~finite_mask] = 0.0
    return out


def formula_ga_max_dwfi20(features: np.ndarray) -> np.ndarray:
    """Fold 3 best: max(dwfi_20, depth_ratio_l1) — test IC=+0.105"""
    a = features[:, IDX_DWFI20].astype(np.float64)
    b = features[:, IDX_DEPTH_L1].astype(np.float64)
    return np.maximum(a, b)


def formula_ga_add_dwfi5(features: np.ndarray) -> np.ndarray:
    """Fold 2: add(depth_ratio_l1, dwfi_5) — test IC=+0.089"""
    a = features[:, IDX_DEPTH_L1].astype(np.float64)
    b = features[:, IDX_DWFI5].astype(np.float64)
    return a + b


def formula_ga_max_ofi(features: np.ndarray) -> np.ndarray:
    """Fold 4: max(ofi_l3_5, depth_ratio_l1) — test IC=+0.091"""
    a = features[:, IDX_OFI_L3_5].astype(np.float64)
    b = features[:, IDX_DEPTH_L1].astype(np.float64)
    return np.maximum(a, b)


def formula_ga_ensemble(features: np.ndarray) -> np.ndarray:
    """Ensemble: mean of z-scores from the top 3, then re-normalised."""
    z1 = zscore(formula_ga_max_dwfi20(features))
    z2 = zscore(formula_ga_add_dwfi5(features))
    z3 = zscore(formula_ga_max_ofi(features))
    return zscore((z1 + z2 + z3) / 3.0)


FORMULAS = {
    'ga_max_dwfi20': formula_ga_max_dwfi20,
    'ga_add_dwfi5':  formula_ga_add_dwfi5,
    'ga_max_ofi':    formula_ga_max_ofi,
    'ga_ensemble':   formula_ga_ensemble,
}


# ─── Signal generation ────────────────────────────────────────────────────────

def get_available_dates() -> list[str]:
    """Return sorted list of dates that have a feature cache file."""
    files = sorted(FEAT_DIR.glob('*_mbo_features.npz'))
    dates = []
    for f in files:
        # filename: YYYY-MM-DD_mbo_features.npz
        stem = f.stem  # YYYY-MM-DD_mbo_features
        date = stem.replace('_mbo_features', '')
        dates.append(date)
    return dates


def generate_signals_for_date(date: str, formula_name: str, formula_fn) -> Path | None:
    """
    Load features for one date, apply formula, z-score, save as NPZ.
    Returns path to saved file, or None on error.
    """
    out_path = SIG_DIR / f'{formula_name}_{date}.npz'
    if out_path.exists():
        # Validate cached signal is non-trivial (std > 0 means real signal)
        try:
            cached = np.load(str(out_path))['predictions']
            if cached.std() > 0.01:
                return out_path
            # Bad cache (all-zeros) — regenerate
            out_path.unlink()
        except Exception:
            out_path.unlink(missing_ok=True)

    feat_file = FEAT_DIR / f'{date}_mbo_features.npz'
    if not feat_file.exists():
        return None

    try:
        data = np.load(str(feat_file))
        features = data['mbo_features']  # (N, 340) float32

        raw_signal = formula_fn(features)
        signal = zscore(raw_signal)  # normalise to std=1.0

        # Replace NaN/Inf with 0
        signal = np.where(np.isfinite(signal), signal, 0.0)

        np.savez_compressed(str(out_path), predictions=signal)
        del features, raw_signal, signal, data
        return out_path

    except Exception as e:
        print(f"  ERROR generating {formula_name} for {date}: {e}")
        return None


# ─── Jupiter SSH helpers ──────────────────────────────────────────────────────

def jupiter_exec(cmd: str, timeout: int = 120) -> dict:
    """Run a command on Jupiter, return result dict."""
    return connect_jupiter(cmd, prefer='ethernet', timeout=timeout)


def ensure_remote_dir(remote_dir: str):
    """Create directory on Jupiter if it doesn't exist."""
    result = jupiter_exec(f'mkdir -p {remote_dir}')
    if not result['success']:
        print(f"  WARNING: could not create remote dir {remote_dir}: {result.get('error','')}")


def upload_signal_file(local_path: Path, remote_dir: str) -> str | None:
    """Upload a local NPZ to Jupiter. Returns remote path or None."""
    remote_path = f"{remote_dir}/{local_path.name}"
    result = sftp_upload(str(local_path), remote_path, prefer='ethernet')
    if result['success']:
        return remote_path
    print(f"  SFTP upload failed for {local_path.name}: {result}")
    return None


def run_fill_sim_remote(
    remote_pred: str,
    date: str,
    cfg_label: str,
    threshold: float,
    hold_ms: int,
    trailing: int,
) -> dict | None:
    """
    Run fill_sim_cli on Jupiter for one (date, config) combo.
    Returns parsed JSON result or None.
    """
    date_nodash = date.replace('-', '')
    mbo_file = f'{JUPITER_MBO_DIR}/glbx-mdp3-{date_nodash}{JUPITER_MBO_EXT}'
    out_file  = f'/tmp/ga_result_{date_nodash}_{cfg_label}.json'

    cmd = (
        f'{JUPITER_BINARY}'
        f' --mbo-file {mbo_file}'
        f' --predictions {remote_pred}'
        f' --output {out_file}'
        f' --signal-threshold {threshold}'
        f' --hold-ms {hold_ms}'
        f' --trailing-ticks {trailing}'
        f' --quiet'
    )

    result = jupiter_exec(cmd, timeout=300)
    if not result['success']:
        print(f"    SSH fail: {result.get('error','')}")
        return None
    if result['exit_code'] != 0:
        stderr = result.get('stderr', '')[:300]
        print(f"    fill_sim error (exit {result['exit_code']}): {stderr}")
        return None

    # Download result JSON
    local_tmp = Path(tempfile.mktemp(suffix='.json'))
    dl = sftp_download(out_file, str(local_tmp), prefer='ethernet')
    if not dl['success']:
        print(f"    Could not download result JSON from {out_file}")
        return None

    try:
        with open(local_tmp) as f:
            parsed = json.load(f)
        local_tmp.unlink(missing_ok=True)
        return parsed
    except Exception as e:
        print(f"    JSON parse error: {e}")
        local_tmp.unlink(missing_ok=True)
        return None


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def compute_sharpe(daily_pnls: list[float]) -> float:
    arr = np.array(daily_pnls, dtype=float)
    if len(arr) < 2:
        return 0.0
    std = np.std(arr)
    if std < 1e-6:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(252))


def aggregate(daily_results: list[dict]) -> dict:
    """Aggregate a list of per-day fill_sim results."""
    pnls   = [r.get('total_pnl_dollars', 0.0) for r in daily_results]
    trades = [r.get('total_trades', 0)         for r in daily_results]
    wr     = [r.get('win_rate', 0.0)           for r in daily_results]

    total_pnl    = sum(pnls)
    total_trades = sum(trades)
    n_days       = len(pnls)
    sharpe       = compute_sharpe(pnls)
    avg_win_rate = float(np.mean(wr)) if wr else 0.0
    profitable_days = sum(1 for p in pnls if p > 0)

    return {
        'total_pnl':      round(total_pnl, 2),
        'avg_daily_pnl':  round(total_pnl / max(n_days, 1), 2),
        'sharpe':         round(sharpe, 4),
        'n_days':         n_days,
        'profitable_days': profitable_days,
        'win_pct_days':   round(profitable_days / max(n_days, 1), 4),
        'total_trades':   total_trades,
        'avg_win_rate':   round(avg_win_rate, 4),
        'daily_pnls':     [round(p, 2) for p in pnls],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"=== GA Signal Fill-Sim Pipeline  [{ts}] ===")

    if not SSH_AVAILABLE:
        print("ERROR: SSH utilities not available. Cannot upload to Jupiter.")
        return

    # 1. Check Jupiter connectivity
    print("\n[1/5] Checking Jupiter connection...")
    ping = jupiter_exec('echo ok', timeout=15)
    if not ping['success']:
        print(f"FATAL: Jupiter unreachable: {ping.get('error','')}")
        discord("GA fill_sim FAILED: Jupiter unreachable.")
        return
    print(f"  Connected via {ping['connection']} @ {ping['host']}")

    # 2. Ensure remote dirs exist
    ensure_remote_dir(JUPITER_SIG_DIR)
    ensure_remote_dir(JUPITER_RESULTS)
    print(f"  Remote dirs ready: {JUPITER_SIG_DIR}")

    # 3. Discover available dates
    print("\n[2/5] Discovering available feature dates...")
    all_dates = get_available_dates()
    print(f"  Found {len(all_dates)} dates: {all_dates[0]} .. {all_dates[-1]}")

    # Check which MBO files exist on Jupiter (list all .dbn files)
    mbo_list_result = jupiter_exec(
        f'ls {JUPITER_MBO_DIR}/glbx-mdp3-*.mbo.dbn 2>/dev/null | xargs -I{{}} basename {{}}',
        timeout=30
    )
    if mbo_list_result['success'] and mbo_list_result['stdout'].strip():
        # Parse filenames like: glbx-mdp3-20250714.mbo.dbn
        jupiter_mbo_dates = set()
        for line in mbo_list_result['stdout'].strip().split('\n'):
            line = line.strip()
            if 'glbx-mdp3-' in line:
                date_part = line.split('glbx-mdp3-')[1].split('.')[0]  # YYYYMMDD
                if len(date_part) == 8:
                    # Convert YYYYMMDD -> YYYY-MM-DD
                    d = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    jupiter_mbo_dates.add(d)
        valid_dates = [d for d in all_dates if d in jupiter_mbo_dates]
        print(f"  Jupiter MBO files found: {len(jupiter_mbo_dates)}")
    else:
        # Fall back: assume all dates have MBO
        valid_dates = all_dates
        print("  WARNING: Could not list Jupiter MBO files — assuming all available")
    print(f"  Dates with Jupiter MBO: {len(valid_dates)}")

    if not valid_dates:
        print("ERROR: No valid dates found with MBO data on Jupiter.")
        discord("GA fill_sim: No valid MBO dates found on Jupiter.")
        return

    # 4. Generate signals, upload, run fill_sim
    print("\n[3/5] Generating signals, uploading, running fill_sim...")

    all_results = {}   # { formula_name: { cfg_label: [per_day_result, ...] } }

    for formula_name, formula_fn in FORMULAS.items():
        print(f"\n--- Formula: {formula_name} ---")
        discord(f"GA pipeline: processing formula {formula_name} ({len(valid_dates)} dates)...")

        all_results[formula_name] = {cfg[0]: [] for cfg in SIM_CONFIGS}
        errors = 0

        for i, date in enumerate(valid_dates):
            print(f"  [{i+1}/{len(valid_dates)}] {date}", end='  ')

            # Generate local signal file
            local_sig = generate_signals_for_date(date, formula_name, formula_fn)
            if local_sig is None:
                print(f"SKIP (no feature file)")
                errors += 1
                continue

            # Upload to Jupiter
            remote_sig = upload_signal_file(local_sig, JUPITER_SIG_DIR)
            if remote_sig is None:
                print(f"SKIP (upload failed)")
                errors += 1
                continue

            # Run each sim config
            for cfg_label, threshold, hold_ms, trailing in SIM_CONFIGS:
                result = run_fill_sim_remote(
                    remote_sig, date, cfg_label,
                    threshold, hold_ms, trailing
                )
                if result is not None:
                    result['date'] = date
                    all_results[formula_name][cfg_label].append(result)
                # Small pause to avoid flooding Jupiter
                time.sleep(0.05)

            cfg_line = ' | '.join(
                f"{cfg[0]}:${all_results[formula_name][cfg[0]][-1].get('total_pnl_dollars', 0):.0f}"
                for cfg in SIM_CONFIGS
                if all_results[formula_name][cfg[0]]
            )
            print(cfg_line if cfg_line else 'no results')

        print(f"  {formula_name}: {len(valid_dates) - errors}/{len(valid_dates)} days processed, {errors} errors")

    # 5. Aggregate and display results
    print("\n[4/5] Aggregating results...")

    summary = {}
    for formula_name in FORMULAS:
        summary[formula_name] = {}
        for cfg_label, _, _, _ in SIM_CONFIGS:
            day_results = all_results[formula_name][cfg_label]
            if day_results:
                summary[formula_name][cfg_label] = aggregate(day_results)
            else:
                summary[formula_name][cfg_label] = {'total_pnl': 0, 'sharpe': 0, 'n_days': 0, 'error': 'no data'}

    # Print table
    print("\n" + "="*90)
    print(f"{'FORMULA':<20} {'CONFIG':<15} {'TOTAL PnL':>12} {'AVG/DAY':>10} {'SHARPE':>8} {'DAYS':>5} {'WIN%':>6} {'TRADES':>8}")
    print("="*90)

    for formula_name in FORMULAS:
        for cfg_label, _, _, _ in SIM_CONFIGS:
            r = summary[formula_name][cfg_label]
            if 'error' in r and r.get('n_days', 0) == 0:
                print(f"{formula_name:<20} {cfg_label:<15} {'NO DATA':>12}")
                continue
            print(
                f"{formula_name:<20} {cfg_label:<15}"
                f" ${r.get('total_pnl',0):>10,.2f}"
                f" ${r.get('avg_daily_pnl',0):>8,.2f}"
                f" {r.get('sharpe',0):>8.3f}"
                f" {r.get('n_days',0):>5}"
                f" {r.get('win_pct_days',0)*100:>5.1f}%"
                f" {r.get('total_trades',0):>8,}"
            )

    print("="*90)

    # 6. Save results JSON
    print("\n[5/5] Saving results...")
    out_file = RESULTS_DIR / f'ga_signal_fillsim_{ts}.json'
    results_payload = {
        'timestamp': ts,
        'formulas': list(FORMULAS.keys()),
        'sim_configs': [c[0] for c in SIM_CONFIGS],
        'n_dates': len(valid_dates),
        'dates': valid_dates,
        'summary': summary,
        'raw': {
            fn: {
                cfg: [
                    {k: v for k, v in d.items() if k != 'trades'}  # skip trades array
                    for d in all_results[fn][cfg]
                ]
                for cfg, _, _, _ in SIM_CONFIGS
            }
            for fn in FORMULAS
        },
    }

    with open(out_file, 'w') as f:
        json.dump(results_payload, f, indent=2)
    print(f"  Results saved to: {out_file}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min.")

    # Final Discord summary
    best_pnl = max(
        (summary[fn][cfg].get('total_pnl', -1e9) for fn in FORMULAS for cfg, *_ in SIM_CONFIGS),
        default=0
    )
    best_sharpe = max(
        (summary[fn][cfg].get('sharpe', -1e9) for fn in FORMULAS for cfg, *_ in SIM_CONFIGS),
        default=0
    )
    discord_msg = (
        f"GA fill_sim COMPLETE in {elapsed/60:.1f}min | "
        f"Best PnL: ${best_pnl:,.0f} | Best Sharpe: {best_sharpe:.3f} | "
        f"Results: {out_file.name}"
    )
    discord(discord_msg)

    return summary


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
