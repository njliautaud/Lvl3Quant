#!/usr/bin/env python3
"""
Combined Portfolio Tracker — V5 CSP + ETF Rotation v3
======================================================
Tracks the combined portfolio (50/50 or Risk Parity allocation).
Reads state from both sub-engines, computes combined metrics,
logs combined equity curve, and alerts if rebalancing needed.

Run via PM2 or cron (daily at market close).

Key finding: 16.6% correlation → Sharpe 3.83 (risk parity), 3.56 (50/50).
Reference: SESSION_STATE item #64, backtest over 8.2 years (2018-2026).

Outputs:
    live_trading_linux/portfolio_combo_state/
        - combined_state.json    — latest combined NAV, weights, drift
        - equity_curve.jsonl     — daily combined NAV snapshots
        - rebal_log.jsonl        — rebalance events
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant/live_trading_linux")
V5_STATE = ROOT / "wheel_v5_state"
ETF_STATE = ROOT / "etf_rotation_v3_state"
COMBO_STATE = ROOT / "portfolio_combo_state"
COMBO_STATE.mkdir(parents=True, exist_ok=True)

STATE_FILE = COMBO_STATE / "combined_state.json"
EQUITY_LOG = COMBO_STATE / "equity_curve.jsonl"
REBAL_LOG = COMBO_STATE / "rebal_log.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────
ALLOCATION_MODE = "risk_parity"  # "equal" or "risk_parity"
TARGET_WEIGHTS = {"v5_csp": 0.50, "etf_rotation_v3": 0.50}  # equal weight default
REBAL_DRIFT_THRESHOLD = 0.10  # 10% drift from target triggers rebal alert
STARTING_CAPITAL = 200_000.0  # Combined notional ($100K each)

# Risk parity lookback for vol estimation
RP_LOOKBACK_DAYS = 60


def load_v5_nav() -> Optional[float]:
    """Load latest V5 CSP NAV from nav_history.json."""
    nav_file = V5_STATE / "nav_history.json"
    if not nav_file.exists():
        return None
    try:
        with open(nav_file) as f:
            history = json.load(f)
        if not history:
            return None
        latest = history[-1]
        return float(latest.get("nav", latest.get("nav_usd", 0)))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def load_v5_nav_series() -> list[dict]:
    """Load V5 NAV history for vol calculation."""
    nav_file = V5_STATE / "nav_history.json"
    if not nav_file.exists():
        return []
    try:
        with open(nav_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, TypeError):
        return []


def load_etf_nav() -> Optional[float]:
    """Load latest ETF Rotation v3 NAV from state.json."""
    state_file = ETF_STATE / "state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file) as f:
            state = json.load(f)
        return float(state.get("nav_usd", 0))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def load_etf_nav_series() -> list[dict]:
    """Load ETF NAV history for vol calculation."""
    eq_file = ETF_STATE / "equity_curve.jsonl"
    if not eq_file.exists():
        return []
    records = []
    try:
        with open(eq_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, TypeError):
        pass
    return records


def compute_risk_parity_weights(
    v5_series: list[dict],
    etf_series: list[dict],
    lookback: int = RP_LOOKBACK_DAYS,
) -> dict[str, float]:
    """
    Compute inverse-vol weights for risk parity.
    Uses last N daily returns to estimate annualized vol.
    Falls back to 50/50 if insufficient data.
    """
    # Extract NAV series
    v5_navs = [entry.get("nav", entry.get("nav_usd", 0)) for entry in v5_series]
    etf_navs = [entry.get("nav", entry.get("nav_usd", 0)) for entry in etf_series]

    # Need at least 20 data points
    if len(v5_navs) < 20 or len(etf_navs) < 20:
        return {"v5_csp": 0.50, "etf_rotation_v3": 0.50}

    # Use last N points
    v5_recent = np.array(v5_navs[-lookback:], dtype=float)
    etf_recent = np.array(etf_navs[-lookback:], dtype=float)

    # Daily returns
    v5_rets = np.diff(v5_recent) / v5_recent[:-1]
    etf_rets = np.diff(etf_recent) / etf_recent[:-1]

    # Annualized vol
    v5_vol = np.std(v5_rets) * np.sqrt(252) if len(v5_rets) > 5 else 0.15
    etf_vol = np.std(etf_rets) * np.sqrt(252) if len(etf_rets) > 5 else 0.15

    # Inverse vol weights
    if v5_vol <= 0 or etf_vol <= 0:
        return {"v5_csp": 0.50, "etf_rotation_v3": 0.50}

    inv_v5 = 1.0 / v5_vol
    inv_etf = 1.0 / etf_vol
    total_inv = inv_v5 + inv_etf

    w_v5 = inv_v5 / total_inv
    w_etf = inv_etf / total_inv

    return {"v5_csp": round(w_v5, 4), "etf_rotation_v3": round(w_etf, 4)}


def compute_combined_metrics(equity_log_path: Path) -> dict:
    """Compute Sharpe, Sortino, MaxDD from combined equity curve."""
    if not equity_log_path.exists():
        return {}

    records = []
    with open(equity_log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) < 5:
        return {"note": f"Only {len(records)} data points, need 5+ for metrics"}

    navs = np.array([r["combined_nav"] for r in records], dtype=float)
    rets = np.diff(navs) / navs[:-1]

    if len(rets) < 2:
        return {}

    # Annualized metrics
    mean_ret = np.mean(rets) * 252
    std_ret = np.std(rets) * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

    # Sortino (downside deviation)
    neg_rets = rets[rets < 0]
    downside_std = np.std(neg_rets) * np.sqrt(252) if len(neg_rets) > 0 else std_ret
    sortino = mean_ret / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(navs)
    dd = (navs - cummax) / cummax
    max_dd = float(np.min(dd))

    # CAGR
    days = len(navs)
    total_ret = navs[-1] / navs[0] - 1
    cagr = (1 + total_ret) ** (252 / max(days, 1)) - 1 if days > 0 else 0.0

    return {
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "cagr_pct": round(cagr * 100, 1),
        "total_return_pct": round(total_ret * 100, 2),
        "n_days": days,
    }


def run_tracker():
    """Main tracking cycle."""
    now = datetime.now(timezone.utc).isoformat()

    # Load current NAVs
    v5_nav = load_v5_nav()
    etf_nav = load_etf_nav()

    if v5_nav is None or etf_nav is None:
        print(f"[{now}] ERROR: Cannot read NAVs (V5={v5_nav}, ETF={etf_nav})")
        return

    # Compute weights
    if ALLOCATION_MODE == "risk_parity":
        v5_series = load_v5_nav_series()
        etf_series = load_etf_nav_series()
        weights = compute_risk_parity_weights(v5_series, etf_series)
    else:
        weights = TARGET_WEIGHTS.copy()

    # Combined NAV (normalized to starting capital * weight)
    # Each engine started at $100K. Combined portfolio = sum of both.
    combined_nav = v5_nav + etf_nav

    # Current actual weights
    actual_w_v5 = v5_nav / combined_nav if combined_nav > 0 else 0.5
    actual_w_etf = etf_nav / combined_nav if combined_nav > 0 else 0.5

    # Drift from target
    drift_v5 = abs(actual_w_v5 - weights["v5_csp"])
    drift_etf = abs(actual_w_etf - weights["etf_rotation_v3"])
    max_drift = max(drift_v5, drift_etf)

    rebal_needed = max_drift > REBAL_DRIFT_THRESHOLD

    # Combined return
    combined_return_pct = (combined_nav / STARTING_CAPITAL - 1) * 100

    # Log equity curve
    eq_entry = {
        "timestamp": now,
        "combined_nav": round(combined_nav, 2),
        "v5_nav": round(v5_nav, 2),
        "etf_nav": round(etf_nav, 2),
        "actual_weights": {"v5_csp": round(actual_w_v5, 4), "etf_rotation_v3": round(actual_w_etf, 4)},
        "target_weights": weights,
        "max_drift": round(max_drift, 4),
    }
    with open(EQUITY_LOG, "a") as f:
        f.write(json.dumps(eq_entry) + "\n")

    # Compute running metrics
    metrics = compute_combined_metrics(EQUITY_LOG)

    # Save state
    state = {
        "last_update": now,
        "combined_nav": round(combined_nav, 2),
        "combined_return_pct": round(combined_return_pct, 2),
        "v5_nav": round(v5_nav, 2),
        "etf_nav": round(etf_nav, 2),
        "allocation_mode": ALLOCATION_MODE,
        "target_weights": weights,
        "actual_weights": {"v5_csp": round(actual_w_v5, 4), "etf_rotation_v3": round(actual_w_etf, 4)},
        "max_drift": round(max_drift, 4),
        "rebal_needed": rebal_needed,
        "metrics": metrics,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    # Log rebal event if needed
    if rebal_needed:
        rebal_entry = {
            "timestamp": now,
            "max_drift": round(max_drift, 4),
            "actual": {"v5_csp": round(actual_w_v5, 4), "etf_rotation_v3": round(actual_w_etf, 4)},
            "target": weights,
            "action": "ALERT — drift exceeds threshold, consider rebalancing",
        }
        with open(REBAL_LOG, "a") as f:
            f.write(json.dumps(rebal_entry) + "\n")

    # Print summary
    print(f"\n{'='*60}")
    print(f"COMBINED PORTFOLIO — {now[:10]}")
    print(f"{'='*60}")
    print(f"  Combined NAV:    ${combined_nav:,.2f} ({combined_return_pct:+.2f}%)")
    print(f"  V5 CSP:          ${v5_nav:,.2f} (weight: {actual_w_v5:.1%})")
    print(f"  ETF Rotation v3: ${etf_nav:,.2f} (weight: {actual_w_etf:.1%})")
    print(f"  Target weights:  V5={weights['v5_csp']:.1%}, ETF={weights['etf_rotation_v3']:.1%}")
    print(f"  Max drift:       {max_drift:.2%} {'⚠️ REBAL NEEDED' if rebal_needed else '✓ OK'}")
    if metrics:
        print(f"  Metrics:         Sharpe={metrics.get('sharpe','?')}, "
              f"Sortino={metrics.get('sortino','?')}, "
              f"MaxDD={metrics.get('max_dd_pct','?')}%, "
              f"Return={metrics.get('total_return_pct','?')}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_tracker()
