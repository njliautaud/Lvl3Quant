#!/usr/bin/env python3
"""
Unified Daily Portfolio Allocator v2
====================================
Reads 5 signal sources, determines current regime, maps regime to strategy
weights, applies kill-switch guards, and outputs a daily allocation with
drift report and plain-English reasoning.

Signal sources:
  1. Kill switch monitor       — pause flags
  2. Cross-asset dashboard     — regime (risk, inflation, credit, dollar)
  3. Strategy regime learner   — rules for what works when
  4. Portfolio simulator       — Monte Carlo optimal weights (baseline)
  5. Sector momentum monitor   — sector leadership & breadth

Run:  python3 portfolio_allocator_v2.py [--json] [--verbose]
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LVL3 = Path("/home/jupiter/Lvl3Quant")
CROSS_ASSET = LVL3 / "output/growth_research/cross_asset_dashboard/latest_snapshot.json"
REGIME_LEARNER = LVL3 / "output/growth_research/strategy_regime_learner/regime_analysis.json"
PORTFOLIO_SIM = LVL3 / "output/growth_research/portfolio_simulator/simulation_results.json"
SECTOR_MONITOR = LVL3 / "output/growth_research/industry_monitor/latest_snapshot.json"
OUT_DIR = LVL3 / "output/growth_research/portfolio_allocator"
DAILY_FILE = OUT_DIR / "daily_allocation.json"
HISTORY_FILE = OUT_DIR / "allocation_history.csv"

# ---------------------------------------------------------------------------
# Baseline weights from Monte Carlo D_Max_Sharpe optimizer
# Rounded to the user's stated baseline:
#   50% Megacap Mom, 25% Strangle, 17% ETF Rotation, 7% Wheel
# ---------------------------------------------------------------------------
STRATEGIES = ["Megacap_Momentum", "Strangle", "ETF_Rotation", "Wheel_CSP",
              "BPS", "Iron_Condor", "Crash_Hedge", "Cash"]
BASELINE = {
    "Megacap_Momentum": 0.50,
    "Strangle":         0.25,
    "ETF_Rotation":     0.17,
    "Wheel_CSP":        0.07,
    "BPS":              0.00,
    "Iron_Condor":      0.00,
    "Crash_Hedge":      0.00,
    "Cash":             0.01,
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: cannot read {path.name}: {e}")
        return None


def load_kill_switches() -> dict:
    """Try importing the kill switch monitor; fall back to safe defaults."""
    try:
        sys.path.insert(0, str(LVL3 / "live_trading_linux"))
        from kill_switch_monitor import check_kill_switches
        return check_kill_switches(force_refresh=True)
    except Exception as e:
        return {
            "pause_all": False,
            "pause_premium": False,
            "pause_rotation": False,
            "pause_strangles": False,
            "reduce_50pct": False,
            "reasons": [f"Kill-switch unavailable: {e}"],
            "raw": {},
            "checked_at": datetime.now().isoformat(),
            "error": True,
        }


def load_cross_asset() -> dict:
    return _load_json(CROSS_ASSET) or {}


def load_regime_learner() -> dict:
    return _load_json(REGIME_LEARNER) or {}


def load_portfolio_sim() -> dict:
    return _load_json(PORTFOLIO_SIM) or {}


def load_sector_monitor() -> dict:
    return _load_json(SECTOR_MONITOR) or {}


def load_previous() -> dict | None:
    return _load_json(DAILY_FILE)


# ---------------------------------------------------------------------------
# Regime determination
# ---------------------------------------------------------------------------

def determine_regime(cross_asset: dict, sector: dict, kill_sw: dict) -> dict:
    """
    Classify current market into composite regime from cross-asset + sector data.
    Returns dict with regime dimensions and a composite label.
    """
    regime = {
        "risk":       "mixed",
        "volatility": "normal_vol",
        "inflation":  "neutral",
        "credit":     "credit_easy",
        "trend":      "neutral",
        "breadth":    "healthy",
        "composite":  "NORMAL",
    }

    signals = cross_asset.get("regime_signals", {})

    # --- Risk regime ---
    risk_sig = signals.get("risk_regime", {}).get("signal", "MIXED").upper()
    if "ON" in risk_sig:
        regime["risk"] = "risk_on"
    elif "OFF" in risk_sig:
        regime["risk"] = "risk_off"
    else:
        regime["risk"] = "mixed"

    # --- Volatility from kill-switch raw data or market snapshot ---
    vix = None
    raw = kill_sw.get("raw", {})
    if raw.get("vix"):
        vix = raw["vix"]
    else:
        snap = cross_asset.get("asset_metrics", {}).get("^VIX", {})
        if snap:
            vix = snap.get("price")
    if vix is not None:
        if vix >= 25:
            regime["volatility"] = "high_vol"
        elif vix <= 15:
            regime["volatility"] = "low_vol"
        else:
            regime["volatility"] = "normal_vol"
    regime["vix"] = vix

    # --- Inflation ---
    infl = signals.get("inflation", {}).get("signal", "").upper()
    if "INFLAT" in infl and "DE" not in infl:
        regime["inflation"] = "inflationary"
    elif "DEFLAT" in infl or "DIS" in infl:
        regime["inflation"] = "deflationary"
    else:
        regime["inflation"] = "neutral"

    # --- Credit ---
    credit = signals.get("credit", {}).get("signal", "").upper()
    if "WIDEN" in credit or "STRESS" in credit:
        regime["credit"] = "credit_stress"
    else:
        regime["credit"] = "credit_easy"

    # --- SPY trend from regime learner market snapshot ---
    mkt = cross_asset.get("asset_metrics", {}).get("SPY", {})
    if mkt:
        trend = mkt.get("trend", "").lower()
        if "strong_up" in trend:
            regime["trend"] = "strong_uptrend"
        elif "up" in trend:
            regime["trend"] = "uptrend"
        elif "down" in trend:
            regime["trend"] = "downtrend"
        else:
            regime["trend"] = "neutral"

    # --- Sector breadth ---
    above_200 = sector.get("sectors_above_200sma", 0)
    total_sectors = len(sector.get("sectors", {})) or 11
    breadth_pct = above_200 / total_sectors if total_sectors else 0
    if breadth_pct >= 0.7:
        regime["breadth"] = "healthy"
    elif breadth_pct >= 0.4:
        regime["breadth"] = "mixed"
    else:
        regime["breadth"] = "narrow"
    regime["breadth_pct"] = round(breadth_pct * 100, 1)

    # --- Composite label ---
    if kill_sw.get("pause_all"):
        regime["composite"] = "CRISIS"
    elif regime["risk"] == "risk_off" or regime["trend"] == "downtrend":
        regime["composite"] = "RISK_OFF"
    elif regime["volatility"] == "high_vol":
        regime["composite"] = "HIGH_VOL"
    elif regime["risk"] == "risk_on" and regime["trend"] in ("uptrend", "strong_uptrend"):
        regime["composite"] = "RISK_ON"
    elif regime["volatility"] == "low_vol" and regime["trend"] in ("uptrend", "strong_uptrend"):
        regime["composite"] = "LOW_VOL_BULL"
    else:
        regime["composite"] = "NORMAL"

    return regime


# ---------------------------------------------------------------------------
# Weight adjustments
# ---------------------------------------------------------------------------

def compute_weights(regime: dict, kill_sw: dict, sector: dict) -> tuple[dict, list[str]]:
    """
    Start from baseline weights and apply regime-driven adjustments.
    Returns (weights_dict, list_of_adjustment_reasons).
    """
    w = dict(BASELINE)
    reasons = []
    vix = regime.get("vix")

    # ── 1. Volatility adjustments ──
    if vix is not None and vix >= 22:
        # VIX rising toward 25+: shift FROM momentum TO premium selling
        shift_pct = min(0.15, (vix - 20) * 0.03)
        taken_from_megacap = min(w["Megacap_Momentum"], shift_pct * 0.6)
        taken_from_rotation = min(w["ETF_Rotation"], shift_pct * 0.4)
        total_shift = taken_from_megacap + taken_from_rotation
        w["Megacap_Momentum"] -= taken_from_megacap
        w["ETF_Rotation"] -= taken_from_rotation
        w["Strangle"] += total_shift * 0.5
        w["Wheel_CSP"] += total_shift * 0.5
        reasons.append(f"VIX at {vix:.1f}: shifted {total_shift*100:.1f}pp from momentum to premium selling")

    if vix is not None and vix <= 14:
        # Low vol: thin premiums, favor momentum
        shift = min(w["Strangle"], 0.05)
        w["Strangle"] -= shift
        w["Megacap_Momentum"] += shift * 0.6
        w["ETF_Rotation"] += shift * 0.4
        reasons.append(f"VIX at {vix:.1f}: shifted {shift*100:.1f}pp from strangle to momentum (thin premiums)")

    # ── 2. Risk-off adjustments ──
    if regime["risk"] == "risk_off" or regime["composite"] == "RISK_OFF":
        cut = min(w["Megacap_Momentum"], 0.15)
        w["Megacap_Momentum"] -= cut
        w["Crash_Hedge"] += cut * 0.3
        w["Wheel_CSP"] += cut * 0.4  # defensive wheel
        w["Cash"] += cut * 0.3
        reasons.append(f"Risk-off: reduced megacap by {cut*100:.1f}pp, added crash hedge + defensive wheel + cash")

    # ── 3. Credit stress ──
    if regime["credit"] == "credit_stress":
        cut_bps = min(w["BPS"], w["BPS"])
        cut_strangle = min(w["Strangle"], 0.05)
        w["BPS"] -= cut_bps
        w["Strangle"] -= cut_strangle
        w["Cash"] += cut_bps + cut_strangle
        reasons.append(f"Credit widening: reduced BPS/strangle, added {(cut_bps+cut_strangle)*100:.1f}pp to cash")

    # ── 4. Inflationary regime ──
    if regime["inflation"] == "inflationary":
        # Check if energy/commodity sectors are leading
        leaders = sector.get("leaders_1m", [])
        energy_leading = any("Energy" in str(l) or "XLE" in str(l) for l in leaders)
        if energy_leading:
            boost = 0.03
            w["ETF_Rotation"] += boost
            w["Cash"] = max(0, w["Cash"] - boost)
            reasons.append("Inflationary + energy leading: boosted ETF rotation for commodity sector exposure")
        else:
            reasons.append("Inflationary regime noted but energy not leading sectors")

    # ── 5. Downtrend adjustments ──
    if regime["trend"] == "downtrend":
        # Sector momentum loses badly in bears (Sharpe -0.92)
        cut_mega = min(w["Megacap_Momentum"], 0.15)
        cut_rot = min(w["ETF_Rotation"], 0.07)
        w["Megacap_Momentum"] -= cut_mega
        w["ETF_Rotation"] -= cut_rot
        w["Cash"] += (cut_mega + cut_rot) * 0.5
        w["Crash_Hedge"] += (cut_mega + cut_rot) * 0.3
        w["Wheel_CSP"] += (cut_mega + cut_rot) * 0.2
        reasons.append(f"Downtrend: cut momentum/rotation, raised cash + crash hedge")

    # ── 6. Low-vol bull: max momentum ──
    if regime["composite"] == "LOW_VOL_BULL":
        boost = 0.03
        w["Megacap_Momentum"] += boost
        w["ETF_Rotation"] += 0.02
        w["Cash"] = max(0, w["Cash"] - boost - 0.02)
        reasons.append("Low-vol bull: max tilt to momentum and rotation")

    # ── 7. Narrow breadth warning ──
    if regime["breadth"] == "narrow":
        cut = min(w["Megacap_Momentum"], 0.05)
        w["Megacap_Momentum"] -= cut
        w["Cash"] += cut
        reasons.append(f"Narrow breadth ({regime['breadth_pct']:.0f}% above 200SMA): reduced megacap concentration")

    # ── 8. Kill-switch overrides (LAST — override everything) ──
    if kill_sw.get("pause_all"):
        # Halt everything except defensive wheel
        for s in STRATEGIES:
            if s not in ("Wheel_CSP", "Cash"):
                w["Cash"] += w[s]
                w[s] = 0.0
        w["Wheel_CSP"] = min(w["Wheel_CSP"], 0.10)
        w["Cash"] = 1.0 - w["Wheel_CSP"]
        reasons.append("KILL SWITCH: pause_all active — moved to cash + defensive wheel only")

    if kill_sw.get("pause_premium"):
        for s in ["Strangle", "Iron_Condor", "BPS", "Wheel_CSP"]:
            w["Cash"] += w[s]
            w[s] = 0.0
        reasons.append("KILL SWITCH: VIX<12, premiums too thin — paused all premium selling")

    if kill_sw.get("pause_rotation"):
        w["Cash"] += w["ETF_Rotation"]
        w["ETF_Rotation"] = 0.0
        reasons.append("KILL SWITCH: trend breakdown — paused ETF rotation")

    if kill_sw.get("pause_strangles"):
        w["Cash"] += w["Strangle"] + w["Iron_Condor"]
        w["Strangle"] = 0.0
        w["Iron_Condor"] = 0.0
        reasons.append("KILL SWITCH: VIX term structure inverted — paused strangles/condors")

    if kill_sw.get("reduce_50pct"):
        for s in STRATEGIES:
            if s != "Cash":
                half = w[s] * 0.5
                w[s] -= half
                w["Cash"] += half
        reasons.append("KILL SWITCH: 50% position reduction active")

    # ── Normalize to 100% ──
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    # Round to 1 decimal as percentages
    w = {k: round(v, 4) for k, v in w.items()}

    if not reasons:
        reasons.append("No regime adjustments — running baseline weights")

    return w, reasons


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------

def compute_drift(current: dict, previous: dict | None) -> list[str]:
    """Compare current weights to yesterday's and report changes."""
    if not previous:
        return ["First run — no previous allocation to compare"]

    prev_weights = previous.get("weights", {})
    drifts = []
    for strat in STRATEGIES:
        curr = current.get(strat, 0) * 100
        prev = prev_weights.get(strat, 0) * 100
        delta = curr - prev
        if abs(delta) >= 0.5:
            direction = "up" if delta > 0 else "down"
            drifts.append(f"{strat}: {prev:.1f}% -> {curr:.1f}% ({direction} {abs(delta):.1f}pp)")

    if not drifts:
        drifts.append("No meaningful changes vs yesterday")
    return drifts


# ---------------------------------------------------------------------------
# Plain English summary
# ---------------------------------------------------------------------------

def build_summary(regime: dict, weights: dict, drift: list[str],
                  adjustments: list[str], kill_sw: dict) -> str:
    """Generate a clean, plain-English summary."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"{'='*60}")
    lines.append(f"  DAILY PORTFOLIO ALLOCATION — {now}")
    lines.append(f"{'='*60}")

    # Regime
    lines.append("")
    lines.append("MARKET REGIME")
    vix_str = f"{regime['vix']:.1f}" if regime.get("vix") else "N/A"
    lines.append(f"  Composite:   {regime['composite']}")
    lines.append(f"  VIX:         {vix_str}")
    lines.append(f"  Risk:        {regime['risk']}")
    lines.append(f"  Volatility:  {regime['volatility']}")
    lines.append(f"  Trend:       {regime['trend']}")
    lines.append(f"  Inflation:   {regime['inflation']}")
    lines.append(f"  Credit:      {regime['credit']}")
    lines.append(f"  Breadth:     {regime['breadth']} ({regime.get('breadth_pct', 0):.0f}% above 200SMA)")

    # Kill switches
    ks_active = [r for r in kill_sw.get("reasons", []) if "ALL CLEAR" not in r]
    if ks_active:
        lines.append("")
        lines.append("KILL SWITCHES ACTIVE")
        for r in ks_active:
            lines.append(f"  * {r}")
    else:
        lines.append("")
        lines.append("KILL SWITCHES: All clear")

    # Target weights
    lines.append("")
    lines.append("TARGET ALLOCATION")
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    for strat, pct in sorted_w:
        if pct >= 0.005:
            bar = "#" * int(pct * 40)
            lines.append(f"  {strat:<22s} {pct*100:5.1f}%  {bar}")

    # Adjustments
    lines.append("")
    lines.append("ADJUSTMENTS APPLIED")
    for adj in adjustments:
        lines.append(f"  - {adj}")

    # Drift
    lines.append("")
    lines.append("DRIFT VS YESTERDAY")
    for d in drift:
        lines.append(f"  {d}")

    lines.append("")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_allocation(allocation: dict):
    """Save daily JSON and append to CSV history."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DAILY_FILE, "w") as f:
        json.dump(allocation, f, indent=2, default=str)

    # Append to CSV history
    csv_exists = HISTORY_FILE.exists()
    row = {
        "date": allocation["date"],
        "composite_regime": allocation["regime"]["composite"],
        "vix": allocation["regime"].get("vix", ""),
    }
    for strat in STRATEGIES:
        row[strat] = round(allocation["weights"].get(strat, 0) * 100, 1)
    row["kill_switch_active"] = allocation.get("kill_switch_active", False)
    row["n_adjustments"] = len(allocation.get("adjustments", []))

    fieldnames = list(row.keys())
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(verbose: bool = False) -> dict:
    """Execute the full allocation pipeline. Returns the allocation dict."""

    # 1. Load all signal sources
    if verbose:
        print("Loading signal sources...")
    kill_sw = load_kill_switches()
    cross_asset = load_cross_asset()
    regime_learner = load_regime_learner()
    portfolio_sim = load_portfolio_sim()
    sector = load_sector_monitor()
    previous = load_previous()

    if verbose:
        sources_ok = sum(1 for s in [cross_asset, regime_learner, portfolio_sim, sector]
                         if s)
        print(f"  Loaded {sources_ok}/4 data sources + kill-switch monitor")

    # 2. Determine current regime
    regime = determine_regime(cross_asset, sector, kill_sw)

    # 3. Compute weights with adjustments
    weights, adjustments = compute_weights(regime, kill_sw, sector)

    # 4. Drift report
    drift = compute_drift(weights, previous)

    # 5. Build allocation record
    allocation = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "weights": weights,
        "adjustments": adjustments,
        "drift": drift,
        "kill_switch_active": any([
            kill_sw.get("pause_all"),
            kill_sw.get("pause_premium"),
            kill_sw.get("pause_rotation"),
            kill_sw.get("pause_strangles"),
            kill_sw.get("reduce_50pct"),
        ]),
        "kill_switch_reasons": kill_sw.get("reasons", []),
        "baseline_weights": BASELINE,
        "signal_sources": {
            "cross_asset": bool(cross_asset),
            "regime_learner": bool(regime_learner),
            "portfolio_sim": bool(portfolio_sim),
            "sector_monitor": bool(sector),
            "kill_switch": not kill_sw.get("error", False),
        },
    }

    # 6. Save
    save_allocation(allocation)

    # 7. Summary
    summary = build_summary(regime, weights, drift, adjustments, kill_sw)
    allocation["summary"] = summary

    return allocation


def main():
    parser = argparse.ArgumentParser(description="Unified Daily Portfolio Allocator v2")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    allocation = run(verbose=args.verbose or not args.json)

    if args.json:
        out = {k: v for k, v in allocation.items() if k != "summary"}
        print(json.dumps(out, indent=2, default=str))
    else:
        print(allocation["summary"])


if __name__ == "__main__":
    main()
