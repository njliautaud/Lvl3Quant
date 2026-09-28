#!/usr/bin/env python3
"""
Portfolio Allocator & Paper Trading Engine
==========================================
Daily monitoring tool for the 3-sleeve portfolio:
  - Core 70-80%: Income strategies (CSP + IC Condors + ETF Rotation)
  - Satellite 15-25%: Leveraged dual momentum (TQQQ/QQQ/SPY)
  - Crisis alpha 5%: VIX mean-reversion (2-6 trades/year)

Walk-forward validated parameters (28 folds, 3400 OOT days):
  - Dual Momentum: lookback=10d, vol_threshold=30%, Sharpe=3.94, Sortino=5.57
  - VIX crisis alpha: buy SPY when VIX > 30 then drops 20%+

Run: python3 portfolio_allocator.py [--account-size 100000] [--json] [--cron]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
STATE_FILE = OUTPUT_DIR / "portfolio_state.json"
WF_RESULTS = OUTPUT_DIR / "walkforward_validation" / "walkforward_results.json"

# Walk-forward validated parameters
DM_LOOKBACK = 10          # days
DM_VOL_THRESHOLD = 0.30   # 30% annualized return threshold for TQQQ
VIX_CRISIS_ENTRY = 30     # buy SPY when VIX crosses above this
VIX_CRISIS_SPIKE = 0.20   # 20% spike-then-drop pattern

# VIX-threshold leverage timing (from drawdown predictor HC #325)
VIX_REDUCE_LEVERAGE = 25  # reduce leverage when VIX > 25
VIX_CASH_GROWTH = 35      # go cash on growth sleeve when VIX > 35

# Target allocations (percentage of total portfolio)
ALLOC_CORE_DEFAULT = 0.75       # 70-80% income strategies
ALLOC_SATELLITE_DEFAULT = 0.20  # 15-25% leveraged momentum
ALLOC_CRISIS_DEFAULT = 0.05     # 5% crisis alpha reserve

# Income strategy monthly yield assumptions (from backtest)
MONTHLY_YIELD_CSP = 0.025       # ~2.5% monthly from cash-secured puts
MONTHLY_YIELD_IC = 0.015        # ~1.5% monthly from iron condors
MONTHLY_YIELD_ETF_ROT = 0.008  # ~0.8% monthly from ETF rotation
CORE_MONTHLY_YIELD = 0.018     # blended ~1.8% monthly


# ── Data Download ──────────────────────────────────────────────────────────────

def download_prices(tickers: list[str], period: str = "3mo") -> dict:
    """Download price data for tickers. Returns {ticker: DataFrame}."""
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, progress=False, auto_adjust=True)
            if df is not None and len(df) > 0:
                # Flatten multi-level columns if present
                if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
        except Exception as e:
            print(f"  WARNING: Failed to download {t}: {e}")
    return data


# ── Dual Momentum Signal Generator ────────────────────────────────────────────

def compute_dual_momentum(prices: dict) -> dict:
    """
    Dual Momentum signal using walk-forward validated parameters.
    lookback=10d, vol_threshold=30% annualized.

    Logic:
      1. If TQQQ 10d return > vol_threshold (annualized) -> hold TQQQ
      2. Elif QQQ 10d return > 0 -> hold QQQ
      3. Else -> hold SPY (safety)

    VIX overlay:
      - VIX > 25: reduce to 0.75x position
      - VIX > 35: go to cash (no growth allocation)
    """
    result = {
        "signal": "SPY",
        "confidence": 0.0,
        "returns_10d": {},
        "rationale": "",
        "vix_override": None,
    }

    for ticker in ["TQQQ", "QQQ", "SPY"]:
        if ticker not in prices or len(prices[ticker]) < DM_LOOKBACK + 1:
            result["rationale"] = f"Insufficient data for {ticker}"
            return result

    # Compute 10-day returns
    for ticker in ["TQQQ", "QQQ", "SPY"]:
        close = prices[ticker]["Close"].values
        ret_10d = (close[-1] / close[-(DM_LOOKBACK + 1)] - 1.0)
        result["returns_10d"][ticker] = round(float(ret_10d) * 100, 2)

    tqqq_ret = result["returns_10d"]["TQQQ"] / 100.0
    qqq_ret = result["returns_10d"]["QQQ"] / 100.0

    # Annualize the 10-day return for threshold comparison
    tqqq_ann = tqqq_ret * (252 / DM_LOOKBACK)

    if tqqq_ann > DM_VOL_THRESHOLD:
        result["signal"] = "TQQQ"
        # Confidence based on how far above threshold
        excess = tqqq_ann - DM_VOL_THRESHOLD
        result["confidence"] = min(1.0, 0.5 + excess / DM_VOL_THRESHOLD)
        result["rationale"] = (
            f"TQQQ 10d return {result['returns_10d']['TQQQ']:.1f}% "
            f"(annualized {tqqq_ann*100:.0f}%) > {DM_VOL_THRESHOLD*100:.0f}% threshold"
        )
    elif qqq_ret > 0:
        result["signal"] = "QQQ"
        result["confidence"] = min(1.0, 0.3 + qqq_ret * 5)
        result["rationale"] = (
            f"QQQ 10d return {result['returns_10d']['QQQ']:.1f}% > 0, "
            f"but TQQQ annualized {tqqq_ann*100:.0f}% below threshold"
        )
    else:
        result["signal"] = "SPY"
        result["confidence"] = 0.8  # high confidence in safety
        result["rationale"] = (
            f"QQQ 10d return {result['returns_10d']['QQQ']:.1f}% < 0 -> safety mode"
        )

    return result


# ── VIX Crisis Alpha Monitor ──────────────────────────────────────────────────

def compute_vix_crisis(prices: dict) -> dict:
    """
    VIX crisis alpha monitor.
    Signal strength 0-3:
      0 = no crisis, VIX normal
      1 = elevated (VIX 20-30), watch mode
      2 = crisis entry zone (VIX > 30)
      3 = spike-then-drop pattern detected (buy SPY aggressively)
    """
    result = {
        "signal_strength": 0,
        "vix_current": None,
        "vix_5d_ago": None,
        "vix_20d_high": None,
        "vix_spike_pct": None,
        "action": "NONE",
        "rationale": "",
    }

    if "^VIX" not in prices or len(prices["^VIX"]) < 21:
        result["rationale"] = "Insufficient VIX data"
        return result

    vix = prices["^VIX"]["Close"].values
    result["vix_current"] = round(float(vix[-1]), 2)
    result["vix_5d_ago"] = round(float(vix[-6]) if len(vix) >= 6 else vix[0], 2)
    result["vix_20d_high"] = round(float(np.max(vix[-21:])), 2)

    current = result["vix_current"]

    # Level 1: Elevated
    if 20 <= current < 30:
        result["signal_strength"] = 1
        result["action"] = "WATCH"
        result["rationale"] = f"VIX elevated at {current:.1f} — monitor for spike"

    # Level 2: Crisis entry zone
    elif current >= 30:
        result["signal_strength"] = 2
        result["action"] = "PREPARE"
        result["rationale"] = f"VIX at {current:.1f} > 30 — crisis alpha zone, prepare SPY entry"

    # Check for spike-then-drop (Level 3)
    if result["vix_20d_high"] > 30 and len(vix) >= 6:
        spike_pct = (result["vix_20d_high"] - current) / result["vix_20d_high"]
        result["vix_spike_pct"] = round(float(spike_pct) * 100, 1)

        if spike_pct >= VIX_CRISIS_SPIKE and current < result["vix_20d_high"]:
            result["signal_strength"] = 3
            result["action"] = "BUY_SPY"
            result["rationale"] = (
                f"VIX spike-then-drop: peaked at {result['vix_20d_high']:.1f}, "
                f"now {current:.1f} ({result['vix_spike_pct']:.0f}% drop) — buy SPY"
            )

    # Normal
    if current < 20:
        result["signal_strength"] = 0
        result["action"] = "NONE"
        result["rationale"] = f"VIX at {current:.1f} — normal, no crisis alpha opportunity"

    return result


# ── Risk Metrics ───────────────────────────────────────────────────────────────

def compute_risk_metrics(prices: dict) -> dict:
    """Compute current risk environment metrics."""
    metrics = {
        "vix_level": None,
        "spy_drawdown_pct": None,
        "spy_trend": None,
        "spy_above_50ma": None,
        "spy_above_200ma": None,
        "leverage_recommendation": "1.0x",
    }

    # VIX
    if "^VIX" in prices and len(prices["^VIX"]) > 0:
        metrics["vix_level"] = round(float(prices["^VIX"]["Close"].values[-1]), 2)

    # SPY metrics
    if "SPY" in prices and len(prices["SPY"]) >= 50:
        close = prices["SPY"]["Close"].values
        high_252 = np.max(close)
        dd = (close[-1] / high_252 - 1.0) * 100
        metrics["spy_drawdown_pct"] = round(float(dd), 2)

        ma50 = np.mean(close[-50:])
        metrics["spy_above_50ma"] = bool(close[-1] > ma50)

        if len(close) >= 60:
            ma_long = np.mean(close[-60:])
            metrics["spy_above_200ma"] = bool(close[-1] > ma_long)
        else:
            metrics["spy_above_200ma"] = True  # assume OK with limited data

        # Trend determination
        if close[-1] > ma50 and (metrics["spy_above_200ma"]):
            metrics["spy_trend"] = "UPTREND"
        elif close[-1] < ma50:
            metrics["spy_trend"] = "DOWNTREND"
        else:
            metrics["spy_trend"] = "NEUTRAL"

    # Leverage recommendation based on VIX thresholds
    vix = metrics["vix_level"]
    if vix is not None:
        if vix > VIX_CASH_GROWTH:
            metrics["leverage_recommendation"] = "0x (CASH on growth)"
        elif vix > VIX_REDUCE_LEVERAGE:
            metrics["leverage_recommendation"] = "0.75x (reduced)"
        elif vix < 15:
            metrics["leverage_recommendation"] = "1.25x (low vol, can add)"
        else:
            metrics["leverage_recommendation"] = "1.0x (normal)"

    return metrics


# ── Leverage Calculator ────────────────────────────────────────────────────────

def compute_leverage_table(account_size: float, risk_metrics: dict) -> dict:
    """
    Compute position sizes and projections at various leverage levels.
    """
    vix = risk_metrics.get("vix_level") or 15
    levels = [1.0, 1.25, 1.5, 2.0]

    table = []
    for lev in levels:
        notional = account_size * lev

        # Allocations
        core_alloc = notional * ALLOC_CORE_DEFAULT
        satellite_alloc = notional * ALLOC_SATELLITE_DEFAULT
        crisis_alloc = notional * ALLOC_CRISIS_DEFAULT

        # Monthly income projection (core income strategies)
        monthly_income = core_alloc * CORE_MONTHLY_YIELD

        # Annual projection
        annual_income = monthly_income * 12

        # Max acceptable drawdown (tighter at higher leverage)
        # Rule: max DD = 15% / leverage
        max_dd_pct = 15.0 / lev

        # Risk-adjusted: if VIX > 25, flag higher leverage as dangerous
        risk_flag = ""
        if vix > VIX_CASH_GROWTH and lev > 1.0:
            risk_flag = "BLOCKED (VIX > 35)"
        elif vix > VIX_REDUCE_LEVERAGE and lev > 1.25:
            risk_flag = "WARNING (VIX > 25)"

        table.append({
            "leverage": f"{lev}x",
            "notional": round(notional),
            "core_income": round(core_alloc),
            "satellite_momentum": round(satellite_alloc),
            "crisis_reserve": round(crisis_alloc),
            "monthly_income_est": round(monthly_income),
            "annual_income_est": round(annual_income),
            "max_drawdown_pct": round(max_dd_pct, 1),
            "max_drawdown_dollars": round(account_size * max_dd_pct / 100),
            "risk_flag": risk_flag,
        })

    return {"account_size": account_size, "levels": table}


# ── Portfolio Allocation Decision ──────────────────────────────────────────────

def compute_allocation(dm_signal: dict, vix_crisis: dict, risk_metrics: dict) -> dict:
    """
    Decide current portfolio allocation based on all signals.
    Adjusts from defaults based on market conditions.
    """
    vix = risk_metrics.get("vix_level") or 15
    trend = risk_metrics.get("spy_trend", "NEUTRAL")

    core_pct = ALLOC_CORE_DEFAULT
    satellite_pct = ALLOC_SATELLITE_DEFAULT
    crisis_pct = ALLOC_CRISIS_DEFAULT

    adjustments = []

    # VIX > 35: zero out growth, move to core income
    if vix > VIX_CASH_GROWTH:
        satellite_pct = 0.0
        core_pct = 0.90
        crisis_pct = 0.10
        adjustments.append(f"VIX {vix:.0f} > 35: growth sleeve zeroed, shifted to income + crisis reserve")

    # VIX > 25: reduce satellite
    elif vix > VIX_REDUCE_LEVERAGE:
        satellite_pct = 0.10
        core_pct = 0.82
        crisis_pct = 0.08
        adjustments.append(f"VIX {vix:.0f} > 25: satellite reduced to 10%, core increased")

    # VIX crisis opportunity: increase crisis allocation
    if vix_crisis["signal_strength"] >= 2:
        crisis_pct = min(0.10, crisis_pct + 0.05)
        core_pct = max(0.65, core_pct - 0.05)
        adjustments.append(f"Crisis alpha signal {vix_crisis['signal_strength']}/3: crisis allocation increased")

    # Downtrend: reduce satellite further
    if trend == "DOWNTREND" and satellite_pct > 0.05:
        shift = satellite_pct * 0.5
        satellite_pct -= shift
        core_pct += shift
        adjustments.append("Downtrend detected: halved satellite allocation")

    # Normalize
    total = core_pct + satellite_pct + crisis_pct
    core_pct /= total
    satellite_pct /= total
    crisis_pct /= total

    # Satellite holding
    satellite_holding = dm_signal["signal"] if satellite_pct > 0.01 else "CASH"

    return {
        "core_pct": round(core_pct * 100, 1),
        "satellite_pct": round(satellite_pct * 100, 1),
        "crisis_pct": round(crisis_pct * 100, 1),
        "satellite_holding": satellite_holding,
        "satellite_confidence": round(dm_signal["confidence"] * 100, 1),
        "crisis_action": vix_crisis["action"],
        "adjustments": adjustments if adjustments else ["Standard allocation, no adjustments"],
        "rebalance_needed": len(adjustments) > 0,
    }


# ── State Management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load previous state for comparison."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def save_state(state: dict):
    """Save current state."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Display ────────────────────────────────────────────────────────────────────

def format_dashboard(
    dm_signal: dict,
    vix_crisis: dict,
    risk_metrics: dict,
    allocation: dict,
    leverage_table: dict,
    prev_state: dict,
) -> str:
    """Format clean dashboard output."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("=" * 60)
    lines.append(f"  PORTFOLIO ALLOCATOR — {now}")
    lines.append("=" * 60)

    # ── Risk Environment ──
    lines.append("")
    lines.append("--- RISK ENVIRONMENT ---")
    vix = risk_metrics.get("vix_level", "N/A")
    dd = risk_metrics.get("spy_drawdown_pct", "N/A")
    trend = risk_metrics.get("spy_trend", "N/A")
    lev_rec = risk_metrics.get("leverage_recommendation", "N/A")
    lines.append(f"  VIX:              {vix}")
    lines.append(f"  SPY Drawdown:     {dd}% from 3mo high")
    lines.append(f"  SPY Trend:        {trend}")
    lines.append(f"  50MA Status:      {'Above' if risk_metrics.get('spy_above_50ma') else 'Below'}")
    lines.append(f"  Leverage Rec:     {lev_rec}")

    # ── Dual Momentum Signal ──
    lines.append("")
    lines.append("--- DUAL MOMENTUM (Satellite Sleeve) ---")
    lines.append(f"  Signal:           {dm_signal['signal']}")
    lines.append(f"  Confidence:       {dm_signal['confidence']*100:.0f}%")
    lines.append(f"  Rationale:        {dm_signal['rationale']}")
    lines.append(f"  10d Returns:")
    for t, r in dm_signal.get("returns_10d", {}).items():
        lines.append(f"    {t:6s}  {r:+.2f}%")

    # Signal change detection
    prev_signal = prev_state.get("dual_momentum", {}).get("signal")
    if prev_signal and prev_signal != dm_signal["signal"]:
        lines.append(f"  ** SIGNAL CHANGE: {prev_signal} -> {dm_signal['signal']} **")

    # ── VIX Crisis Alpha ──
    lines.append("")
    lines.append("--- VIX CRISIS ALPHA ---")
    strength = vix_crisis["signal_strength"]
    strength_bar = ["____", "=___", "==__", "===="][strength]
    lines.append(f"  Signal Strength:  {strength}/3  [{strength_bar}]")
    lines.append(f"  Action:           {vix_crisis['action']}")
    lines.append(f"  Rationale:        {vix_crisis['rationale']}")
    if vix_crisis.get("vix_20d_high"):
        lines.append(f"  VIX 20d High:     {vix_crisis['vix_20d_high']}")
    if vix_crisis.get("vix_spike_pct"):
        lines.append(f"  VIX Drop from Peak: {vix_crisis['vix_spike_pct']}%")

    # ── Current Allocation ──
    lines.append("")
    lines.append("--- RECOMMENDED ALLOCATION ---")
    lines.append(f"  Core (Income):    {allocation['core_pct']}%")
    lines.append(f"  Satellite:        {allocation['satellite_pct']}% -> {allocation['satellite_holding']}")
    lines.append(f"  Crisis Reserve:   {allocation['crisis_pct']}%")
    if allocation["adjustments"]:
        lines.append(f"  Adjustments:")
        for adj in allocation["adjustments"]:
            lines.append(f"    - {adj}")
    if allocation["rebalance_needed"]:
        lines.append(f"  ** REBALANCE NEEDED **")

    # ── Leverage Table ──
    lines.append("")
    lines.append(f"--- LEVERAGE CALCULATOR (Account: ${leverage_table['account_size']:,.0f}) ---")
    lines.append(f"  {'Lev':>5s}  {'Notional':>10s}  {'Core':>10s}  {'Satell':>10s}  {'Mo.Inc':>8s}  {'MaxDD%':>6s}  {'Flag'}")
    lines.append(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*20}")
    for row in leverage_table["levels"]:
        flag = row["risk_flag"] or "OK"
        lines.append(
            f"  {row['leverage']:>5s}  "
            f"${row['notional']:>9,}  "
            f"${row['core_income']:>9,}  "
            f"${row['satellite_momentum']:>9,}  "
            f"${row['monthly_income_est']:>7,}  "
            f"{row['max_drawdown_pct']:>5.1f}%  "
            f"{flag}"
        )

    # ── Next Action ──
    lines.append("")
    lines.append("--- NEXT ACTION ---")
    actions = []
    if allocation["rebalance_needed"]:
        actions.append(f"Rebalance satellite to {allocation['satellite_holding']} at {allocation['satellite_pct']}%")
    if vix_crisis["action"] == "BUY_SPY":
        actions.append("Crisis alpha: execute SPY buy (VIX spike-drop pattern)")
    elif vix_crisis["action"] == "PREPARE":
        actions.append("Crisis alpha: prepare SPY limit orders near support")
    if prev_signal and prev_signal != dm_signal["signal"]:
        actions.append(f"Momentum rotation: switch from {prev_signal} to {dm_signal['signal']}")
    if not actions:
        actions.append("Hold current positions. Next check tomorrow.")
    for a in actions:
        lines.append(f"  -> {a}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_discord_summary(
    dm_signal: dict,
    vix_crisis: dict,
    risk_metrics: dict,
    allocation: dict,
    prev_state: dict,
) -> str:
    """Short Discord-friendly summary (no file paths, no jargon)."""
    vix = risk_metrics.get("vix_level", "?")
    trend = risk_metrics.get("spy_trend", "?")
    lev = risk_metrics.get("leverage_recommendation", "?")

    lines = [
        f"**Portfolio Check** — {datetime.now().strftime('%b %d %H:%M')}",
        f"VIX {vix} | SPY {trend} | Leverage rec: {lev}",
        f"Momentum signal: **{dm_signal['signal']}** ({dm_signal['confidence']*100:.0f}% conf)",
        f"Allocation: Core {allocation['core_pct']}% / Satellite {allocation['satellite_pct']}% / Crisis {allocation['crisis_pct']}%",
    ]

    if vix_crisis["signal_strength"] >= 2:
        lines.append(f"Crisis alpha: {vix_crisis['action']} (strength {vix_crisis['signal_strength']}/3)")

    prev_signal = prev_state.get("dual_momentum", {}).get("signal")
    if prev_signal and prev_signal != dm_signal["signal"]:
        lines.append(f"ROTATION: {prev_signal} -> {dm_signal['signal']}")

    if allocation["adjustments"] and allocation["rebalance_needed"]:
        lines.append("Rebalance needed: " + "; ".join(allocation["adjustments"]))

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio Allocator & Signal Monitor")
    parser.add_argument("--account-size", type=float, default=100000,
                        help="Account size in USD (default: 100000)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of dashboard")
    parser.add_argument("--cron", action="store_true",
                        help="Cron mode: minimal output, save state only")
    parser.add_argument("--discord", action="store_true",
                        help="Output Discord-friendly summary")
    args = parser.parse_args()

    # Load previous state
    prev_state = load_state()

    # Download market data
    tickers = ["TQQQ", "QQQ", "SPY", "^VIX"]
    if not args.cron:
        print("Downloading market data...")
    prices = download_prices(tickers, period="3mo")

    if not prices:
        print("ERROR: Could not download any market data")
        sys.exit(1)

    # Compute signals
    dm_signal = compute_dual_momentum(prices)
    vix_crisis = compute_vix_crisis(prices)
    risk_metrics = compute_risk_metrics(prices)
    allocation = compute_allocation(dm_signal, vix_crisis, risk_metrics)
    leverage_table = compute_leverage_table(args.account_size, risk_metrics)

    # Build state
    state = {
        "timestamp": datetime.now().isoformat(),
        "dual_momentum": dm_signal,
        "vix_crisis": vix_crisis,
        "risk_metrics": risk_metrics,
        "allocation": allocation,
        "leverage_table": leverage_table,
        "account_size": args.account_size,
    }

    # Save state
    save_state(state)

    # Output
    if args.json:
        print(json.dumps(state, indent=2, default=str))
    elif args.discord:
        print(format_discord_summary(dm_signal, vix_crisis, risk_metrics, allocation, prev_state))
    elif args.cron:
        # In cron mode, only print if there's a signal change or rebalance needed
        prev_signal = prev_state.get("dual_momentum", {}).get("signal")
        if (prev_signal and prev_signal != dm_signal["signal"]) or allocation["rebalance_needed"]:
            print(format_discord_summary(dm_signal, vix_crisis, risk_metrics, allocation, prev_state))
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] No changes. "
                  f"{dm_signal['signal']} | VIX {risk_metrics.get('vix_level', '?')} | "
                  f"Crisis {vix_crisis['signal_strength']}/3")
    else:
        print(format_dashboard(dm_signal, vix_crisis, risk_metrics, allocation, leverage_table, prev_state))


if __name__ == "__main__":
    main()
