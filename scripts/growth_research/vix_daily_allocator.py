#!/usr/bin/env python3
"""
VIX Daily Allocator — Recommendation Engine
============================================

Checks current VIX level, determines target allocation for the
VIX-threshold leveraged ETF strategy, compares to current portfolio,
and generates trade recommendations.

Strategy rules (optimized: 1,152 configs, adversarial-validated):
  - VIX < 17:  100% UPRO (3x S&P 500)
  - VIX 17-25: 30% UPRO + 70% cash
  - VIX > 25:  100% cash
UPRO version: CAGR 87.7%, Sharpe 3.69, MaxDD -20.0%

CRITICAL: 1-day VIX lag drops Sharpe by 71%. This script MUST run
during market hours — ideally 3:30-3:45 PM ET — to act on same-day VIX.

Usage:
  python vix_daily_allocator.py              # dry-run (default)
  python vix_daily_allocator.py --execute    # live execution (future)
  python vix_daily_allocator.py --portfolio  # show portfolio only
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

# ─── Configuration ──────────────────────────────────────────────────────

# VIX thresholds (optimized: 1,152 configs tested, adversarial-validated)
# Best risk-adjusted: scale-down at 17, cash at 25, 30% in middle zone
# UPRO: CAGR 87.7%, Sharpe 3.69, MaxDD -20.0%
# TQQQ: CAGR 119.7%, Sharpe 3.85, MaxDD -23.1%
VIX_SCALEDOWN_THRESHOLD = 17.0   # Above this: reduce to 30% UPRO
VIX_CASH_THRESHOLD = 25.0        # Above this: 100% cash
SCALEDOWN_ALLOC = 0.30           # 30% allocation in VIX 17-25 zone

# Tickers
UPRO_TICKER = "UPRO"
VIX_TICKER = "^VIX"
CASH_PROXY = "SHV"  # iShares Short Treasury Bond ETF (cash-like)

# Account
ACCOUNT_VALUE_APPROX = 440.0  # Will be overridden by actual portfolio read

# Output paths
LOG_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
LOG_FILE = LOG_DIR / "vix_allocator_log.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─── VIX Data ───────────────────────────────────────────────────────────

def get_current_vix():
    """Fetch current VIX level using yfinance."""
    try:
        vix = yf.Ticker(VIX_TICKER)
        # Use fast_info for current price, fall back to history
        try:
            price = vix.fast_info.get("lastPrice", None)
            if price and price > 0:
                return float(price)
        except Exception:
            pass

        # Fallback: last close from history
        hist = vix.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

        # Second fallback: 5d history
        hist = vix.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

        return None
    except Exception as e:
        print(f"ERROR fetching VIX: {e}")
        return None


def get_upro_price():
    """Fetch current UPRO price."""
    try:
        upro = yf.Ticker(UPRO_TICKER)
        try:
            price = upro.fast_info.get("lastPrice", None)
            if price and price > 0:
                return float(price)
        except Exception:
            pass

        hist = upro.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None
    except Exception as e:
        print(f"ERROR fetching UPRO price: {e}")
        return None


# ─── Drawdown Protection (HC #709 + #710 validated rules) ─────────────

def check_drawdown_protection():
    """
    Check the 4 validated drawdown protection signals.

    HC #709/710 validated composite ("aggressive_all"):
      1. VIX < 20
      2. SPY above 50-day SMA
      3. Credit not stressed (HYG-LQD spread z-score > -1.0)
      4. Sector breadth > 50% above 50-day SMA

    Backtest: Sharpe 0.59→2.91, MaxDD -77%→-8.5%, R1 pass, perm p=0.000.

    Returns dict with each signal status and overall recommendation.
    """
    import pandas as pd

    signals = {}
    details = []

    try:
        # 1. VIX check (already have from main flow, but re-fetch for completeness)
        vix_data = yf.download("^VIX", period="5d", progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
        current_vix = float(vix_data['Close'].iloc[-1]) if len(vix_data) > 0 else None
        if current_vix is not None:
            signals['vix_below_20'] = current_vix < 20.0
            details.append(f"VIX {current_vix:.1f} {'< 20 ✅' if current_vix < 20 else '>= 20 ⚠️'}")

        # 2. SPY above 50-day SMA
        spy_data = yf.download("SPY", period="100d", progress=False)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)
        if len(spy_data) >= 50:
            spy_close = float(spy_data['Close'].iloc[-1])
            spy_50sma = float(spy_data['Close'].rolling(50).mean().iloc[-1])
            above_50 = spy_close > spy_50sma
            signals['spy_above_50sma'] = above_50
            details.append(f"SPY ${spy_close:.0f} {'>' if above_50 else '<'} 50SMA ${spy_50sma:.0f} {'✅' if above_50 else '⚠️'}")

        # 3. Credit spread (HYG vs LQD)
        hyg = yf.download("HYG", period="100d", progress=False)
        lqd = yf.download("LQD", period="100d", progress=False)
        if isinstance(hyg.columns, pd.MultiIndex):
            hyg.columns = hyg.columns.get_level_values(0)
        if isinstance(lqd.columns, pd.MultiIndex):
            lqd.columns = lqd.columns.get_level_values(0)
        if len(hyg) >= 60 and len(lqd) >= 60:
            hyg_ret = hyg['Close'].pct_change()
            lqd_ret = lqd['Close'].pct_change()
            spread = (hyg_ret - lqd_ret).rolling(20).mean()
            spread_z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
            z_val = float(spread_z.iloc[-1])
            credit_ok = z_val > -1.0
            signals['credit_not_stressed'] = credit_ok
            details.append(f"Credit z-score {z_val:.2f} {'> -1.0 ✅' if credit_ok else '<= -1.0 ⚠️'}")

        # 4. Sector breadth (% above 50-day SMA)
        sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLY', 'XLC']
        above_count = 0
        total_count = 0
        for s in sectors:
            try:
                sd = yf.download(s, period="100d", progress=False)
                if isinstance(sd.columns, pd.MultiIndex):
                    sd.columns = sd.columns.get_level_values(0)
                if len(sd) >= 50:
                    total_count += 1
                    if float(sd['Close'].iloc[-1]) > float(sd['Close'].rolling(50).mean().iloc[-1]):
                        above_count += 1
            except Exception:
                pass

        if total_count > 0:
            breadth = above_count / total_count
            breadth_ok = breadth > 0.5
            signals['breadth_above_50pct'] = breadth_ok
            details.append(f"Breadth {above_count}/{total_count} ({breadth*100:.0f}%) {'> 50% ✅' if breadth_ok else '<= 50% ⚠️'}")

    except Exception as e:
        print(f"  WARNING: Drawdown protection check error: {e}")
        # Default to safe (don't block on data issues)
        return {
            'all_clear': True,
            'signals': {},
            'n_passing': 0,
            'n_total': 0,
            'details': [f"Error checking signals: {e}"],
            'recommendation': 'DATA_ERROR — defaulting to VIX-only allocation',
        }

    n_passing = sum(1 for v in signals.values() if v)
    n_total = len(signals)
    all_clear = all(signals.values()) if signals else True
    majority_clear = n_passing >= max(1, int(n_total * 0.6))

    if all_clear:
        rec = 'ALL_CLEAR — full exposure OK'
    elif majority_clear:
        rec = 'MAJORITY_CLEAR — standard exposure OK, monitor warnings'
    else:
        rec = 'DANGER — scale down exposure, multiple warning signals'

    return {
        'all_clear': all_clear,
        'majority_clear': majority_clear,
        'signals': {k: bool(v) for k, v in signals.items()},
        'n_passing': n_passing,
        'n_total': n_total,
        'details': details,
        'recommendation': rec,
    }


# ─── Allocation Logic ──────────────────────────────────────────────────

def determine_target_allocation(vix_level, protection=None):
    """
    Determine target allocation based on VIX level + drawdown protection overlay.

    Two-layer system:
      Layer 1 (VIX thresholds): Sets base allocation (100%/30%/0%)
      Layer 2 (Drawdown protection): Can REDUCE allocation if multiple
              cross-asset signals flash warning, even if VIX is low.

    Returns:
        dict with keys 'upro_pct', 'cash_pct', 'regime', 'protection_override'
    """
    # Layer 1: VIX-based allocation
    if vix_level < VIX_SCALEDOWN_THRESHOLD:
        base = {
            "upro_pct": 1.0,
            "cash_pct": 0.0,
            "regime": "LOW_VOL",
            "description": f"VIX {vix_level:.1f} < {VIX_SCALEDOWN_THRESHOLD} => 100% UPRO"
        }
    elif vix_level < VIX_CASH_THRESHOLD:
        base = {
            "upro_pct": SCALEDOWN_ALLOC,
            "cash_pct": 1.0 - SCALEDOWN_ALLOC,
            "regime": "ELEVATED_VOL",
            "description": f"VIX {vix_level:.1f} in [{VIX_SCALEDOWN_THRESHOLD}, {VIX_CASH_THRESHOLD}) => {SCALEDOWN_ALLOC*100:.0f}% UPRO / {(1-SCALEDOWN_ALLOC)*100:.0f}% cash"
        }
    else:
        base = {
            "upro_pct": 0.0,
            "cash_pct": 1.0,
            "regime": "HIGH_VOL",
            "description": f"VIX {vix_level:.1f} >= {VIX_CASH_THRESHOLD} => 100% cash"
        }

    base["protection_override"] = False

    # Layer 2: Drawdown protection overlay
    if protection and not protection.get('all_clear', True):
        if not protection.get('majority_clear', True):
            # Multiple signals flashing danger — scale down by 50%
            original_upro = base["upro_pct"]
            base["upro_pct"] = original_upro * 0.5
            base["cash_pct"] = 1.0 - base["upro_pct"]
            base["protection_override"] = True
            base["description"] += f" | PROTECTION: scaled down 50% ({protection['n_passing']}/{protection['n_total']} signals clear)"
        else:
            # Majority clear but not all — just note it, don't override
            base["description"] += f" | NOTE: {protection['n_passing']}/{protection['n_total']} protection signals clear"

    return base


def compute_trades(target, current_upro_shares, current_upro_value,
                   cash_available, upro_price, total_portfolio_value):
    """
    Compute the trades needed to reach target allocation.

    Returns:
        dict with trade recommendation details
    """
    target_upro_value = total_portfolio_value * target["upro_pct"]
    target_cash_value = total_portfolio_value * target["cash_pct"]

    current_upro_pct = current_upro_value / total_portfolio_value if total_portfolio_value > 0 else 0
    current_cash_pct = cash_available / total_portfolio_value if total_portfolio_value > 0 else 0

    upro_delta_value = target_upro_value - current_upro_value
    upro_delta_shares = upro_delta_value / upro_price if upro_price > 0 else 0

    # Robinhood supports fractional shares for UPRO
    # But minimum order is typically $1
    action = "HOLD"
    if abs(upro_delta_value) < 5.0:
        action = "HOLD"
        trade_description = "No trade needed (within $5 tolerance)"
    elif upro_delta_value > 0:
        action = "BUY"
        trade_description = f"BUY {abs(upro_delta_shares):.4f} shares UPRO (~${abs(upro_delta_value):.2f})"
    else:
        action = "SELL"
        trade_description = f"SELL {abs(upro_delta_shares):.4f} shares UPRO (~${abs(upro_delta_value):.2f})"

    return {
        "action": action,
        "trade_description": trade_description,
        "upro_delta_shares": upro_delta_shares,
        "upro_delta_value": upro_delta_value,
        "target_upro_value": target_upro_value,
        "target_cash_value": target_cash_value,
        "current_upro_pct": current_upro_pct,
        "current_cash_pct": current_cash_pct,
    }


# ─── Portfolio Reading ──────────────────────────────────────────────────

def get_portfolio_from_args():
    """
    Get current portfolio state.

    For now, uses manual/approximate values. When --execute mode is
    enabled, this will use the Robinhood MCP tools to read live positions.

    Returns:
        dict with portfolio details
    """
    # TODO: When ready for live execution, integrate Robinhood MCP:
    #   - mcp__robinhood-trading__get_equity_positions
    #   - mcp__robinhood-trading__get_portfolio
    #   - mcp__robinhood-trading__get_accounts
    #
    # For now, we'll use yfinance price + assume we need to check
    # a local state file for position tracking.

    state_file = LOG_DIR / "vix_allocator_positions.json"

    if state_file.exists():
        import json
        try:
            with open(state_file) as f:
                state = json.load(f)
            return {
                "upro_shares": state.get("upro_shares", 0),
                "cash_available": state.get("cash_available", ACCOUNT_VALUE_APPROX),
                "total_value": state.get("total_value", ACCOUNT_VALUE_APPROX),
                "source": "state_file"
            }
        except Exception:
            pass

    # Default: assume all cash (no positions yet)
    return {
        "upro_shares": 0,
        "cash_available": ACCOUNT_VALUE_APPROX,
        "total_value": ACCOUNT_VALUE_APPROX,
        "source": "default"
    }


def save_portfolio_state(upro_shares, cash_available, total_value):
    """Save current portfolio state for next run."""
    import json
    state_file = LOG_DIR / "vix_allocator_positions.json"
    state = {
        "upro_shares": upro_shares,
        "cash_available": cash_available,
        "total_value": total_value,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# ─── Logging ────────────────────────────────────────────────────────────

def log_decision(date_str, vix, target_alloc, current_alloc, action, executed):
    """Append decision to CSV log."""
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "date", "time_utc", "vix", "regime",
                "target_upro_pct", "target_cash_pct",
                "current_upro_pct", "current_cash_pct",
                "action", "trade_detail", "executed"
            ])
        writer.writerow([
            date_str,
            datetime.now(timezone.utc).strftime("%H:%M:%S"),
            f"{vix:.2f}",
            target_alloc["regime"],
            f"{target_alloc['upro_pct']:.2f}",
            f"{target_alloc['cash_pct']:.2f}",
            f"{current_alloc.get('current_upro_pct', 0):.2f}",
            f"{current_alloc.get('current_cash_pct', 0):.2f}",
            current_alloc.get("action", "UNKNOWN"),
            current_alloc.get("trade_description", ""),
            executed
        ])


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIX Daily Allocator — leveraged ETF strategy recommendation engine"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute trades via Robinhood (NOT YET IMPLEMENTED — dry-run only for now)"
    )
    parser.add_argument(
        "--portfolio", action="store_true",
        help="Show current portfolio state and exit"
    )
    parser.add_argument(
        "--update-portfolio", nargs=2, metavar=("UPRO_SHARES", "CASH"),
        type=float,
        help="Manually update portfolio state: --update-portfolio <upro_shares> <cash_available>"
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    dry_run = not args.execute

    print("=" * 60)
    print("  VIX DAILY ALLOCATOR — Leveraged ETF Strategy")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE EXECUTION'}")
    print("=" * 60)

    # Handle portfolio update
    if args.update_portfolio:
        upro_shares, cash = args.update_portfolio
        upro_price = get_upro_price()
        if upro_price:
            total = upro_shares * upro_price + cash
            save_portfolio_state(upro_shares, cash, total)
            print(f"\nPortfolio updated:")
            print(f"  UPRO: {upro_shares:.4f} shares @ ${upro_price:.2f} = ${upro_shares * upro_price:.2f}")
            print(f"  Cash: ${cash:.2f}")
            print(f"  Total: ${total:.2f}")
        else:
            print("ERROR: Could not fetch UPRO price to compute total value")
        return

    # Get current data
    print("\nFetching market data...")
    vix_level = get_current_vix()
    upro_price = get_upro_price()

    if vix_level is None:
        print("FATAL: Could not fetch VIX level. Aborting.")
        sys.exit(1)
    if upro_price is None:
        print("FATAL: Could not fetch UPRO price. Aborting.")
        sys.exit(1)

    print(f"  VIX:  {vix_level:.2f}")
    print(f"  UPRO: ${upro_price:.2f}")

    # Get portfolio
    portfolio = get_portfolio_from_args()
    upro_shares = portfolio["upro_shares"]
    cash_available = portfolio["cash_available"]
    upro_value = upro_shares * upro_price
    total_value = upro_value + cash_available

    print(f"\nCurrent Portfolio (source: {portfolio['source']}):")
    print(f"  UPRO: {upro_shares:.4f} shares = ${upro_value:.2f} ({upro_value/total_value*100:.1f}%)")
    print(f"  Cash: ${cash_available:.2f} ({cash_available/total_value*100:.1f}%)")
    print(f"  Total: ${total_value:.2f}")

    if args.portfolio:
        return

    # Check drawdown protection signals (HC #709 validated rules)
    print("\nDrawdown Protection Check (HC #709 validated):")
    protection = check_drawdown_protection()
    for detail in protection.get('details', []):
        print(f"  {detail}")
    print(f"  => {protection['recommendation']}")

    # Determine target allocation (VIX + protection overlay)
    target = determine_target_allocation(vix_level, protection)
    print(f"\nRegime: {target['regime']}")
    print(f"  {target['description']}")
    if target.get('protection_override'):
        print(f"  ⚠️ PROTECTION OVERRIDE ACTIVE — exposure reduced")

    # Compute trades
    trades = compute_trades(
        target, upro_shares, upro_value,
        cash_available, upro_price, total_value
    )

    print(f"\nTarget Allocation:")
    print(f"  UPRO: {target['upro_pct']*100:.0f}% (${total_value * target['upro_pct']:.2f})")
    print(f"  Cash: {target['cash_pct']*100:.0f}% (${total_value * target['cash_pct']:.2f})")

    print(f"\nAction: {trades['action']}")
    print(f"  {trades['trade_description']}")

    # Whole-share breakdown (useful for non-fractional brokers)
    if trades["action"] != "HOLD":
        whole_shares = int(abs(trades["upro_delta_shares"]))
        frac = abs(trades["upro_delta_shares"]) - whole_shares
        if whole_shares > 0 or frac > 0.01:
            print(f"  (Whole shares: {whole_shares}, fractional: {frac:.4f})")
            print(f"  Robinhood supports fractional shares for UPRO")

    # Execute or log
    executed = False
    if args.execute:
        print("\n*** EXECUTION NOT YET IMPLEMENTED ***")
        print("*** When ready, this will use Robinhood MCP tools ***")
        print("*** For now, manually execute the trade above ***")
        # TODO: Implement via Robinhood MCP:
        #   mcp__robinhood-trading__place_equity_order
        #   mcp__robinhood-trading__review_equity_order
        executed = False

    # Log the decision
    log_decision(date_str, vix_level, target, trades, trades["action"], executed)
    print(f"\nDecision logged to: {LOG_FILE.name}")

    # Summary line for cron/monitoring
    print(f"\n--- SUMMARY: VIX={vix_level:.1f} | {target['regime']} | {trades['action']} | {'EXECUTED' if executed else 'DRY RUN'} ---")


if __name__ == "__main__":
    main()
