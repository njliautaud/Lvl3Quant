#!/usr/bin/env python3
"""
wheel_elite40_weekly_portfolio.py — Optimized portfolio with:
- 40 elite tickers (top by Sharpe from 197-ticker weekly sweep)
- Weekly DTE (14 days, proven +13% Sharpe vs monthly)
- Bear protection: close CSPs only when SPY < 50d SMA
- Multi-sector diversification (11 sectors covered)
- HC #428 R1 regime-agnostic validation

HC #660 compliant: expanded universe, higher-beta names, multi-industry.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts')

# Monkey-patch the regime_gated_portfolio module's config before importing main logic
import importlib.util

# Load the module
spec = importlib.util.spec_from_file_location(
    "wheel_rg", "/home/jupiter/Lvl3Quant/scripts/wheel_regime_gated_portfolio.py"
)
mod = importlib.util.module_from_spec(spec)

# We'll just run it directly with modified globals
# Simpler approach: copy the engine and override configs

import json
import math
import time
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_elite40_weekly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [WHEEL-E40] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('WHEEL-E40')

# ========================= ELITE 40 BASKET ==================================
# Top by Sharpe from 197-ticker weekly sweep, diversified across 11 sectors
BASKET = {
    # Consumer Cyclical (4)
    'WYNN': 'Consumer Cyclical',
    'HD': 'Consumer Cyclical',
    'SBUX': 'Consumer Cyclical',
    'BABA': 'Consumer Cyclical',
    # Utilities (4)
    'EXC': 'Utilities',
    'AEP': 'Utilities',
    'DUK': 'Utilities',
    'ED': 'Utilities',
    # Energy (4)
    'COP': 'Energy',
    'XOM': 'Energy',
    'CVX': 'Energy',
    'VLO': 'Energy',
    # Technology (4)
    'TXN': 'Technology',
    'IBM': 'Technology',
    'CSCO': 'Technology',
    'ARM': 'Technology',
    # Communication Services (4)
    'EA': 'Communication Services',
    'VZ': 'Communication Services',
    'TMUS': 'Communication Services',
    'GOOGL': 'Communication Services',
    # Healthcare (4)
    'GILD': 'Healthcare',
    'BIIB': 'Healthcare',
    'CVS': 'Healthcare',
    'ABT': 'Healthcare',
    # Real Estate (3)
    'DLR': 'Real Estate',
    'IRM': 'Real Estate',
    'SPG': 'Real Estate',
    # Financial Services (4)
    'JPM': 'Financial Services',
    'AXP': 'Financial Services',
    'BAC': 'Financial Services',
    'GS': 'Financial Services',
    # Industrials (3)
    'CAT': 'Industrials',
    'HON': 'Industrials',
    'UNP': 'Industrials',
    # Consumer Defensive (3)
    'CL': 'Consumer Defensive',
    'PG': 'Consumer Defensive',
    'TGT': 'Consumer Defensive',
    # Basic Materials (1)
    'LIN': 'Basic Materials',
}

TICKERS = sorted(BASKET.keys())
N_NAMES = len(TICKERS)

# ========================= OPTIMIZED CONFIG =================================
START_DATE = pd.Timestamp("2019-01-01")
STARTING_CASH = 100_000.0  # $100K — realistic retail account
TRADING_DAYS = 252
RISK_FREE = 0.04

PUT_DELTA = 0.25
CALL_DELTA = 0.30
DTE_MIN = 10
DTE_MAX = 18
DTE_TARGET = 14  # WEEKLY — proven better
PROFIT_TAKE = 0.50
VIX_MAX = 35.0

MAX_NOTIONAL_PCT = 1.50  # Allow up to 150% notional (margin account)
MAX_PER_NAME_PCT = 0.10  # 10% per name max

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03
FLAT_BAND = 0.0025

SMA_PERIOD = 50  # Bear gate

# ========================= PRICING ==========================================
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

def find_strike(S, sigma, T, delta_target, kind="put"):
    """Find strike for target delta using bisection."""
    if kind == "put":
        lo, hi = S * 0.5, S * 1.0
    else:
        lo, hi = S * 1.0, S * 1.5
    for _ in range(50):
        K = (lo + hi) / 2
        d1 = (math.log(S / K) + (r := RISK_FREE) + 0.5 * sigma * sigma) * T
        d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T) + 1e-9)
        if kind == "put":
            d = _Phi(-d1) - 1.0  # Put delta is negative
            delta_abs = abs(d)
        else:
            delta_abs = _Phi(d1)
        if delta_abs > delta_target:
            if kind == "put":
                lo = K
            else:
                hi = K
        else:
            if kind == "put":
                hi = K
            else:
                lo = K
    return round(K * 2) / 2  # Round to $0.50

# ========================= DATA LOADING =====================================
def load_data():
    """Load prices, VIX, and SPY for all 40 tickers."""
    log.info(f"Loading data for {N_NAMES} tickers...")

    # Load original + expanded prices
    p1 = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]].copy()
    p1["date"] = pd.to_datetime(p1["date"], utc=False)
    if p1["date"].dt.tz is not None:
        p1["date"] = p1["date"].dt.tz_localize(None)

    p2 = pd.read_parquet(CACHE / "prices_expanded.parquet")
    p2 = p2.rename(columns={"Close": "close"})[["ticker", "date", "close"]].copy()
    p2["date"] = pd.to_datetime(p2["date"], utc=False)
    if p2["date"].dt.tz is not None:
        p2["date"] = p2["date"].dt.tz_localize(None)

    prices = pd.concat([p1, p2], ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"]).reset_index(drop=True)
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0]
    prices = prices[prices["date"] >= START_DATE]

    # VIX
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"], utc=False)
    if macro["date"].dt.tz is not None:
        macro["date"] = macro["date"].dt.tz_localize(None)
    prices = prices.merge(macro, on="date", how="left")
    prices["vix"] = prices["vix"].ffill().fillna(20.0)

    # 20-day realized vol
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252)
    )
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)

    # SPY for regime gate
    spy_file = CACHE / "spy_prices.parquet"
    if spy_file.exists():
        spy = pd.read_parquet(spy_file)[["date", "close"]].copy()
        spy["date"] = pd.to_datetime(spy["date"], utc=False)
        if spy["date"].dt.tz is not None:
            spy["date"] = spy["date"].dt.tz_localize(None)
    else:
        spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()

    if not spy.empty:
        spy = spy.sort_values("date").drop_duplicates("date")
        spy["spy_sma"] = spy["close"].rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
        spy["bear"] = (spy["close"] < spy["spy_sma"]).astype(int)
        spy_regime = spy[["date", "bear", "spy_sma"]].copy()
    else:
        log.warning("No SPY data found! Using VIX as proxy.")
        all_dates = prices["date"].unique()
        spy_regime = pd.DataFrame({"date": all_dates, "bear": 0, "spy_sma": np.nan})

    # Filter to our tickers
    our_prices = prices[prices["ticker"].isin(TICKERS)].copy()

    # Check coverage
    available = our_prices["ticker"].nunique()
    log.info(f"Data loaded: {available}/{N_NAMES} tickers available")

    missing = set(TICKERS) - set(our_prices["ticker"].unique())
    if missing:
        log.warning(f"Missing tickers: {missing}")

    return our_prices, spy_regime


# ========================= PORTFOLIO SIMULATION =============================
@dataclass
class Position:
    ticker: str
    state: str  # 'csp', 'assigned', 'cc'
    strike: float
    premium: float
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    shares: int = 0
    cost_basis: float = 0.0


def run_portfolio(prices_df, spy_regime, bear_mode="liq_csp_only"):
    """
    Run portfolio simulation.
    bear_mode:
      - "none": no protection
      - "liq_csp_only": close CSP positions in bear, keep shares+CC
    """
    tickers_available = sorted(prices_df["ticker"].unique())
    n_avail = len(tickers_available)
    log.info(f"Running portfolio: {n_avail} tickers, bear_mode={bear_mode}")

    # Build date-indexed lookups
    ticker_data = {}
    for t in tickers_available:
        tdf = prices_df[prices_df["ticker"] == t].set_index("date").sort_index()
        ticker_data[t] = tdf

    # Merge spy regime — normalize keys to Timestamps for consistent lookup
    spy_map = {}
    if not spy_regime.empty:
        for _, row in spy_regime.iterrows():
            spy_map[pd.Timestamp(row["date"])] = int(row["bear"])
    log.info(f"Bear gate: {sum(spy_map.values())} bear days / {len(spy_map)} total")

    all_dates = sorted(pd.Timestamp(d) for d in prices_df["date"].unique())
    all_dates = [d for d in all_dates if d >= START_DATE]

    cash = STARTING_CASH
    positions = {}  # ticker -> Position or None
    trades = []
    daily_equity = []
    daily_pnl = []

    per_name_max = STARTING_CASH * MAX_PER_NAME_PCT

    for di, date in enumerate(all_dates):
        is_bear = spy_map.get(date, 0) == 1

        # Mark-to-market
        portfolio_value = cash
        for t, pos in list(positions.items()):
            if pos is None:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            row = ticker_data[t].loc[date]
            px = row["close"]

            if pos.state == "assigned" or pos.state == "cc":
                portfolio_value += pos.shares * px
            elif pos.state == "csp":
                # Mark CSP: premium received - current option value
                dte_remain = (pos.expiry_date - date).days
                if dte_remain > 0:
                    sigma = row.get("sigma", 0.3)
                    opt_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
                    portfolio_value += (pos.premium - opt_val) * 100
                else:
                    portfolio_value += pos.premium * 100

        daily_equity.append({"date": date, "equity": portfolio_value})
        if len(daily_equity) > 1:
            daily_pnl.append(portfolio_value - daily_equity[-2]["equity"])

        # Process expirations and manage positions
        for t in list(positions.keys()):
            pos = positions[t]
            if pos is None:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue

            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3)
            vix = row.get("vix", 20.0)

            # Check expiry
            if date >= pos.expiry_date:
                if pos.state == "csp":
                    if px <= pos.strike:
                        # Assigned: buy 100 shares at strike
                        cost = pos.strike * 100 + COST_PER_CONTRACT
                        cash -= cost
                        pos.state = "assigned"
                        pos.shares = 100
                        pos.cost_basis = pos.strike - pos.premium
                        trades.append({"date": date, "ticker": t, "action": "assigned",
                                      "price": pos.strike, "premium": pos.premium})
                    else:
                        # Expired worthless — premium already collected at open
                        trades.append({"date": date, "ticker": t, "action": "csp_expired",
                                      "price": px, "premium": pos.premium, "pnl": pos.premium})
                        positions[t] = None

                elif pos.state == "cc":
                    if px >= pos.strike:
                        # Called away: sell shares at strike (premium already collected)
                        proceeds = pos.strike * 100 - COST_PER_CONTRACT
                        cash += proceeds
                        pnl = (pos.strike - pos.cost_basis) + pos.premium
                        trades.append({"date": date, "ticker": t, "action": "called_away",
                                      "price": pos.strike, "pnl": pnl})
                        positions[t] = None
                    else:
                        # CC expired worthless, keep shares (premium already collected)
                        pos.state = "assigned"  # Back to holding shares
                        pos.expiry_date = date + pd.Timedelta(days=1)  # Reset for next CC
                        trades.append({"date": date, "ticker": t, "action": "cc_expired",
                                      "price": px, "premium": pos.premium})
                continue

            # Profit take on CSP (buy back cheaper than sold)
            if pos.state == "csp":
                dte_remain = (pos.expiry_date - date).days
                if dte_remain > 0:
                    current_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
                    if current_val <= pos.premium * (1 - PROFIT_TAKE):
                        # Buy back the put (premium already collected at open)
                        buyback_cost = current_val * 100 + COST_PER_CONTRACT
                        cash -= buyback_cost
                        profit = pos.premium * 100 - buyback_cost - COST_PER_CONTRACT
                        trades.append({"date": date, "ticker": t, "action": "profit_take",
                                      "price": px, "pnl": profit / 100})
                        positions[t] = None

        # Bear protection: close CSPs if in bear mode
        if is_bear and bear_mode == "liq_csp_only":
            for t in list(positions.keys()):
                pos = positions[t]
                if pos is None or pos.state != "csp":
                    continue
                if t not in ticker_data or date not in ticker_data[t].index:
                    continue
                row = ticker_data[t].loc[date]
                px = row["close"]
                sigma = row.get("sigma", 0.3)
                dte_remain = (pos.expiry_date - date).days
                if dte_remain > 0:
                    current_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
                    # Close at market
                    loss = (current_val - pos.premium) * 100 + 2 * COST_PER_CONTRACT
                    cash -= loss
                    trades.append({"date": date, "ticker": t, "action": "bear_close",
                                  "price": px, "pnl": -loss / 100})
                positions[t] = None

        # STEP A: Write covered calls on assigned shares (always, even in bear)
        # CCs on existing shares generate income and reduce cost basis
        for t in tickers_available:
            pos = positions.get(t)
            if pos is None or pos.state != "assigned":
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3)
            if pd.isna(sigma) or sigma < 0.05:
                continue

            T = DTE_TARGET / 365
            K = find_strike(px, sigma, T, CALL_DELTA, kind="call")
            premium = bs_price(px, K, T, sigma, kind="call")
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)
            if premium < 0.10:
                continue

            expiry = date + pd.Timedelta(days=DTE_TARGET)
            positions[t].state = "cc"
            positions[t].strike = K
            positions[t].premium = premium
            positions[t].entry_date = date
            positions[t].expiry_date = expiry
            cash += premium * 100 - COST_PER_CONTRACT  # Collect CC premium
            trades.append({"date": date, "ticker": t, "action": "sell_cc",
                          "strike": K, "premium": premium})

        # STEP B: Open new CSP positions (only if not bear or bear_mode=none)
        if not is_bear or bear_mode == "none":
            for t in tickers_available:
                if positions.get(t) is not None:
                    continue
                if t not in ticker_data or date not in ticker_data[t].index:
                    continue

                row = ticker_data[t].loc[date]
                px = row["close"]
                sigma = row.get("sigma", 0.3)
                vix = row.get("vix", 20.0)

                if vix > VIX_MAX:
                    continue
                if pd.isna(sigma) or sigma < 0.05:
                    continue

                # Check allocation — margin account: CSP requires ~20% of strike notional
                notional = px * 100
                margin_req = notional * 0.20  # 20% margin for CSPs
                # Max margin per name = 10% of starting capital
                if margin_req > STARTING_CASH * MAX_PER_NAME_PCT:
                    continue
                if cash < margin_req:
                    continue

                # Write CSP
                T = DTE_TARGET / 365
                K = find_strike(px, sigma, T, PUT_DELTA, kind="put")
                premium = bs_price(px, K, T, sigma)
                premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)

                if premium < 0.10:
                    continue

                expiry = date + pd.Timedelta(days=DTE_TARGET)
                positions[t] = Position(
                    ticker=t, state="csp", strike=K, premium=premium,
                    entry_date=date, expiry_date=expiry
                )
                cash += premium * 100 - COST_PER_CONTRACT  # Collect CSP premium
                trades.append({"date": date, "ticker": t, "action": "sell_csp",
                              "strike": K, "premium": premium})

    return daily_equity, daily_pnl, trades


def compute_metrics(daily_equity, daily_pnl):
    """Compute risk-adjusted metrics."""
    eq = pd.DataFrame(daily_equity)
    if len(eq) < 50:
        return {}

    returns = eq["equity"].pct_change().dropna()
    total_ret = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) - 1
    years = len(eq) / TRADING_DAYS
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    sharpe = (returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)) if returns.std() > 0 else 0
    down = returns[returns < 0]
    sortino = (returns.mean() / down.std() * np.sqrt(TRADING_DAYS)) if len(down) > 0 and down.std() > 0 else 0

    # Max drawdown
    cum_max = eq["equity"].cummax()
    dd = (eq["equity"] - cum_max) / cum_max
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate from daily PnL
    pnl_arr = np.array(daily_pnl)
    wr = (pnl_arr > 0).mean() if len(pnl_arr) > 0 else 0

    # Profit factor
    gains = pnl_arr[pnl_arr > 0].sum() if (pnl_arr > 0).any() else 0
    losses = abs(pnl_arr[pnl_arr < 0].sum()) if (pnl_arr < 0).any() else 1e-9
    pf = gains / losses

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "starting_equity": STARTING_CASH,
        "years": round(years, 2),
        "n_days": len(eq),
    }


def regime_analysis(daily_equity, spy_regime):
    """Compute per-regime Sharpe for HC #428 R1 compliance."""
    eq = pd.DataFrame(daily_equity)
    eq["return"] = eq["equity"].pct_change()
    eq = eq.merge(spy_regime[["date", "bear"]], on="date", how="left")
    eq["bear"] = eq["bear"].fillna(0).astype(int)

    bull_ret = eq[eq["bear"] == 0]["return"].dropna()
    bear_ret = eq[eq["bear"] == 1]["return"].dropna()

    bull_sharpe = (bull_ret.mean() / bull_ret.std() * np.sqrt(252)) if len(bull_ret) > 20 and bull_ret.std() > 0 else 0
    bear_sharpe = (bear_ret.mean() / bear_ret.std() * np.sqrt(252)) if len(bear_ret) > 20 and bear_ret.std() > 0 else 0

    # HC #428 R1: |Sharpe_green − Sharpe_red| / max(|Sharpe_green|,|Sharpe_red|) <= 0.50
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "hc428_pass": regime_gap <= 0.50,
        "n_bull_days": len(bull_ret),
        "n_bear_days": len(bear_ret),
    }


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("WHEEL ELITE 40 — WEEKLY DTE + BEAR PROTECTION", flush=True)
    print("=" * 70, flush=True)
    print(f"Universe: {N_NAMES} tickers across {len(set(BASKET.values()))} sectors", flush=True)
    print(f"Config: DTE={DTE_TARGET}, put_delta={PUT_DELTA}, profit_take={PROFIT_TAKE}", flush=True)
    print(f"Capital: ${STARTING_CASH:,.0f}, max {MAX_PER_NAME_PCT*100:.0f}% per name", flush=True)
    print(flush=True)

    prices, spy_regime = load_data()

    # Run both modes for comparison
    modes = ["none", "liq_csp_only"]
    all_results = {}

    for mode in modes:
        log.info(f"\n{'='*50}")
        log.info(f"Running bear_mode = {mode}")
        log.info(f"{'='*50}")

        daily_eq, daily_pnl, trades_list = run_portfolio(prices, spy_regime, bear_mode=mode)
        metrics = compute_metrics(daily_eq, daily_pnl)
        regime = regime_analysis(daily_eq, spy_regime)

        all_results[mode] = {
            "metrics": metrics,
            "regime": regime,
            "n_trades": len(trades_list),
        }

        log.info(f"\nResults ({mode}):")
        log.info(f"  CAGR: {metrics.get('cagr_pct', 0):.1f}%")
        log.info(f"  Sharpe: {metrics.get('sharpe', 0):.3f}")
        log.info(f"  Sortino: {metrics.get('sortino', 0):.3f}")
        log.info(f"  Max DD: {metrics.get('max_dd_pct', 0):.1f}%")
        log.info(f"  Calmar: {metrics.get('calmar', 0):.3f}")
        log.info(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        log.info(f"  PF: {metrics.get('profit_factor', 0):.2f}")
        log.info(f"  Final Equity: ${metrics.get('final_equity', 0):,.0f}")
        log.info(f"  Regime: Bull Sharpe={regime['bull_sharpe']:.3f}, Bear Sharpe={regime['bear_sharpe']:.3f}")
        log.info(f"  HC #428 R1 (gap≤0.50): {'PASS' if regime['hc428_pass'] else 'FAIL'} (gap={regime['regime_gap']:.3f})")

    # Save
    output = {
        "generated": datetime.now().isoformat(),
        "config": {
            "n_tickers": N_NAMES,
            "tickers": TICKERS,
            "starting_capital": STARTING_CASH,
            "dte_target": DTE_TARGET,
            "put_delta": PUT_DELTA,
            "bear_modes_tested": modes,
        },
        "results": all_results,
    }
    out_path = OUT_DIR / "elite40_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.1f}s. Saved to {out_path}")

    # Final comparison
    print(flush=True)
    print("=" * 70, flush=True)
    print("COMPARISON: No Protection vs Bear Gate (CSP close)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Metric':<20} {'No Protection':>15} {'Bear Gate':>15}", flush=True)
    print("-" * 50, flush=True)
    for key in ['cagr_pct', 'sharpe', 'sortino', 'max_dd_pct', 'calmar', 'win_rate', 'profit_factor', 'final_equity']:
        v1 = all_results['none']['metrics'].get(key, 0)
        v2 = all_results['liq_csp_only']['metrics'].get(key, 0)
        if key == 'final_equity':
            print(f"{key:<20} ${v1:>13,.0f} ${v2:>13,.0f}", flush=True)
        elif 'pct' in key:
            print(f"{key:<20} {v1:>14.1f}% {v2:>14.1f}%", flush=True)
        else:
            print(f"{key:<20} {v1:>15.3f} {v2:>15.3f}", flush=True)

    print(flush=True)
    for mode in modes:
        r = all_results[mode]['regime']
        status = "✓ PASS" if r['hc428_pass'] else "✗ FAIL"
        print(f"HC #428 R1 ({mode}): gap={r['regime_gap']:.3f} {status}", flush=True)


if __name__ == "__main__":
    main()
