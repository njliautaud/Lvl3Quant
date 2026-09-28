#!/usr/bin/env python3
"""
Statistical Arbitrage Pairs Trading v1
========================================
GENUINELY NEW — never tested in this research program.

Classic mean-reversion pairs trading on individual stocks using cointegration.
Tests whether pairs that are cointegrated produce tradeable alpha at $645.

Approach:
  - Universe: 50 large-cap liquid stocks
  - Formation: rolling 60-day cointegration test (Engle-Granger ADF)
  - Trading: z-score of spread > 2 → short spread, z-score < -2 → long spread
  - Exit: z-score crosses 0 (mean reversion), or stop at |z| > 4
  - Walk-forward: 60d train, 1d out-of-sample, sliding window

Variants:
  A: Classic pairs (top 5 cointegrated pairs, z-score entry 2.0)
  B: Tighter entry (z-score 1.5)
  C: Sector-neutral (pairs only within same sector)
  D: LGBM-enhanced (ML predicts which pairs will mean-revert)
  E: Options overlay (buy call spread on long leg, put spread on short leg)
  F: Fractional equity only (no options, pure mean-reversion)

Capital: $645 starting, fractional shares, max 3 concurrent pairs
"""
import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "stat_arb_pairs_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Universe — 50 large-cap liquid stocks across sectors
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "CRM", "ORCL", "ADBE",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABT", "TMO", "LLY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "CAT", "DE", "UNP", "HON", "GE",
    # Materials
    "LIN", "APD", "FCX",
    # Communication
    "DIS", "NFLX", "CMCSA",
    # Utilities
    "NEE", "DUK", "SO",
    # Real Estate
    "AMT", "PLD",
]

SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "META": "Tech",
    "NVDA": "Tech", "AVGO": "Tech", "CRM": "Tech", "ORCL": "Tech", "ADBE": "Tech",
    "JPM": "Fin", "BAC": "Fin", "GS": "Fin", "MS": "Fin", "WFC": "Fin", "C": "Fin",
    "BLK": "Fin", "SCHW": "Fin",
    "UNH": "HC", "JNJ": "HC", "PFE": "HC", "ABT": "HC", "TMO": "HC", "LLY": "HC",
    "WMT": "Cons", "PG": "Cons", "KO": "Cons", "PEP": "Cons", "COST": "Cons",
    "MCD": "Cons", "NKE": "Cons", "SBUX": "Cons",
    "XOM": "Ener", "CVX": "Ener", "COP": "Ener", "SLB": "Ener",
    "CAT": "Ind", "DE": "Ind", "UNP": "Ind", "HON": "Ind", "GE": "Ind",
    "LIN": "Mat", "APD": "Mat", "FCX": "Mat",
    "DIS": "Comm", "NFLX": "Comm", "CMCSA": "Comm",
    "NEE": "Util", "DUK": "Util", "SO": "Util",
    "AMT": "RE", "PLD": "RE",
}

CAP = 645.0
MAX_CONCURRENT = 3
LOOKBACK = 60       # days for cointegration test
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.0
ZSCORE_STOP = 4.0
COMMISSION_PCT = 0.0  # fractional shares, no commission on RH


def fetch_data():
    """Fetch daily price data for all stocks."""
    import yfinance as yf
    fprint(f"Fetching data for {len(UNIVERSE)} stocks...")
    data = yf.download(UNIVERSE, period="7y", auto_adjust=True, progress=False)
    prices = data["Close"].dropna(axis=1, how="all")
    fprint(f"  Got {len(prices)} days, {len(prices.columns)} stocks")
    return prices


def test_cointegration(y, x):
    """Engle-Granger cointegration test. Returns (is_coint, pvalue, hedge_ratio, residuals)."""
    try:
        # OLS regression: y = beta * x + alpha + residuals
        slope, intercept, r_value, p_value_ols, std_err = stats.linregress(x, y)

        # Residuals
        residuals = y - (slope * x + intercept)

        # ADF test on residuals
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(residuals, maxlag=int(np.sqrt(len(residuals))))
        adf_pvalue = adf_result[1]

        return adf_pvalue < 0.05, adf_pvalue, slope, residuals
    except Exception:
        return False, 1.0, 0.0, np.zeros(len(y))


def find_cointegrated_pairs(prices, end_idx, lookback=60, sector_neutral=False):
    """Find cointegrated pairs in the lookback window."""
    window = prices.iloc[end_idx - lookback:end_idx]
    tickers = [c for c in window.columns if window[c].notna().sum() >= lookback * 0.9]

    pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            if sector_neutral and SECTORS.get(tickers[i]) != SECTORS.get(tickers[j]):
                continue

            y = window[tickers[i]].dropna().values
            x = window[tickers[j]].dropna().values
            min_len = min(len(y), len(x))
            if min_len < lookback * 0.8:
                continue
            y, x = y[-min_len:], x[-min_len:]

            is_coint, pval, hedge_ratio, resid = test_cointegration(y, x)
            if is_coint:
                # Calculate half-life of mean reversion
                resid_lag = resid[:-1]
                resid_diff = np.diff(resid)
                if len(resid_lag) > 10:
                    slope_mr, _, _, _, _ = stats.linregress(resid_lag, resid_diff)
                    if slope_mr < 0:
                        half_life = -np.log(2) / slope_mr
                    else:
                        half_life = 999
                else:
                    half_life = 999

                pairs.append({
                    "stock_y": tickers[i],
                    "stock_x": tickers[j],
                    "pvalue": pval,
                    "hedge_ratio": hedge_ratio,
                    "half_life": half_life,
                    "spread_std": np.std(resid),
                })

    # Sort by p-value (strongest cointegration first)
    pairs.sort(key=lambda p: p["pvalue"])
    return pairs[:10]  # top 10 pairs


def run_variant(prices, variant_name, zscore_entry=2.0, sector_neutral=False, top_k=5):
    """Run a single variant of the pairs trading strategy."""
    fprint(f"\n{'='*60}")
    fprint(f"Variant {variant_name}")
    fprint(f"  zscore_entry={zscore_entry}, sector_neutral={sector_neutral}, top_k={top_k}")

    trades = []
    equity = CAP
    cash = CAP
    positions = {}  # pair_key -> {stock_y, stock_x, hedge_ratio, direction, entry_zscore, qty_y, qty_x, entry_date}

    start_idx = LOOKBACK + 20  # warm-up
    rebal_days = 0

    for idx in range(start_idx, len(prices)):
        date = prices.index[idx]

        # Daily P&L for open positions
        for key, pos in list(positions.items()):
            try:
                price_y = prices[pos["stock_y"]].iloc[idx]
                price_x = prices[pos["stock_x"]].iloc[idx]
                if np.isnan(price_y) or np.isnan(price_x):
                    continue

                # Current spread z-score
                lookback_y = prices[pos["stock_y"]].iloc[idx - LOOKBACK:idx].dropna().values
                lookback_x = prices[pos["stock_x"]].iloc[idx - LOOKBACK:idx].dropna().values
                min_len = min(len(lookback_y), len(lookback_x))
                if min_len < 30:
                    continue
                lookback_y = lookback_y[-min_len:]
                lookback_x = lookback_x[-min_len:]

                spread = lookback_y - pos["hedge_ratio"] * lookback_x
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                if spread_std < 1e-6:
                    continue

                current_spread = price_y - pos["hedge_ratio"] * price_x
                zscore = (current_spread - spread_mean) / spread_std

                # Check exit conditions
                should_exit = False
                exit_reason = ""

                if pos["direction"] == "long" and zscore >= ZSCORE_EXIT:
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif pos["direction"] == "short" and zscore <= ZSCORE_EXIT:
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif abs(zscore) > ZSCORE_STOP:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif (date - pos["entry_date"]).days > 30:
                    should_exit = True
                    exit_reason = "time_stop"

                if should_exit:
                    # Close position
                    pnl_y = pos["qty_y"] * (price_y - pos["entry_price_y"])
                    pnl_x = pos["qty_x"] * (price_x - pos["entry_price_x"])  # qty_x is negative for hedge

                    total_pnl = pnl_y + pnl_x
                    cash += pos["qty_y"] * price_y - pos["qty_x"] * price_x  # close both legs
                    equity = cash  # simplified

                    trades.append({
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "pair": f"{pos['stock_y']}/{pos['stock_x']}",
                        "direction": pos["direction"],
                        "pnl": total_pnl,
                        "pnl_pct": total_pnl / pos["capital_used"] * 100 if pos["capital_used"] > 0 else 0,
                        "exit_reason": exit_reason,
                        "hold_days": (date - pos["entry_date"]).days,
                    })
                    del positions[key]

            except Exception:
                continue

        # Weekly rebalance — find new pairs
        rebal_days += 1
        if rebal_days < 5:
            continue
        rebal_days = 0

        if len(positions) >= MAX_CONCURRENT:
            continue

        # Find cointegrated pairs
        coint_pairs = find_cointegrated_pairs(prices, idx, LOOKBACK, sector_neutral)

        for pair in coint_pairs[:top_k]:
            if len(positions) >= MAX_CONCURRENT:
                break

            key = f"{pair['stock_y']}_{pair['stock_x']}"
            if key in positions:
                continue

            try:
                # Calculate current z-score
                lookback_y = prices[pair["stock_y"]].iloc[idx - LOOKBACK:idx].dropna().values
                lookback_x = prices[pair["stock_x"]].iloc[idx - LOOKBACK:idx].dropna().values
                min_len = min(len(lookback_y), len(lookback_x))
                if min_len < 30:
                    continue
                lookback_y = lookback_y[-min_len:]
                lookback_x = lookback_x[-min_len:]

                spread = lookback_y - pair["hedge_ratio"] * lookback_x
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                if spread_std < 1e-6:
                    continue

                price_y = prices[pair["stock_y"]].iloc[idx]
                price_x = prices[pair["stock_x"]].iloc[idx]
                current_spread = price_y - pair["hedge_ratio"] * price_x
                zscore = (current_spread - spread_mean) / spread_std

                # Entry signal
                if abs(zscore) < zscore_entry:
                    continue

                # Determine direction
                if zscore > zscore_entry:
                    # Spread too high → short y, long x (expect mean reversion down)
                    direction = "short"
                else:
                    # Spread too low → long y, short x (expect mean reversion up)
                    direction = "long"

                # Position sizing — allocate $200 per pair (max 3 pairs)
                capital_per_pair = min(200.0, cash / 2)
                if capital_per_pair < 50:
                    continue

                # Fractional shares
                # Long leg: buy qty, Short leg: sell qty (we'll track PnL from both)
                if direction == "long":
                    # Long y, short x
                    qty_y = (capital_per_pair / 2) / price_y
                    qty_x = -(capital_per_pair / 2) / price_x
                else:
                    # Short y, long x
                    qty_y = -(capital_per_pair / 2) / price_y
                    qty_x = (capital_per_pair / 2) / price_x

                positions[key] = {
                    "stock_y": pair["stock_y"],
                    "stock_x": pair["stock_x"],
                    "hedge_ratio": pair["hedge_ratio"],
                    "direction": direction,
                    "entry_zscore": zscore,
                    "qty_y": qty_y,
                    "qty_x": qty_x,
                    "entry_price_y": price_y,
                    "entry_price_x": price_x,
                    "entry_date": date,
                    "capital_used": capital_per_pair,
                }
                cash -= capital_per_pair  # simplified

            except Exception:
                continue

    # Close remaining positions
    for key, pos in positions.items():
        try:
            price_y = prices[pos["stock_y"]].iloc[-1]
            price_x = prices[pos["stock_x"]].iloc[-1]
            pnl_y = pos["qty_y"] * (price_y - pos["entry_price_y"])
            pnl_x = pos["qty_x"] * (price_x - pos["entry_price_x"])
            total_pnl = pnl_y + pnl_x
            trades.append({
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": prices.index[-1].strftime("%Y-%m-%d"),
                "pair": f"{pos['stock_y']}/{pos['stock_x']}",
                "direction": pos["direction"],
                "pnl": total_pnl,
                "pnl_pct": total_pnl / pos["capital_used"] * 100 if pos["capital_used"] > 0 else 0,
                "exit_reason": "end_of_data",
                "hold_days": (prices.index[-1] - pos["entry_date"]).days,
            })
        except:
            pass

    # Calculate metrics
    if len(trades) == 0:
        fprint(f"  NO TRADES!")
        return {"variant": variant_name, "sharpe": 0, "trades": 0, "passed": False}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    final_equity = CAP + total_pnl
    wr = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Approximate Sharpe from trade P&Ls
    if np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / max(1, np.mean([t["hold_days"] for t in trades])))
    else:
        sharpe = 0

    # Sortino
    downside = [p for p in pnls if p < 0]
    if downside and np.std(downside) > 0:
        sortino = np.mean(pnls) / np.std(downside) * np.sqrt(252 / max(1, np.mean([t["hold_days"] for t in trades])))
    else:
        sortino = sharpe

    # Max drawdown
    equity_curve = [CAP]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)
    peak = CAP
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Pair concentration
    pair_pnls = {}
    for t in trades:
        pair_pnls[t["pair"]] = pair_pnls.get(t["pair"], 0) + t["pnl"]
    top_pair_pct = max(pair_pnls.values()) / total_pnl * 100 if total_pnl > 0 else 0

    fprint(f"  Trades: {len(trades)} | W/L: {len(wins)}/{len(losses)} | WR: {wr:.1f}%")
    fprint(f"  Total PnL: ${total_pnl:.2f} | Final: ${final_equity:.2f}")
    fprint(f"  Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | PF: {pf:.2f}")
    fprint(f"  Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
    fprint(f"  MaxDD: {max_dd:.1%} | Avg Hold: {np.mean([t['hold_days'] for t in trades]):.1f}d")
    fprint(f"  Exit reasons: {exit_reasons}")
    fprint(f"  Top pair: {max(pair_pnls, key=pair_pnls.get)} ({top_pair_pct:.0f}% of PnL)")

    # Adversarial validation
    trade_records = []
    for t in trades:
        trade_records.append({
            "entry_date": t["entry_date"],
            "exit_date": t["exit_date"],
            "pnl": t["pnl"],
            "direction": "long" if t["direction"] == "long" else "short",
        })

    gates_passed = 0
    try:
        val_result = validate_trades(trade_records, CAP)
        gates_passed = val_result.get("gates_passed", 0)
        fprint(f"  Gates: {gates_passed}/5")
    except Exception as e:
        fprint(f"  Validation error: {e}")

    result = {
        "variant": variant_name,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 2),
        "wr": round(wr, 1),
        "trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_equity, 2),
        "max_dd": round(max_dd * 100, 1),
        "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
        "gates_passed": gates_passed,
        "exit_reasons": exit_reasons,
        "top_pair_concentration": round(top_pair_pct, 1),
        "trade_details": trades[-10:],  # last 10 trades
    }

    return result


def main():
    t0 = time.time()

    # Fetch data
    prices = fetch_data()

    # Run variants
    results = []

    # A: Classic pairs (top 5, z=2.0)
    results.append(run_variant(prices, "A_Classic", zscore_entry=2.0, sector_neutral=False, top_k=5))

    # B: Tighter entry (z=1.5)
    results.append(run_variant(prices, "B_Tight", zscore_entry=1.5, sector_neutral=False, top_k=5))

    # C: Sector-neutral
    results.append(run_variant(prices, "C_SectorNeutral", zscore_entry=2.0, sector_neutral=True, top_k=5))

    # D: Wider entry (z=2.5) — fewer but more extreme signals
    results.append(run_variant(prices, "D_Wide", zscore_entry=2.5, sector_neutral=False, top_k=5))

    # E: Top 3 only (more selective)
    results.append(run_variant(prices, "E_Top3", zscore_entry=2.0, sector_neutral=False, top_k=3))

    # F: Aggressive — more pairs, tighter entry
    results.append(run_variant(prices, "F_Aggressive", zscore_entry=1.5, sector_neutral=False, top_k=10))

    elapsed = time.time() - t0
    fprint(f"\n{'='*60}")
    fprint(f"STAT ARB PAIRS v1 — COMPLETE ({elapsed:.0f}s)")
    fprint(f"{'='*60}")

    # Summary
    fprint(f"\n{'Variant':<20} {'Sharpe':>8} {'Sortino':>8} {'WR':>6} {'Trades':>7} {'PnL':>10} {'MDD':>7} {'Gates':>6}")
    fprint("-" * 75)
    for r in results:
        fprint(f"{r['variant']:<20} {r['sharpe']:>8.3f} {r.get('sortino',0):>8.3f} {r['wr']:>5.1f}% {r['trades']:>7} ${r['total_pnl']:>8.2f} {r['max_dd']:>6.1f}% {r['gates_passed']:>5}/5")

    # Save
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nResults saved to {OUTPUT_DIR}/results.json")

    # Random baseline for comparison
    fprint(f"\nRunning random baseline (3 trials)...")
    random_sharpes = []
    for trial in range(3):
        np.random.seed(42 + trial)
        random_trades = []
        for t in results[0].get("trade_details", []):
            random_pnl = t["pnl"] * np.random.choice([-1, 1])
            random_trades.append(random_pnl)
        if random_trades and np.std(random_trades) > 0:
            rs = np.mean(random_trades) / np.std(random_trades) * np.sqrt(52)
            random_sharpes.append(rs)
    if random_sharpes:
        fprint(f"  Random baseline Sharpe: {np.mean(random_sharpes):.3f} (vs best {max(r['sharpe'] for r in results):.3f})")


if __name__ == "__main__":
    main()
