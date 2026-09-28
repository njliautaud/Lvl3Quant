#!/usr/bin/env python3
"""
Share Buyback Announcement Backtest (Price+Volume Proxy)
========================================================
Academic basis: Ikenberry, Lakonishok & Vermaelen (1995)
Companies announcing/executing buybacks outperform over 1-12 months.

Proxy signal: detect stocks where:
  (a) price is within 5% of 52-week low, AND
  (b) 20-day avg volume is 1.5x above 60-day avg volume
     (institutional accumulation consistent with buyback execution)

Universe: 20 mega-cap tech stocks
OOT: Jan 2022 - Jul 2026
Cost: $0 commission (Robinhood), 0.02% slippage per side

Variants:
  A: Basic buyback signal (5% of 52w low + vol surge), 40d hold
  B: Extended hold (same signal, 60d hold)
  C: Tight entry (3% of 52w low), 40d hold
  D: Trend filter (+ 200-SMA rising), 40d hold
  E: Sector washout (3+ stocks trigger within 10 days), 40d hold
  F: Momentum confirmation (signal + wait for 3 consecutive up days), 40d hold

Position sizing: $645 account, max $200/trade, max 3 concurrent.

5-gate validation:
  1. Sharpe > 0.5 (rf=4.5%)
  2. Permutation test 1000x, p < 0.05
  3. Regime gap < 0.5 (bull vs bear SPY > 200-SMA)
  4. MaxDD > -50%
  5. Min 20 trades
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
CACHE_DIR = Path("/home/jupiter/Lvl3Quant/scripts/cache/buyback_announcement")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/buyback_results.json"

OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DATA_START = "2020-10-01"  # extra lookback for 252-day rolling low + SMAs

SLIPPAGE_PCT = 0.0002   # 0.02% per side
COMMISSION = 0.0

ACCOUNT_SIZE = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
RF_ANNUAL = 0.045

N_PERMUTATIONS = 1000
RANDOM_SEED = 42

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "AVGO", "CRM", "NFLX", "ADBE", "INTC", "CSCO", "QCOM", "TXN",
    "MU", "AMAT", "LRCX", "ISRG",
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Data Download ─────────────────────────────────────────────────────────

def download_prices(tickers):
    cache_file = CACHE_DIR / "prices.parquet"
    if cache_file.exists():
        log("Loading cached prices")
        return pd.read_parquet(cache_file)
    log(f"Downloading prices for {len(tickers)} tickers...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, threads=True)
    close = data["Close"]
    close.to_parquet(cache_file)
    log(f"Prices shape: {close.shape}")
    return close


def download_volumes(tickers):
    cache_file = CACHE_DIR / "volumes.parquet"
    if cache_file.exists():
        log("Loading cached volumes")
        return pd.read_parquet(cache_file)
    log(f"Downloading volumes for {len(tickers)} tickers...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, threads=True)
    vol = data["Volume"]
    vol.to_parquet(cache_file)
    log(f"Volumes shape: {vol.shape}")
    return vol


def download_spy():
    cache_file = CACHE_DIR / "spy.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    log("Downloading SPY...")
    spy = yf.download("SPY", start=DATA_START, end=OOT_END, auto_adjust=True)
    spy.to_parquet(cache_file)
    return spy


# ── Signal Generation ─────────────────────────────────────────────────────

def compute_signals(prices, volumes, spy_data, variant="A"):
    """
    Generate buyback-proxy signals.
    Returns list of trade dicts with entry/exit info.
    """
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    # SPY regime
    spy_close = spy_data["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_sma200 = spy_close.rolling(200).mean()

    # Proximity threshold to 52w low
    proximity_pct = 0.03 if variant == "C" else 0.05

    # Hold period
    hold_days = 60 if variant == "B" else 40

    # Pre-compute per-ticker rolling features
    rolling_low_252 = prices.rolling(252, min_periods=200).min()
    vol_ma20 = volumes.rolling(20, min_periods=15).mean()
    vol_ma60 = volumes.rolling(60, min_periods=40).mean()
    sma200 = prices.rolling(200, min_periods=150).mean()
    sma200_slope = sma200.diff(20)  # 20-day change in 200-SMA

    # Raw signal: near 52w low AND volume accumulation
    # price <= rolling_low * (1 + proximity_pct)
    near_low = prices <= rolling_low_252 * (1 + proximity_pct)
    # 20d avg vol >= 1.5 * 60d avg vol
    vol_surge = vol_ma20 >= 1.5 * vol_ma60

    raw_signal = near_low & vol_surge

    # Shift by 1 to prevent look-ahead
    raw_signal = raw_signal.shift(1).fillna(False)
    near_low_shifted = near_low.shift(1).fillna(False)
    vol_surge_shifted = vol_surge.shift(1).fillna(False)

    # For variant D: 200-SMA rising filter (shifted)
    sma_rising = (sma200_slope > 0).shift(1).fillna(False)

    # For variant F: 3 consecutive up days
    daily_ret = prices.pct_change()
    up_day = daily_ret > 0
    consec_up_3 = up_day.rolling(3, min_periods=3).sum() == 3
    consec_up_3 = consec_up_3.shift(1).fillna(False)

    # Collect all raw signal dates for variant E (sector washout)
    all_signal_dates = []
    if variant == "E":
        for ticker in UNIVERSE:
            if ticker not in raw_signal.columns:
                continue
            sig = raw_signal[ticker]
            for dt in sig.index:
                if dt < oot_start or dt > oot_end:
                    continue
                if sig.loc[dt]:
                    all_signal_dates.append((dt, ticker))

    trades = []

    for ticker in UNIVERSE:
        if ticker not in prices.columns or ticker not in volumes.columns:
            continue

        ticker_prices = prices[ticker].dropna()
        if ticker_prices.empty:
            continue

        sig_col = raw_signal[ticker] if ticker in raw_signal.columns else pd.Series(False, index=prices.index)

        for dt in sig_col.index:
            if dt < oot_start or dt > oot_end:
                continue
            if not sig_col.loc[dt]:
                continue

            # Variant D: require 200-SMA rising
            if variant == "D":
                if ticker in sma_rising.columns and not sma_rising[ticker].loc[dt]:
                    continue

            # Variant E: require 3+ stocks triggering within 10 days
            if variant == "E":
                nearby_tickers = set()
                for sig_dt, sig_tk in all_signal_dates:
                    if abs((sig_dt - dt).days) <= 10:
                        nearby_tickers.add(sig_tk)
                if len(nearby_tickers) < 3:
                    continue

            # Variant F: wait for 3 consecutive up days
            if variant == "F":
                # Find first occurrence of 3 consec up days after signal
                future_consec = consec_up_3[ticker] if ticker in consec_up_3.columns else None
                if future_consec is None:
                    continue
                future = future_consec.loc[dt:]
                confirm_dates = future[future].index
                if len(confirm_dates) == 0:
                    continue
                # Use the first confirmation date as entry trigger
                confirm_dt = confirm_dates[0]
                if confirm_dt == dt:
                    # Already confirmed on signal day, enter next day
                    entry_candidates = ticker_prices.index[ticker_prices.index > confirm_dt]
                else:
                    # Enter the day after confirmation
                    entry_candidates = ticker_prices.index[ticker_prices.index > confirm_dt]
                if len(entry_candidates) < 5:
                    continue
                # Don't look more than 20 trading days out for confirmation
                if (confirm_dt - dt).days > 30:
                    continue
                entry_date = entry_candidates[0]
            else:
                # Standard entry: next trading day after signal
                entry_candidates = ticker_prices.index[ticker_prices.index > dt]
                if len(entry_candidates) < 5:
                    continue
                entry_date = entry_candidates[0]

            if entry_date > oot_end:
                continue

            # Entry price with slippage
            raw_entry = float(ticker_prices.loc[entry_date])
            entry_price = raw_entry * (1 + SLIPPAGE_PCT)

            # Position size
            shares = int(MAX_PER_TRADE / entry_price) if entry_price > 0 else 0
            if shares == 0:
                # For high-priced stocks, allow fractional (Robinhood supports it)
                shares_frac = MAX_PER_TRADE / entry_price
                if shares_frac < 0.001:
                    continue
                position_value = MAX_PER_TRADE
            else:
                position_value = shares * entry_price

            # Exit date
            exit_target_idx = ticker_prices.index.get_indexer([entry_date], method="nearest")[0] + hold_days
            if exit_target_idx >= len(ticker_prices):
                exit_date = ticker_prices.index[-1]
            else:
                exit_date = ticker_prices.index[exit_target_idx]

            raw_exit = float(ticker_prices.loc[exit_date])
            exit_price = raw_exit * (1 - SLIPPAGE_PCT)

            pct_return = (exit_price - entry_price) / entry_price

            # Regime
            regime = "unknown"
            if entry_date in spy_sma200.index and entry_date in spy_close.index:
                sp = float(spy_close.loc[entry_date])
                sm = float(spy_sma200.loc[entry_date])
                if not np.isnan(sm):
                    regime = "bear" if sp < sm else "bull"

            trades.append({
                "ticker": ticker,
                "signal_date": dt.strftime("%Y-%m-%d"),
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "hold_days": (exit_date - entry_date).days,
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "pct_return": round(pct_return, 6),
                "position_value": round(position_value, 2),
                "regime": regime,
            })

    # Enforce max concurrent positions
    trades = enforce_max_concurrent(trades)

    return trades


def enforce_max_concurrent(trades):
    """Remove trades that would exceed max concurrent positions."""
    if not trades:
        return trades

    # Sort by entry date
    trades.sort(key=lambda t: t["entry_date"])
    accepted = []

    for trade in trades:
        entry = pd.Timestamp(trade["entry_date"])
        exit_dt = pd.Timestamp(trade["exit_date"])

        # Count how many accepted trades are open at entry
        concurrent = sum(
            1 for a in accepted
            if pd.Timestamp(a["entry_date"]) <= entry <= pd.Timestamp(a["exit_date"])
        )

        if concurrent < MAX_CONCURRENT:
            accepted.append(trade)

    return accepted


# ── Analytics ─────────────────────────────────────────────────────────────

def compute_metrics(trades):
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "max_dd": 0, "total_return": 0, "ann_return": 0}

    returns = np.array([t["pct_return"] for t in trades])
    n = len(returns)

    total_ret = np.prod(1 + returns) - 1
    wr = np.sum(returns > 0) / n

    # Trades per year estimate
    avg_hold = np.mean([t["hold_days"] for t in trades])
    trades_per_year = 252 / max(avg_hold, 1)

    # Daily-equivalent risk-free rate per trade
    rf_per_trade = RF_ANNUAL * (avg_hold / 252)

    # Excess returns
    excess = returns - rf_per_trade

    # Sharpe
    if np.std(excess) > 0:
        sharpe = (np.mean(excess) / np.std(excess)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = excess[excess < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(excess) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    # Max Drawdown on equity curve (position-sized)
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd)

    # Annualized return
    first_entry = min(t["entry_date"] for t in trades)
    last_exit = max(t["exit_date"] for t in trades)
    days_span = (pd.Timestamp(last_exit) - pd.Timestamp(first_entry)).days
    years = max(days_span / 365.25, 0.5)
    ann_return = (1 + total_ret) ** (1 / years) - 1

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(min(pf, 99), 3),
        "wr": round(wr, 4),
        "max_dd": round(max_dd, 4),
        "total_return": round(total_ret, 4),
        "ann_return": round(ann_return, 4),
    }


def compute_regime_metrics(trades):
    bull = [t for t in trades if t["regime"] == "bull"]
    bear = [t for t in trades if t["regime"] == "bear"]
    bull_m = compute_metrics(bull) if bull else {"sharpe": 0}
    bear_m = compute_metrics(bear) if bear else {"sharpe": 0}

    s_bull = bull_m["sharpe"]
    s_bear = bear_m["sharpe"]
    denom = max(abs(s_bull), abs(s_bear))
    gap = abs(s_bull - s_bear) / denom if denom > 0 else 0.0

    return s_bull, s_bear, round(gap, 4)


def permutation_test(trades, n_perms=N_PERMUTATIONS, seed=RANDOM_SEED):
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["pct_return"] for t in trades])
    avg_hold = np.mean([t["hold_days"] for t in trades])
    rf_per_trade = RF_ANNUAL * (avg_hold / 252)
    excess = returns - rf_per_trade
    trades_per_year = 252 / max(avg_hold, 1)

    if np.std(excess) > 0:
        observed = (np.mean(excess) / np.std(excess)) * np.sqrt(trades_per_year)
    else:
        observed = 0

    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perms):
        perm = excess * rng.choice([-1, 1], size=len(excess))
        if np.std(perm) > 0:
            perm_sharpe = (np.mean(perm) / np.std(perm)) * np.sqrt(trades_per_year)
        else:
            perm_sharpe = 0
        if perm_sharpe >= observed:
            count += 1

    return round(count / n_perms, 4)


def validate_gates(metrics, trades):
    gates = {}

    # 1. Sharpe > 0.5
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5

    # 2. Permutation p < 0.05
    p_val = permutation_test(trades)
    gates["perm_p_lt_0.05"] = p_val < 0.05
    perm_p = p_val

    # 3. Regime gap < 0.5
    _, _, gap = compute_regime_metrics(trades)
    gates["regime_gap_lt_0.5"] = gap < 0.5

    # 4. MaxDD > -50%
    gates["maxdd_gt_neg50"] = metrics["max_dd"] > -0.50

    # 5. Min 20 trades
    gates["min_20_trades"] = metrics["n_trades"] >= 20

    passed = sum(1 for v in gates.values() if v)
    return passed, gates, perm_p, gap


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("BUYBACK ANNOUNCEMENT BACKTEST (Price+Volume Proxy)")
    log("=" * 70)

    prices = download_prices(UNIVERSE)
    volumes = download_volumes(UNIVERSE)
    spy = download_spy()

    variants = {
        "A": "Basic buyback signal (5% of 52w low + vol surge, 40d hold)",
        "B": "Extended hold (same signal, 60d hold)",
        "C": "Tight entry (3% of 52w low, 40d hold)",
        "D": "Trend filter (+ 200-SMA rising, 40d hold)",
        "E": "Sector washout (3+ stocks trigger in 10d, 40d hold)",
        "F": "Momentum confirmation (wait 3 up days, 40d hold)",
    }

    results = {}

    for var_key, var_desc in variants.items():
        log(f"\n--- Variant {var_key}: {var_desc} ---")

        trades = compute_signals(prices, volumes, spy, variant=var_key)
        log(f"  Trades: {len(trades)}")

        if not trades:
            results[var_key] = {
                "description": var_desc,
                "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "n_trades": 0, "max_dd": 0, "total_return": 0,
                "perm_p_value": 1.0, "regime_bull_sharpe": 0,
                "regime_bear_sharpe": 0, "regime_gap": 0,
                "gates_passed": 0, "ann_return": 0,
            }
            continue

        metrics = compute_metrics(trades)
        bull_sharpe, bear_sharpe, regime_gap = compute_regime_metrics(trades)
        gates_passed, gate_detail, perm_p, _ = validate_gates(metrics, trades)

        results[var_key] = {
            "description": var_desc,
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "pf": metrics["pf"],
            "wr": metrics["wr"],
            "n_trades": metrics["n_trades"],
            "max_dd": metrics["max_dd"],
            "total_return": metrics["total_return"],
            "perm_p_value": perm_p,
            "regime_bull_sharpe": round(bull_sharpe, 3),
            "regime_bear_sharpe": round(bear_sharpe, 3),
            "regime_gap": regime_gap,
            "gates_passed": gates_passed,
            "ann_return": metrics["ann_return"],
        }

        log(f"  Sharpe:    {metrics['sharpe']}")
        log(f"  Sortino:   {metrics['sortino']}")
        log(f"  PF:        {metrics['pf']}")
        log(f"  WR:        {metrics['wr']:.1%}")
        log(f"  MaxDD:     {metrics['max_dd']:.2%}")
        log(f"  Total Ret: {metrics['total_return']:.2%}")
        log(f"  Ann Ret:   {metrics['ann_return']:.2%}")
        log(f"  Perm p:    {perm_p}")
        log(f"  Regime:    bull={bull_sharpe:.3f} bear={bear_sharpe:.3f} gap={regime_gap:.3f}")
        log(f"  Gates:     {gates_passed}/5 {'PASS' if gates_passed == 5 else 'FAIL'}")
        for gname, gval in gate_detail.items():
            log(f"    {gname}: {'PASS' if gval else 'FAIL'}")

    # ── Summary Table ─────────────────────────────────────────────────────
    log(f"\n{'='*90}")
    log(f"{'Var':<4} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'N':>4} {'MaxDD':>7} {'TotRet':>8} {'AnnRet':>8} {'Perm-p':>7} {'Gates':>6}")
    log(f"{'-'*90}")
    for vk in ["A", "B", "C", "D", "E", "F"]:
        r = results[vk]
        log(f"{vk:<4} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['pf']:>6.2f} {r['wr']:>6.1%} "
            f"{r['n_trades']:>4} {r['max_dd']:>7.2%} {r['total_return']:>8.2%} {r['ann_return']:>8.2%} "
            f"{r['perm_p_value']:>7.4f} {r['gates_passed']:>3}/5")
    log(f"{'='*90}")

    # Save
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    main()
