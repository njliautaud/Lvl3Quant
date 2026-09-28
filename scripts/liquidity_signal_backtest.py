#!/usr/bin/env python3
"""
Liquidity-Based Entry Signals on Quality Stocks — Backtest
Variants A-F testing volume/liquidity patterns for mean-reversion timing.
"""

import json
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─── CONFIG ───
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2021-06-01"  # extra lookback for indicators
TRADE_START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 2
PERMUTATION_N = 1000

# ─── DATA ───
print("Downloading data...")
raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)
# Handle multi-level columns from yfinance
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"]
    high = raw["High"]
    low = raw["Low"]
    volume = raw["Volume"]
    opn = raw["Open"]
else:
    # Single ticker fallback
    close = raw[["Close"]].rename(columns={"Close": UNIVERSE[0]})
    high = raw[["High"]].rename(columns={"High": UNIVERSE[0]})
    low = raw[["Low"]].rename(columns={"Low": UNIVERSE[0]})
    volume = raw[["Volume"]].rename(columns={"Volume": UNIVERSE[0]})
    opn = raw[["Open"]].rename(columns={"Open": UNIVERSE[0]})

print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")

# ─── INDICATORS ───
print("Computing indicators...")
ret = close.pct_change()
high20 = close.rolling(20).max()
pct_below_high = (close - high20) / high20
vol_avg20 = volume.rolling(20).mean()
dollar_vol = close * volume
dollar_vol_avg20 = dollar_vol.rolling(20).mean()

# RSI(14)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rsi = close.apply(lambda col: compute_rsi(col, 14))

# Amihud illiquidity: avg(|return|/volume) over 20 days
abs_ret = ret.abs()
amihud_daily = abs_ret / volume.replace(0, np.nan)
amihud_20 = amihud_daily.rolling(20).mean()
amihud_60_avg = amihud_20.rolling(60).mean()

# High-low spread proxy
hl_spread = (high - low) / ((high + low) / 2)
hl_spread_avg60 = hl_spread.rolling(60).mean()

# VWAP proxy: cumulative (typical_price * volume) / cumulative volume, daily reset
# For daily bars, VWAP ~ typical price. We use a rolling ratio approach:
# "VWAP above close" means typical_price > close, accumulated over days.
typical_price = (high + low + close) / 3
vwap_above_close = (typical_price - close) / close  # positive = VWAP above close

# Count consecutive days where VWAP > close by >2%
vwap_above_2pct = vwap_above_close > 0.02

def consecutive_true(series):
    """Count consecutive True values ending at each point."""
    result = pd.Series(0, index=series.index)
    count = 0
    for i in range(len(series)):
        if series.iloc[i]:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result

# Count consecutive days dollar_vol < avg
dv_below_avg = dollar_vol < dollar_vol_avg20

# ─── SIGNAL GENERATION ───
print("Generating signals...")
trade_mask = close.index >= pd.Timestamp(TRADE_START)

def get_signals_A():
    """Volume dry-up entry."""
    signals = []
    cond_below = pct_below_high < -0.05
    cond_vol = volume < (0.5 * vol_avg20)
    cond_rsi = rsi < 40
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        mask = cond_below[ticker].fillna(False) & cond_vol[ticker].fillna(False) & cond_rsi[ticker].fillna(False)
        for date in close.index[mask & trade_mask]:
            signals.append((date, ticker))
    return signals

def get_signals_B():
    """Volume spike reversal — buy day after capitulation."""
    signals = []
    drop_3pct = ret < -0.03
    vol_spike = volume > (2 * vol_avg20)
    cap_day = drop_3pct & vol_spike
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        cap = cap_day[ticker].fillna(False)
        vol_t = volume[ticker]
        avg_t = vol_avg20[ticker]
        cap_dates = close.index[cap & trade_mask]
        for d in cap_dates:
            idx = close.index.get_loc(d)
            if idx + 1 < len(close.index):
                next_day = close.index[idx + 1]
                if pd.notna(vol_t.iloc[idx + 1]) and pd.notna(avg_t.iloc[idx + 1]):
                    if vol_t.iloc[idx + 1] < avg_t.iloc[idx + 1]:
                        signals.append((next_day, ticker))
    return signals

def get_signals_C():
    """Amihud illiquidity spike."""
    signals = []
    amihud_spike = amihud_20 > (2 * amihud_60_avg)
    cond_below = pct_below_high < -0.05
    combined = amihud_spike & cond_below
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        mask = combined[ticker].fillna(False)
        for date in close.index[mask & trade_mask]:
            signals.append((date, ticker))
    return signals

def get_signals_D():
    """Dollar volume mean reversion — 3+ consecutive days below avg + dip + RSI."""
    signals = []
    cond_below = pct_below_high < -0.05
    cond_rsi = rsi < 40
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        consec = consecutive_true(dv_below_avg[ticker].fillna(False))
        mask = (consec >= 3) & cond_below[ticker].fillna(False) & cond_rsi[ticker].fillna(False)
        for date in close.index[mask & trade_mask]:
            signals.append((date, ticker))
    return signals

def get_signals_E():
    """Volume-weighted dip — VWAP above close for 3+ days + below high."""
    signals = []
    cond_below = pct_below_high < -0.05
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        consec = consecutive_true(vwap_above_2pct[ticker].fillna(False))
        mask = (consec >= 3) & cond_below[ticker].fillna(False)
        for date in close.index[mask & trade_mask]:
            signals.append((date, ticker))
    return signals

def get_signals_F():
    """Bid-ask proxy — narrowing spread during dip."""
    signals = []
    spread_narrow = hl_spread < hl_spread_avg60
    cond_below = pct_below_high < -0.05
    cond_rsi = rsi < 40
    combined = spread_narrow & cond_below & cond_rsi
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        mask = combined[ticker].fillna(False)
        for date in close.index[mask & trade_mask]:
            signals.append((date, ticker))
    return signals


# ─── BACKTEST ENGINE ───
def run_backtest(signals, label=""):
    """
    Run backtest with position limits.
    Returns dict of metrics + equity curve.
    """
    if not signals:
        return None

    # Sort signals by date
    signals = sorted(signals, key=lambda x: x[0])

    trades = []
    active_positions = []  # list of (exit_date, ticker)
    equity = CAPITAL
    equity_curve = [(pd.Timestamp(TRADE_START), CAPITAL)]
    peak = CAPITAL

    for entry_date, ticker in signals:
        # Remove expired positions
        active_positions = [(ed, tk) for ed, tk in active_positions if ed > entry_date]

        if len(active_positions) >= MAX_CONCURRENT:
            continue

        entry_idx = close.index.get_loc(entry_date)
        exit_idx = min(entry_idx + HOLD_DAYS, len(close.index) - 1)
        exit_date = close.index[exit_idx]

        entry_price = close[ticker].iloc[entry_idx]
        exit_price = close[ticker].iloc[exit_idx]

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        # Position sizing
        shares_by_capital = int(MAX_PER_TRADE / entry_price)
        if shares_by_capital < 1:
            continue

        # Slippage
        entry_cost = entry_price * (1 + SLIPPAGE_BPS / 10000)
        exit_cost = exit_price * (1 - SLIPPAGE_BPS / 10000)

        pnl = (exit_cost - entry_cost) * shares_by_capital
        pnl_pct = (exit_cost - entry_cost) / entry_cost

        trades.append({
            "entry_date": str(pd.Timestamp(entry_date).date()),
            "exit_date": str(pd.Timestamp(exit_date).date()),
            "ticker": ticker,
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "shares": shares_by_capital,
            "pnl": round(float(pnl), 2),
            "pnl_pct": round(float(pnl_pct), 4),
        })

        equity += pnl
        equity_curve.append((exit_date, equity))
        peak = max(peak, equity)

        active_positions.append((exit_date, ticker))

    if not trades:
        return None

    # ─── METRICS ───
    trade_returns = [t["pnl_pct"] for t in trades]
    trade_pnls = [t["pnl"] for t in trades]
    n_trades = len(trades)

    total_pnl = sum(trade_pnls)
    win_rate = sum(1 for r in trade_returns if r > 0) / n_trades
    avg_return = np.mean(trade_returns)
    std_return = np.std(trade_returns) if n_trades > 1 else 1e-9

    # Annualize: ~252/HOLD_DAYS trades per year per slot
    trades_per_year = 252 / HOLD_DAYS
    sharpe = (avg_return / max(std_return, 1e-9)) * np.sqrt(trades_per_year)

    downside_returns = [r for r in trade_returns if r < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-9
    sortino = (avg_return / max(downside_std, 1e-9)) * np.sqrt(trades_per_year)

    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    profit_factor = gross_profit / max(gross_loss, 1e-9)

    # Max drawdown from equity curve
    eq_series = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
    eq_peak = eq_series.cummax()
    drawdown = (eq_series - eq_peak) / eq_peak
    max_dd = float(drawdown.min())

    # Regime analysis: split by SPY performance
    spy_data = close.get("AAPL")  # Use AAPL as proxy if SPY not in universe
    # Actually download SPY for regime
    spy_raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
    spy_close = spy_raw["Close"]
    spy_ret_20d = spy_close.pct_change(20)

    bull_trades = []
    bear_trades = []
    for t in trades:
        edate = pd.Timestamp(t["entry_date"])
        if edate in spy_ret_20d.index:
            sr = spy_ret_20d.loc[edate]
            val = float(sr.iloc[0]) if hasattr(sr, 'iloc') else float(sr)
            if val > 0:
                bull_trades.append(t["pnl_pct"])
            else:
                bear_trades.append(t["pnl_pct"])

    bull_sharpe = (np.mean(bull_trades) / max(np.std(bull_trades), 1e-9)) * np.sqrt(trades_per_year) if len(bull_trades) > 2 else 0
    bear_sharpe = (np.mean(bear_trades) / max(np.std(bear_trades), 1e-9)) * np.sqrt(trades_per_year) if len(bear_trades) > 2 else 0
    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)

    return {
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / CAPITAL * 100, 2),
        "win_rate": round(win_rate, 4),
        "avg_return_pct": round(avg_return * 100, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "trades": trades,
        "trade_returns": trade_returns,
    }


# ─── PERMUTATION TEST ───
def permutation_test(signals, actual_sharpe, n_perms=PERMUTATION_N):
    """Shuffle entry dates randomly, recompute Sharpe, get p-value."""
    if not signals:
        return 1.0

    all_dates = close.index[close.index >= pd.Timestamp(TRADE_START)]
    tickers = list(set(s[1] for s in signals))
    n_signals = len(signals)
    count_better = 0

    for _ in range(n_perms):
        random_signals = [
            (np.random.choice(all_dates), np.random.choice(tickers))
            for _ in range(n_signals)
        ]
        result = run_backtest(random_signals)
        if result and result["sharpe"] >= actual_sharpe:
            count_better += 1

    return (count_better + 1) / (n_perms + 1)


# ─── 5-GATE VALIDATION ───
def validate_5gate(result, perm_p):
    """Apply 5-gate validation."""
    if result is None:
        return {"pass": False, "reason": "No trades"}

    gates = {
        "G1_sharpe_gt_0.5": result["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": perm_p < 0.05,
        "G3_regime_gap_lt_0.5": result["regime_gap"] < 0.5,
        "G4_max_dd_gt_neg50": result["max_drawdown_pct"] > -50,
        "G5_min_20_trades": result["n_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ─── RUN ALL VARIANTS ───
variant_funcs = {
    "A_volume_dryup": get_signals_A,
    "B_volume_spike_reversal": get_signals_B,
    "C_amihud_illiquidity": get_signals_C,
    "D_dollar_volume_mr": get_signals_D,
    "E_vwap_dip": get_signals_E,
    "F_bidask_proxy": get_signals_F,
}

results = {}
for name, func in variant_funcs.items():
    print(f"\n{'='*60}")
    print(f"Running variant: {name}")
    print(f"{'='*60}")

    signals = func()
    print(f"  Raw signals generated: {len(signals)}")

    result = run_backtest(signals, label=name)

    if result is None:
        print(f"  NO TRADES — skipping")
        results[name] = {"status": "no_trades", "gates": {"all_pass": False}}
        continue

    print(f"  Trades: {result['n_trades']}")
    print(f"  Total PnL: ${result['total_pnl']}")
    print(f"  Total Return: {result['total_return_pct']}%")
    print(f"  Sharpe: {result['sharpe']}")
    print(f"  Sortino: {result['sortino']}")
    print(f"  PF: {result['profit_factor']}")
    print(f"  WR: {result['win_rate']:.1%}")
    print(f"  Max DD: {result['max_drawdown_pct']}%")
    print(f"  Regime gap: {result['regime_gap']}")

    # Permutation test
    print(f"  Running permutation test ({PERMUTATION_N} shuffles)...")
    perm_p = permutation_test(signals, result["sharpe"])
    print(f"  Permutation p-value: {perm_p:.4f}")

    gates = validate_5gate(result, perm_p)
    print(f"  5-Gate results: {gates}")

    # Clean up for JSON
    result_clean = {k: v for k, v in result.items() if k != "trade_returns"}
    result_clean["permutation_p"] = round(perm_p, 4)
    result_clean["gates"] = gates

    results[name] = result_clean

# ─── SUMMARY ───
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Variant':<30} {'Trades':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>7} {'WR':>7} {'DD%':>8} {'Perm-p':>8} {'Pass':>6}")
print("-" * 100)

for name, res in results.items():
    if res.get("status") == "no_trades":
        print(f"{name:<30} {'N/A':>7} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'N/A':>7} {'N/A':>8} {'N/A':>8} {'FAIL':>6}")
        continue
    gates = res.get("gates", {})
    passed = "PASS" if gates.get("all_pass", False) else "FAIL"
    print(f"{name:<30} {res['n_trades']:>7} {res['sharpe']:>8.3f} {res['sortino']:>8.3f} {res['profit_factor']:>7.2f} {res['win_rate']:>6.1%} {res['max_drawdown_pct']:>7.2f}% {res.get('permutation_p', 1):>8.4f} {passed:>6}")

# ─── SAVE RESULTS ───
output_path = "/home/jupiter/Lvl3Quant/data/liquidity_signal_results.json"

# Strip trade-level detail for compactness in JSON, keep top 5 trades per variant
for name, res in results.items():
    if "trades" in res and len(res["trades"]) > 0:
        res["sample_trades"] = res["trades"][:5]
        res["all_trades_count"] = len(res["trades"])
        del res["trades"]

output = {
    "strategy": "Liquidity-Based Entry Signals on Quality Stocks",
    "run_date": str(datetime.datetime.now()),
    "parameters": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "hold_days": HOLD_DAYS,
        "slippage_bps": SLIPPAGE_BPS,
        "universe_size": len(UNIVERSE),
        "period": f"{TRADE_START} to {END}",
    },
    "variants": results,
}

with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
