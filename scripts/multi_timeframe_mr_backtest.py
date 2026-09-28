#!/usr/bin/env python3
"""
Multi-Timeframe Mean Reversion Confirmation Backtest
Tests whether requiring mean reversion signals on multiple timeframes improves signal quality.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
FETCH_START = "2020-06-01"  # extra history for indicators
STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.02 / 100  # 0.02% each way
PERM_ITERATIONS = 1000
OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/multi_timeframe_mr_results.json")
SPY_TICKER = "SPY"

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
tickers = UNIVERSE + [SPY_TICKER]
raw = yf.download(tickers, start=FETCH_START, end=OOT_END, auto_adjust=True, progress=False)

# Handle multi-level columns from yf.download
if isinstance(raw.columns, pd.MultiIndex):
    close_df = raw["Close"].copy()
else:
    close_df = raw[["Close"]].copy()
    close_df.columns = tickers

# Also get Open, High, Low for each ticker
open_df = raw["Open"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Open"]].copy()
high_df = raw["High"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["High"]].copy()
low_df = raw["Low"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Low"]].copy()

close_df = close_df.ffill()
open_df = open_df.ffill()
high_df = high_df.ffill()
low_df = low_df.ffill()

# SPY for regime
spy_close = close_df[SPY_TICKER].dropna()
spy_sma200 = spy_close.rolling(200).mean()

print(f"Data range: {close_df.index[0].date()} to {close_df.index[-1].date()}")
print(f"Tickers loaded: {len(UNIVERSE)}")

# ── Indicator Helpers ───────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def resample_weekly(daily_close):
    """Resample daily close to weekly (last trading day of each week)."""
    return daily_close.resample("W-FRI").last().dropna()

def compute_monthly_returns(daily_close):
    """Compute monthly returns (last trading day of each month)."""
    monthly = daily_close.resample("ME").last().dropna()
    return monthly.pct_change()


# ── Pre-compute all indicators ──────────────────────────────────────────────
print("Computing indicators...")

daily_rsi = {}
daily_high20 = {}
daily_green = {}  # True if close > open
weekly_rsi = {}
weekly_high10 = {}
weekly_close = {}
monthly_ret = {}

for tkr in UNIVERSE:
    dc = close_df[tkr].dropna()
    do = open_df[tkr].dropna()

    # Daily
    daily_rsi[tkr] = compute_rsi(dc, 14)
    daily_high20[tkr] = dc.rolling(20).max()
    daily_green[tkr] = (dc > do)

    # Weekly
    wc = resample_weekly(dc)
    weekly_close[tkr] = wc
    weekly_rsi[tkr] = compute_rsi(wc, 14)
    weekly_high10[tkr] = wc.rolling(10).max()

    # Monthly
    monthly_ret[tkr] = compute_monthly_returns(dc)


def get_weekly_value_for_date(weekly_series, date):
    """Get the most recent weekly value as of a given daily date."""
    mask = weekly_series.index <= date
    if mask.any():
        return weekly_series.loc[mask].iloc[-1]
    return np.nan

def get_monthly_ret_for_date(monthly_series, date):
    """Get last month's return as of date (i.e., the monthly return ending before this month)."""
    # Find the most recent completed month
    target_month_end = (date.replace(day=1) - dt.timedelta(days=1))
    # pd Timestamp
    target_month_end = pd.Timestamp(target_month_end)
    mask = monthly_series.index <= target_month_end
    if mask.any():
        return monthly_series.loc[mask].iloc[-1]
    return np.nan


# ── Signal Detection ────────────────────────────────────────────────────────

def check_first_green_after_3red(tkr, idx, dates):
    """Check if date at idx is first green day after 3+ consecutive red days."""
    if idx < 3:
        return False
    if not daily_green[tkr].get(dates[idx], False):
        return False
    # Need at least 3 consecutive red days before
    for k in range(1, 4):
        if daily_green[tkr].get(dates[idx - k], True):  # if green, fail
            return False
    return True


def generate_signals(variant):
    """Generate trade signals for a variant. Returns list of (date, ticker, hold_days)."""
    oot_mask = (close_df.index >= OOT_START) & (close_df.index <= OOT_END)
    oot_dates = close_df.index[oot_mask].tolist()
    all_dates = close_df.index.tolist()

    signals = []

    for tkr in UNIVERSE:
        dc = close_df[tkr]
        for i_global, date in enumerate(oot_dates):
            # Find index in all_dates
            try:
                idx = all_dates.index(date)
            except ValueError:
                continue
            if idx < 25:
                continue

            price = dc.get(date, np.nan)
            if np.isnan(price):
                continue

            if variant == "E":
                # Weekly only: weekly RSI < 35 + price > 7% below 10-week high
                w_rsi_val = get_weekly_value_for_date(weekly_rsi[tkr], date)
                w_high_val = get_weekly_value_for_date(weekly_high10[tkr], date)
                if np.isnan(w_rsi_val) or np.isnan(w_high_val):
                    continue
                dip_from_whigh = (price - w_high_val) / w_high_val

                if w_rsi_val < 35 and dip_from_whigh < -0.07:
                    # Enter on first trading day of next week
                    # Find next Monday (or next trading day)
                    next_dates = [d for d in all_dates if d > date]
                    if not next_dates:
                        continue
                    # Find the start of next week
                    current_week = date.isocalendar()[1]
                    entry_date = None
                    for nd in next_dates:
                        if nd.isocalendar()[1] != current_week:
                            entry_date = nd
                            break
                    if entry_date is None:
                        continue
                    signals.append((entry_date, tkr, 15))
                continue

            # ── All other variants start with Daily Dual Signal D ──
            # 5% dip from 20d high
            h20 = daily_high20[tkr].get(date, np.nan)
            if np.isnan(h20) or h20 == 0:
                continue
            dip_from_20d = (price - h20) / h20
            if dip_from_20d >= -0.05:
                continue

            # RSI(14) < 35
            rsi_val = daily_rsi[tkr].get(date, np.nan)
            if np.isnan(rsi_val) or rsi_val >= 35:
                continue

            # First green after 3+ red
            if not check_first_green_after_3red(tkr, idx, all_dates):
                continue

            # ── Daily signal confirmed. Now check multi-timeframe filters ──
            hold_days = 10

            if variant == "A":
                pass  # baseline, no extra filter

            elif variant == "B":
                # Also require weekly RSI(14) < 40
                w_rsi_val = get_weekly_value_for_date(weekly_rsi[tkr], date)
                if np.isnan(w_rsi_val) or w_rsi_val >= 40:
                    continue

            elif variant == "C":
                # Also require price > 7% below 10-week high
                w_high_val = get_weekly_value_for_date(weekly_high10[tkr], date)
                if np.isnan(w_high_val):
                    continue
                dip_from_whigh = (price - w_high_val) / w_high_val
                if dip_from_whigh >= -0.07:
                    continue

            elif variant == "D":
                # Weekly RSI < 40 AND > 7% below 10-week high
                w_rsi_val = get_weekly_value_for_date(weekly_rsi[tkr], date)
                w_high_val = get_weekly_value_for_date(weekly_high10[tkr], date)
                if np.isnan(w_rsi_val) or np.isnan(w_high_val):
                    continue
                dip_from_whigh = (price - w_high_val) / w_high_val
                if w_rsi_val >= 40 or dip_from_whigh >= -0.07:
                    continue
                hold_days = 15

            elif variant == "F":
                # Require negative monthly return last month
                m_ret = get_monthly_ret_for_date(monthly_ret[tkr], date)
                if np.isnan(m_ret) or m_ret >= 0:
                    continue

            signals.append((date, tkr, hold_days))

    return signals


# ── Backtester ──────────────────────────────────────────────────────────────

def run_backtest(signals, label=""):
    """Run backtest with position sizing, concurrency limits, slippage."""
    if not signals:
        return {
            "label": label, "trades": 0, "total_return_pct": 0, "sharpe": 0,
            "sortino": 0, "win_rate": 0, "profit_factor": 0, "max_drawdown_pct": 0,
            "trade_returns": [],
        }

    # Sort signals by date
    signals = sorted(signals, key=lambda x: x[0])

    all_dates = close_df.index.tolist()
    capital = STARTING_CAPITAL
    peak_capital = capital
    max_dd = 0.0
    active_trades = []  # (exit_date_idx, ticker, shares, entry_price)
    trade_returns = []
    trade_dates = []

    for entry_date, tkr, hold_days in signals:
        # Check concurrency
        # Remove expired trades
        try:
            entry_idx = all_dates.index(entry_date)
        except ValueError:
            continue

        active_trades = [t for t in active_trades if t[0] > entry_idx]

        if len(active_trades) >= MAX_CONCURRENT:
            continue

        entry_price = close_df[tkr].get(entry_date, np.nan)
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Apply slippage on entry
        entry_cost = entry_price * (1 + SLIPPAGE_PCT)

        # Position size
        alloc = min(MAX_PER_TRADE, capital * 0.95)  # keep 5% buffer
        if alloc < 1:
            continue
        shares = alloc / entry_cost

        # Find exit date (hold_days trading days later)
        exit_idx = min(entry_idx + hold_days, len(all_dates) - 1)
        exit_date = all_dates[exit_idx]
        exit_price = close_df[tkr].get(exit_date, np.nan)
        if np.isnan(exit_price):
            continue

        # Apply slippage on exit
        exit_proceeds = exit_price * (1 - SLIPPAGE_PCT)

        pnl = shares * (exit_proceeds - entry_cost)
        ret = (exit_proceeds - entry_cost) / entry_cost
        trade_returns.append(ret)
        trade_dates.append(entry_date)

        capital += pnl
        peak_capital = max(peak_capital, capital)
        dd = (capital - peak_capital) / peak_capital if peak_capital > 0 else 0
        max_dd = min(max_dd, dd)

        active_trades.append((exit_idx, tkr, shares, entry_cost))

    # Compute metrics
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    n_trades = len(trade_returns)
    wins = (tr > 0).sum()
    losses = (tr <= 0).sum()
    win_rate = wins / n_trades if n_trades > 0 else 0
    total_ret = (capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    gross_profit = tr[tr > 0].sum() if (tr > 0).any() else 0
    gross_loss = abs(tr[tr <= 0].sum()) if (tr <= 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Annualized Sharpe / Sortino (assume ~50 trades/year as normalizer, or use actual)
    if n_trades > 1 and tr.std() > 0:
        # Per-trade Sharpe, annualized assuming ~252/avg_hold trades per year
        avg_hold = 12  # approximate
        trades_per_year = 252 / avg_hold
        sharpe = (tr.mean() / tr.std()) * np.sqrt(trades_per_year)
        downside = tr[tr < 0].std() if (tr < 0).any() else 1e-9
        sortino = (tr.mean() / downside) * np.sqrt(trades_per_year) if downside > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    return {
        "label": label,
        "trades": n_trades,
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "trade_returns": trade_returns,
        "trade_dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in trade_dates],
    }


# ── Regime Analysis ─────────────────────────────────────────────────────────

def regime_analysis(result):
    """Split trades into bull/bear regime and compute per-regime Sharpe."""
    if result["trades"] < 2:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0,
                "bull_trades": 0, "bear_trades": 0}

    bull_rets = []
    bear_rets = []
    for ret, dstr in zip(result["trade_returns"], result["trade_dates"]):
        d = pd.Timestamp(dstr)
        sma = spy_sma200.get(d, np.nan)
        spy_p = spy_close.get(d, np.nan)
        if np.isnan(sma) or np.isnan(spy_p):
            bull_rets.append(ret)  # default to bull
            continue
        if spy_p > sma:
            bull_rets.append(ret)
        else:
            bear_rets.append(ret)

    avg_hold = 12
    trades_per_year = 252 / avg_hold

    def regime_sharpe(rets):
        r = np.array(rets)
        if len(r) < 2 or r.std() == 0:
            return 0
        return (r.mean() / r.std()) * np.sqrt(trades_per_year)

    bs = regime_sharpe(bull_rets)
    brs = regime_sharpe(bear_rets)
    gap = abs(bs - brs) / max(abs(bs), abs(brs), 1e-9)

    return {
        "bull_sharpe": round(bs, 3),
        "bear_sharpe": round(brs, 3),
        "regime_gap": round(gap, 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
    }


# ── Permutation Test ────────────────────────────────────────────────────────

def permutation_test(trade_returns, n_iter=PERM_ITERATIONS):
    """Permutation test: shuffle returns to get p-value for mean > 0."""
    if len(trade_returns) < 5:
        return 1.0
    tr = np.array(trade_returns)
    obs_mean = tr.mean()
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_iter):
        shuffled = tr * rng.choice([-1, 1], size=len(tr))
        if shuffled.mean() >= obs_mean:
            count += 1
    return round(count / n_iter, 4)


# ── 5-Gate Validation ───────────────────────────────────────────────────────

def five_gate(result, regime, perm_p):
    """Sharpe>0.5, perm p<0.05, regime gap<0.5, MDD>-50%, trades>=20."""
    gates = {
        "sharpe_gt_0.5": result["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "mdd_gt_neg50": result["max_drawdown_pct"] > -50,
        "trades_gte_20": result["trades"] >= 20,
    }
    gates["pass_all"] = all(gates.values())
    return gates


# ── Main ────────────────────────────────────────────────────────────────────

VARIANTS = {
    "A": "Daily Only (Baseline)",
    "B": "Daily + Weekly RSI",
    "C": "Daily + Weekly Dip",
    "D": "Daily + Weekly Both",
    "E": "Weekly Only",
    "F": "Monthly Confluence",
}

results = {}

for var_key, var_name in VARIANTS.items():
    print(f"\n{'='*60}")
    print(f"Variant {var_key}: {var_name}")
    print(f"{'='*60}")

    sigs = generate_signals(var_key)
    print(f"  Signals generated: {len(sigs)}")

    bt = run_backtest(sigs, label=f"{var_key}: {var_name}")
    regime = regime_analysis(bt)
    perm_p = permutation_test(bt["trade_returns"])
    gates = five_gate(bt, regime, perm_p)

    print(f"  Trades: {bt['trades']}")
    print(f"  Total Return: {bt['total_return_pct']:.1f}%")
    print(f"  Sharpe: {bt['sharpe']:.3f}  |  Sortino: {bt['sortino']:.3f}")
    print(f"  Win Rate: {bt['win_rate']:.1%}  |  PF: {bt['profit_factor']:.2f}")
    print(f"  Max DD: {bt['max_drawdown_pct']:.1f}%")
    print(f"  Bull Sharpe: {regime['bull_sharpe']:.3f} ({regime['bull_trades']} trades)")
    print(f"  Bear Sharpe: {regime['bear_sharpe']:.3f} ({regime['bear_trades']} trades)")
    print(f"  Regime Gap: {regime['regime_gap']:.3f}")
    print(f"  Perm p-value: {perm_p:.4f}")
    print(f"  5-Gate: {'PASS' if gates['pass_all'] else 'FAIL'} — {gates}")

    results[var_key] = {
        "variant": var_key,
        "name": var_name,
        "metrics": {
            "sharpe": bt["sharpe"],
            "sortino": bt["sortino"],
            "win_rate": round(bt["win_rate"] * 100, 1),
            "profit_factor": bt["profit_factor"],
            "max_drawdown_pct": bt["max_drawdown_pct"],
            "total_return_pct": bt["total_return_pct"],
            "trades": bt["trades"],
        },
        "regime": regime,
        "permutation_p_value": perm_p,
        "five_gate": gates,
    }

# ── Comparative Summary ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("COMPARATIVE SUMMARY")
print(f"{'='*70}")
print(f"{'Var':<4} {'Name':<28} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD':>7} {'Ret%':>7} {'Gate':>5}")
print("-" * 90)

baseline_sharpe = results["A"]["metrics"]["sharpe"]
for var_key in VARIANTS:
    r = results[var_key]
    m = r["metrics"]
    gate_str = "PASS" if r["five_gate"]["pass_all"] else "FAIL"
    print(f"{var_key:<4} {r['name']:<28} {m['trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
          f"{m['win_rate']:>5.1f}% {m['profit_factor']:>6.2f} {m['max_drawdown_pct']:>6.1f}% "
          f"{m['total_return_pct']:>6.1f}% {gate_str:>5}")

# ── Comparison to Baseline ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("MULTI-TIMEFRAME IMPACT vs BASELINE (A)")
print(f"{'='*70}")
for var_key in ["B", "C", "D", "E", "F"]:
    r = results[var_key]
    m = r["metrics"]
    sharpe_delta = m["sharpe"] - baseline_sharpe
    signal_reduction = (1 - m["trades"] / max(results["A"]["metrics"]["trades"], 1)) * 100
    print(f"  {var_key} ({r['name']}):")
    print(f"    Sharpe delta: {sharpe_delta:+.3f}  |  Signal reduction: {signal_reduction:.0f}%")
    if sharpe_delta > 0 and m["trades"] >= 10:
        print(f"    → Multi-timeframe filter IMPROVES quality (higher Sharpe)")
    elif m["trades"] < 10:
        print(f"    → Too few trades to conclude (filter too aggressive)")
    else:
        print(f"    → Multi-timeframe filter does NOT improve (or worsens) quality")

# ── Save Results ────────────────────────────────────────────────────────────
output = {
    "backtest": "multi_timeframe_mean_reversion_confirmation",
    "universe": UNIVERSE,
    "oot_period": f"{OOT_START} to {OOT_END}",
    "starting_capital": STARTING_CAPITAL,
    "max_per_trade": MAX_PER_TRADE,
    "max_concurrent": MAX_CONCURRENT,
    "slippage_pct_each_way": SLIPPAGE_PCT * 100,
    "timestamp": dt.datetime.now().isoformat(),
    "variants": results,
    "baseline_sharpe": baseline_sharpe,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_PATH}")
print("Done.")
